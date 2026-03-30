from __future__ import annotations

import logging
import time
import uuid
import warnings
from typing import Annotated, Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, Response

from tts_gateway.config import GatewayConfig, load_config
from tts_gateway.gateway import EngineAttempt, SynthesisError, TtsGateway

logger = logging.getLogger(__name__)


def _attempt_to_dict(attempt: EngineAttempt) -> dict[str, Any]:
  return {
    'chunkIndex': attempt.chunk_index,
    'engine': attempt.engine,
    'ok': attempt.ok,
    'durationMs': attempt.duration_ms,
    'error': attempt.error,
  }


def _attempts_payload(attempts: list[EngineAttempt]) -> list[dict[str, Any]]:
  return [_attempt_to_dict(attempt) for attempt in attempts]


def _attempts_log_fields(attempts: list[EngineAttempt]) -> dict[str, Any]:
  return {'engineAttempts': _attempts_payload(attempts)}


def _status_from_synthesis_error(exc: SynthesisError) -> int:
  if exc.unavailable:
    return 503
  if exc.timed_out:
    return 504
  return 502


def _configure_warning_filters() -> None:
  warnings.filterwarnings(
    'ignore',
    message='dropout option adds dropout after all but last recurrent layer.*',
    category=UserWarning,
    module=r'torch\.nn\.modules\.rnn',
  )
  warnings.filterwarnings(
    'ignore',
    message='`torch\\.nn\\.utils\\.weight_norm` is deprecated in favor of `torch\\.nn\\.utils\\.parametrizations\\.weight_norm`.',
    category=FutureWarning,
    module=r'torch\.nn\.utils\.weight_norm',
  )


def create_app(config: GatewayConfig | None = None) -> FastAPI:
  if config is None:
    config = load_config()
  gateway = TtsGateway(config)

  app = FastAPI(title='TTS Gateway', version='0.1.2')
  _configure_warning_filters()

  @app.middleware('http')
  async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
      'http-request',
      extra={
        'request_id': request_id,
        'method': request.method,
        'path': str(request.url.path),
        'status_code': response.status_code,
        'duration_ms': elapsed_ms,
      },
    )
    response.headers['x-request-id'] = request_id
    return response

  @app.get('/health')
  async def health() -> dict[str, Any]:
    return {
      'ok': True,
      'primaryEngine': config.primary_engine,
      'fallbackEngine': config.fallback_engine,
      'outputFormat': config.output_format,
      'chunkConcurrency': gateway.chunk_concurrency(),
      'chunkMaxChars': config.chunk_max_chars,
      'requestTimeoutSeconds': config.request_timeout_seconds,
      'engineTimeoutSeconds': config.engine_timeout_seconds,
      'defaultVoice': config.default_voice,
      'engineChain': gateway.engine_chain(),
      'engines': gateway.engine_info(),
    }

  @app.post('/warmup')
  async def warmup() -> dict[str, Any]:
    results = await gateway.warmup()
    return {'ok': True, 'engines': results}

  @app.post('/tts')
  async def tts(
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    request_start = time.perf_counter()
    normalized = text.strip()
    if not normalized:
      return JSONResponse(
        status_code=422, content={'error': 'Field "text" must not be empty'}
      )
    text_len = len(normalized)

    try:
      logger.info(
        'tts-request',
        extra={'text_len': text_len, 'max_chunks': config.chunk_max_chars},
      )
      result = await gateway.synthesize_with_timeout(normalized, voice=voice)
      elapsed_ms = int((time.perf_counter() - request_start) * 1000)
      logger.info(
        'tts-success',
        extra={
          'text_len': text_len,
          'chunks': result.chunks_total,
          'duration_ms': elapsed_ms,
          'content_type': result.content_type,
        },
      )
      logger.debug('tts-attempts', extra=_attempts_log_fields(result.attempts))
    except SynthesisError as exc:
      elapsed_ms = int((time.perf_counter() - request_start) * 1000)
      error_message = str(exc)
      status = _status_from_synthesis_error(exc)
      attempts_data = _attempts_payload(exc.attempts)
      logger.warning(
        'tts-failure',
        extra={
          'text_len': text_len,
          'duration_ms': elapsed_ms,
          'status': status,
          'error': error_message,
          'engineAttempts': attempts_data,
        },
      )
      return JSONResponse(
        status_code=status,
        content={
          'error': error_message,
          'engineAttempts': attempts_data,
        },
      )

    return Response(
      content=result.payload,
      media_type=result.content_type,
      headers={
        'X-TTS-Chunk-Count': str(result.chunks_total),
        'X-TTS-Primary-Engine': config.primary_engine,
      },
    )

  return app
