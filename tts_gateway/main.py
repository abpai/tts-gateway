from __future__ import annotations

import logging
import time
import uuid
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from tts_gateway.config import GatewayConfig, load_config
from tts_gateway.gateway import SynthesisError, TtsGateway
from tts_gateway.jobs.store import JobStore
from tts_gateway.jobs.worker import run_worker

logger = logging.getLogger(__name__)


def _status_from_synthesis_error(exc: SynthesisError) -> int:
  if exc.unavailable:
    return 503
  if exc.timed_out:
    return 504
  return 502


def _configure_warning_filters() -> None:
  # Suppress noisy torch/kokoro warnings that fire on every inference call
  warnings.filterwarnings('ignore', category=UserWarning, module=r'torch\.')
  warnings.filterwarnings('ignore', category=UserWarning, module=r'kokoro\.')
  warnings.filterwarnings('ignore', category=FutureWarning, module=r'torch\.')
  warnings.filterwarnings('ignore', message='.*unauthenticated requests.*HF.*')


def create_app(config: GatewayConfig | None = None) -> FastAPI:
  if config is None:
    config = load_config()
  gateway = TtsGateway(config)

  # Job infrastructure
  data_dir = Path(config.data_dir)
  artifacts_dir = data_dir / 'artifacts'
  artifacts_dir.mkdir(parents=True, exist_ok=True)
  store = JobStore(data_dir / 'jobs.db')

  worker_task: list = []  # mutable container for the background task

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    # Start embedded worker
    import asyncio

    task = asyncio.create_task(
      run_worker(
        store,
        gateway.engine_list(),
        artifacts_dir,
        poll_seconds=config.worker_poll_seconds,
        concurrency=gateway.chunk_concurrency(),
        engine_timeout=config.engine_timeout_seconds,
        ffmpeg_path=config.ffmpeg_path,
      )
    )
    worker_task.append(task)
    yield
    task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      pass
    store.close()

  app = FastAPI(title='TTS Gateway', version='0.2.0', lifespan=lifespan)
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

  # ---------------------------------------------------------------------------
  # Legacy buffered TTS (backward-compatible, returns audio bytes)
  # ---------------------------------------------------------------------------

  @app.post('/tts/sync')
  async def tts_sync(
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    return await _tts_buffered(text, voice)

  @app.post('/tts')
  async def tts(
    request: Request,
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    accept = request.headers.get('accept', '')
    # Only use job mode when client explicitly requests JSON
    if accept == 'application/json':
      return await _tts_job_submit(text, voice)
    # Default: legacy buffered audio (preserves backward compat)
    return await _tts_buffered(text, voice)

  async def _tts_buffered(text: str, voice: str | None) -> Response:
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
    except SynthesisError as exc:
      elapsed_ms = int((time.perf_counter() - request_start) * 1000)
      error_message = str(exc)
      status = _status_from_synthesis_error(exc)
      logger.warning(
        'tts-failure',
        extra={
          'text_len': text_len,
          'duration_ms': elapsed_ms,
          'status': status,
          'error': error_message,
        },
      )
      return JSONResponse(
        status_code=status,
        content={'error': error_message},
      )

    return Response(
      content=result.payload,
      media_type=result.content_type,
      headers={
        'X-TTS-Chunk-Count': str(result.chunks_total),
        'X-TTS-Primary-Engine': config.primary_engine,
      },
    )

  # ---------------------------------------------------------------------------
  # Job-based TTS (returns JSON, content-addressed)
  # ---------------------------------------------------------------------------

  async def _tts_job_submit(text: str, voice: str | None) -> Response:
    normalized = text.strip()
    if not normalized:
      return JSONResponse(
        status_code=422, content={'error': 'Field "text" must not be empty'}
      )

    synth_request = gateway._make_request(normalized, voice=voice)
    job_key = synth_request.content_hash
    job = store.create_or_get(job_key, synth_request.to_json())
    status_code = 200 if job.status == 'ready' else 202
    return JSONResponse(
      status_code=status_code,
      content=_job_response(job),
    )

  @app.get('/tts/{job_key}')
  async def tts_job_status(job_key: str) -> Response:
    job = store.get(job_key)
    if job is None:
      return JSONResponse(status_code=404, content={'error': 'job not found'})

    status_code = 200 if job.status == 'ready' else 202
    return JSONResponse(status_code=status_code, content=_job_response(job))

  @app.get('/tts/{job_key}/audio')
  async def tts_job_audio(job_key: str) -> Response:
    job = store.get(job_key)
    if job is None:
      return JSONResponse(status_code=404, content={'error': 'job not found'})

    if job.status != 'ready':
      return JSONResponse(
        status_code=409,
        content={'error': f'job is {job.status}, not ready', 'status': job.status},
      )

    audio_path = Path(job.artifact_path)
    try:
      return FileResponse(
        path=audio_path,
        media_type=job.content_type or 'application/octet-stream',
        filename=f'{job_key[:16]}.{_ext(job.content_type)}',
      )
    except FileNotFoundError:
      return JSONResponse(status_code=404, content={'error': 'artifact file missing'})

  def _job_response(job) -> dict[str, Any]:
    resp: dict[str, Any] = {
      'key': job.key,
      'status': job.status,
      'created_at': job.created_at,
      'started_at': job.started_at,
      'completed_at': job.completed_at,
      'chunks_total': job.chunks_total,
      'chunks_done': job.chunks_done,
      'content_type': job.content_type,
      'error': job.error,
    }
    if job.status == 'ready':
      resp['audio_url'] = f'/tts/{job.key}/audio'
    return resp

  def _ext(content_type: str | None) -> str:
    if content_type == 'audio/mpeg':
      return 'mp3'
    return 'wav'

  # ---------------------------------------------------------------------------
  # Streaming TTS (progressive MP3)
  # ---------------------------------------------------------------------------

  @app.post('/tts/stream')
  async def tts_stream(
    text: Annotated[str, Body()],
    voice: Annotated[str | None, Body()] = None,
  ) -> StreamingResponse:
    normalized = text.strip()
    if not normalized:
      return JSONResponse(
        status_code=422, content={'error': 'Field "text" must not be empty'}
      )

    try:
      audio_stream = gateway.stream(normalized, voice=voice, output_format='mp3')
      # Eagerly produce the first chunk so synthesis errors surface
      # as proper HTTP errors instead of broken/empty streams.
      first_chunk = await audio_stream.__anext__()
    except (SynthesisError, RuntimeError) as exc:
      status = (
        _status_from_synthesis_error(exc) if isinstance(exc, SynthesisError) else 502
      )
      return JSONResponse(
        status_code=status,
        content={'error': str(exc)},
      )
    except StopAsyncIteration:
      return JSONResponse(
        status_code=502,
        content={'error': 'synthesis produced no audio'},
      )

    async def _prepend_first(first: bytes, rest: AsyncIterator[bytes]):
      yield first
      async for chunk in rest:
        yield chunk

    return StreamingResponse(
      _prepend_first(first_chunk, audio_stream),
      media_type='audio/mpeg',
      headers={
        'X-TTS-Mode': 'stream',
        'X-TTS-Primary-Engine': config.primary_engine,
      },
    )

  return app
