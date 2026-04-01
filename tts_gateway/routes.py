"""FastAPI routes for the TTS gateway.

Canonical API:       /v1/jobs, /v1/jobs/{key}, /v1/jobs/{key}/audio, /v1/speech
Streaming:           /tts/stream
Legacy shims:        /tts, /tts/sync, /tts/{key}, /tts/{key}/audio
Infrastructure:      /health, /warmup
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from tts_gateway.config import GatewayConfig, load_config
from tts_gateway.render import stream_audio
from tts_gateway.runtime import JobRuntime, NoEnginesError, run_worker_loop
from tts_gateway.types import JobView

logger = logging.getLogger(__name__)


def _job_response(view: JobView) -> dict[str, Any]:
  resp = asdict(view)
  if view.status == 'ready':
    resp['audio_url'] = f'/v1/jobs/{view.key}/audio'
  return resp


def _ext(content_type: str | None) -> str:
  if content_type == 'audio/mpeg':
    return 'mp3'
  return 'wav'


def create_app(config: GatewayConfig | None = None) -> FastAPI:
  if config is None:
    config = load_config()
  runtime = JobRuntime(config)

  worker_task: list = []

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    import asyncio

    task = asyncio.create_task(
      run_worker_loop(runtime, poll_seconds=config.worker_poll_seconds)
    )
    worker_task.append(task)
    yield
    task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      pass
    runtime.close()

  app = FastAPI(title='TTS Gateway', version='0.3.0', lifespan=lifespan)

  import warnings

  warnings.filterwarnings('ignore', category=UserWarning, module=r'torch\.')
  warnings.filterwarnings('ignore', category=UserWarning, module=r'kokoro\.')
  warnings.filterwarnings('ignore', category=FutureWarning, module=r'torch\.')
  warnings.filterwarnings('ignore', message='.*unauthenticated requests.*HF.*')

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

  # -----------------------------------------------------------------------
  # Infrastructure
  # -----------------------------------------------------------------------

  @app.get('/health')
  async def health() -> dict[str, Any]:
    return {
      'ok': True,
      'primaryEngine': config.primary_engine,
      'fallbackEngine': config.fallback_engine,
      'outputFormat': config.output_format,
      'chunkConcurrency': runtime.concurrency,
      'chunkMaxChars': config.chunk_max_chars,
      'requestTimeoutSeconds': config.request_timeout_seconds,
      'engineTimeoutSeconds': config.engine_timeout_seconds,
      'defaultVoice': config.default_voice,
      'engineChain': runtime.engine_chain(),
      'engines': runtime.engine_info(),
    }

  @app.post('/warmup')
  async def warmup() -> dict[str, Any]:
    results = await runtime.warmup()
    return {'ok': True, 'engines': results}

  # -----------------------------------------------------------------------
  # Canonical API: /v1/speech (sync)
  # -----------------------------------------------------------------------

  @app.post('/v1/speech')
  async def v1_speech(
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    normalized = text.strip()
    if not normalized:
      return JSONResponse(
        status_code=422, content={'error': 'Field "text" must not be empty'}
      )

    try:
      spec = runtime.make_spec(normalized, voice=voice)
      artifact = await runtime.run_until_complete(
        spec, timeout=config.request_timeout_seconds
      )
    except NoEnginesError:
      return JSONResponse(
        status_code=503,
        content={'error': 'all engines in chain are unavailable'},
      )
    except TimeoutError:
      return JSONResponse(
        status_code=504,
        content={'error': 'gateway request timed out'},
      )
    except RuntimeError as exc:
      return JSONResponse(status_code=502, content={'error': str(exc)})

    try:
      return FileResponse(
        path=artifact.output_path,
        media_type=artifact.content_type,
        filename=f'{spec.content_hash[:16]}.{_ext(artifact.content_type)}',
        headers={
          'X-TTS-Chunk-Count': str(artifact.chunks_total),
          'X-TTS-Primary-Engine': config.primary_engine,
        },
      )
    except FileNotFoundError:
      return JSONResponse(status_code=404, content={'error': 'artifact file missing'})

  # -----------------------------------------------------------------------
  # Canonical API: /v1/jobs (async)
  # -----------------------------------------------------------------------

  @app.post('/v1/jobs')
  async def v1_jobs_submit(
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    normalized = text.strip()
    if not normalized:
      return JSONResponse(
        status_code=422, content={'error': 'Field "text" must not be empty'}
      )

    spec = runtime.make_spec(normalized, voice=voice)
    view = runtime.submit(spec)
    status_code = 200 if view.status == 'ready' else 202
    return JSONResponse(status_code=status_code, content=_job_response(view))

  @app.get('/v1/jobs/{job_key}')
  async def v1_jobs_status(job_key: str) -> Response:
    view = runtime.get(job_key)
    if view is None:
      return JSONResponse(status_code=404, content={'error': 'job not found'})
    status_code = 200 if view.status == 'ready' else 202
    return JSONResponse(status_code=status_code, content=_job_response(view))

  @app.get('/v1/jobs/{job_key}/audio')
  async def v1_jobs_audio(job_key: str) -> Response:
    view = runtime.get(job_key)
    if view is None:
      return JSONResponse(status_code=404, content={'error': 'job not found'})

    if view.status != 'ready':
      return JSONResponse(
        status_code=409,
        content={'error': f'job is {view.status}, not ready', 'status': view.status},
      )

    result = runtime.get_artifact_path(job_key)
    if result is None:
      return JSONResponse(status_code=404, content={'error': 'artifact file missing'})

    audio_path, content_type = result
    try:
      return FileResponse(
        path=audio_path,
        media_type=content_type,
        filename=f'{job_key[:16]}.{_ext(content_type)}',
      )
    except FileNotFoundError:
      return JSONResponse(status_code=404, content={'error': 'artifact file missing'})

  # -----------------------------------------------------------------------
  # Streaming TTS
  # -----------------------------------------------------------------------

  @app.post('/tts/stream')
  async def tts_stream(
    text: Annotated[str, Body()],
    voice: Annotated[str | None, Body()] = None,
  ) -> Response:
    normalized = text.strip()
    if not normalized:
      return JSONResponse(
        status_code=422, content={'error': 'Field "text" must not be empty'}
      )

    engines = runtime.engines
    if not engines:
      return JSONResponse(
        status_code=503,
        content={'error': 'all engines in chain are unavailable'},
      )

    spec = runtime.make_spec(normalized, voice=voice)
    from dataclasses import replace

    spec = replace(spec, output_format='mp3')

    try:
      audio_stream = stream_audio(
        spec,
        engines,
        concurrency=runtime.concurrency,
        engine_timeout=config.engine_timeout_seconds,
        ffmpeg_path=config.ffmpeg_path,
      )
      first_chunk = await audio_stream.__anext__()
    except RuntimeError as exc:
      return JSONResponse(status_code=502, content={'error': str(exc)})
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

  # -----------------------------------------------------------------------
  # Legacy shims (forward to canonical routes)
  # -----------------------------------------------------------------------

  @app.post('/tts/sync')
  async def legacy_tts_sync(
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    logger.info('legacy-route: POST /tts/sync → /v1/speech')
    return await v1_speech(text=text, voice=voice)

  @app.post('/tts')
  async def legacy_tts(
    request: Request,
    text: Annotated[str, Form(...)],
    voice: Annotated[str | None, Form()] = None,
  ) -> Response:
    accept = request.headers.get('accept', '')
    if accept == 'application/json':
      logger.info('legacy-route: POST /tts (json) → /v1/jobs')
      return await v1_jobs_submit(text=text, voice=voice)
    logger.info('legacy-route: POST /tts → /v1/speech')
    return await v1_speech(text=text, voice=voice)

  @app.get('/tts/{job_key}')
  async def legacy_tts_status(job_key: str) -> Response:
    logger.info('legacy-route: GET /tts/%s → /v1/jobs/%s', job_key[:16], job_key[:16])
    return await v1_jobs_status(job_key=job_key)

  @app.get('/tts/{job_key}/audio')
  async def legacy_tts_audio(job_key: str) -> Response:
    logger.info(
      'legacy-route: GET /tts/%s/audio → /v1/jobs/%s/audio',
      job_key[:16],
      job_key[:16],
    )
    return await v1_jobs_audio(job_key=job_key)

  return app
