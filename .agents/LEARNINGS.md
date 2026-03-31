# LEARNINGS

## Corrections

| Date | Source | What Went Wrong | What To Do Instead |
| ---- | ------ | --------------- | ------------------ |
| 2026-03-11 | self | Started review in a repo without `.agents/LEARNINGS.md` | Create the project memory file first when it is missing |
| 2026-03-30 | self | Updated runtime defaults without updating shared test config fixtures | Keep `tests/conftest.py` defaults aligned with `load_config()` so helper-based tests reflect real gateway defaults |
| 2026-03-30 | self | Planned Bunny final-state deployment without first adding a publishable artifact here | Add Dockerfile, `.dockerignore`, and GHCR publishing in this repo before switching Bunny away from its vendored copy |
| 2026-03-30 | self | Tried to verify `/tts` from a repo venv that only had dev deps installed | For local runtime smoke tests, install the engine extra first, e.g. `uv sync --group dev --extra kokoro`, or use the documented Docker path |
| 2026-03-30 | self | Ran packaging checks against stale files left in `dist/` | In CI and release workflows, `rm -rf dist` before `uv build` so `twine check` only sees the current release artifacts |

## User Preferences

- Return code review findings as prioritized, high-signal issues.

## Patterns That Work

- Run `uv run pytest`, `uv run ruff check .`, and `uv run ty check` during reviews; this repo can have passing tests while the required typecheck still fails.
- In this sandbox, set `UV_CACHE_DIR=/tmp/uv-cache` before `uv run ...` commands if the default `~/.cache/uv` path triggers permission errors.
- Keep optional backend imports (`pocket_tts`, similar) behind runtime `import_module()` calls plus local `Protocol` types so `ty check` stays green without installing extras.
- Clear `LazyNativeEngine._load_error` on successful retries so `/health` and `/warmup` do not report stale failures after recovery.
- Apply process-wide default voice fallback inside `TtsGateway`, not just the CLI/env layer, so API requests that omit `voice` still honor `TTS_DEFAULT_VOICE`.
- For readability-only refactors, small helper extractions in `gateway.py` and `audio.py` are low-risk and well-covered by the current test suite plus `ty check`.
- For concurrent chunk synthesis, keep per-chunk attempt logs local and publish them through a shared ordered sink in a shielded `finally` block so request-level timeouts still preserve completed attempt history.
- Container verification is smoother when the image exposes a first-class Docker `HEALTHCHECK` against `/health`, with the port sourced from `TTS_GATEWAY_PORT`.

## Patterns That Don't Work

- Assuming this repo's CI status from pytest alone.
- Treating `git status --porcelain` snapshots as a reliable change detector while the whole repo is still untracked; content edits inside untracked files will not show up there.

## Domain Notes

- This project is a FastAPI-based TTS gateway with optional native Kokoro and Pocket backends.
