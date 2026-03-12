# LEARNINGS

## Corrections

| Date | Source | What Went Wrong | What To Do Instead |
| ---- | ------ | --------------- | ------------------ |
| 2026-03-11 | self | Started review in a repo without `.agents/LEARNINGS.md` | Create the project memory file first when it is missing |

## User Preferences

- Return code review findings as prioritized, high-signal issues.

## Patterns That Work

- Run `uv run pytest`, `uv run ruff check .`, and `uv run ty check` during reviews; this repo can have passing tests while the required typecheck still fails.
- Keep optional backend imports (`pocket_tts`, similar) behind runtime `import_module()` calls plus local `Protocol` types so `ty check` stays green without installing extras.
- Apply process-wide default voice fallback inside `TtsGateway`, not just the CLI/env layer, so API requests that omit `voice` still honor `TTS_DEFAULT_VOICE`.
- For readability-only refactors, small helper extractions in `gateway.py` and `audio.py` are low-risk and well-covered by the current test suite plus `ty check`.

## Patterns That Don't Work

- Assuming this repo's CI status from pytest alone.
- Treating `git status --porcelain` snapshots as a reliable change detector while the whole repo is still untracked; content edits inside untracked files will not show up there.

## Domain Notes

- This project is a FastAPI-based TTS gateway with optional native Kokoro and Pocket backends.
