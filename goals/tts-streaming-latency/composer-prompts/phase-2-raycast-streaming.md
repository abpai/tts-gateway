# Composer Task: Phase 2 Raycast Streaming Playback

You are implementing the second slice of the approved goal in
`/Users/andypai/Projects/tts-gateway/goals/tts-streaming-latency/goal.md`.

Work in this repo:

`/Users/andypai/Projects/raycast-tts-reader`

## Why

The current Option+R path waits for the gateway to finish `/v1/speech`, buffers
the whole response with `arrayBuffer()`, writes an audio file, then starts
playback. The measured Kokoro/MPS baseline in
`/Users/andypai/Projects/tts-gateway/goals/tts-streaming-latency/benchmarks/kokoro-baseline.json`
shows that this costs real startup latency. For detected `tts-gateway`
servers, Raycast should post to `/tts/stream` and pipe the response body into
`ffplay -nodisp -autoexit -loglevel error -i -` so playback can begin before the
full response exists.

## Current files to inspect

- `src/tts-utils.ts`
- `src/play.ts`
- `src/playback-controller.ts`
- `src/read-selected-text.tsx`
- `src/read-text-editor.tsx`
- `src/useTTS.ts`
- `src/stop-audio.ts`
- `src/types.ts`
- `package.json`

## Build

Implement streaming-first playback for detected gateway servers:

1. Keep `getConfigError` and server URL parsing behavior intact.
2. For base URLs that pass the gateway `/health` probe and do not include a
   custom path, use `POST {baseUrl}/tts/stream` with JSON `{ text, voice? }`.
3. Pipe `response.body` to an `ffplay` child process on stdin.
4. Persist playback state before or immediately after spawning so the existing
   Stop Audio command can terminate the streaming player.
5. When Stop Audio kills the player, ensure the command resolves as
   `"stopped"` and the fetch/body pump is aborted or allowed to terminate
   promptly.
6. Preserve the existing buffered `createSpeech` + `play` path for custom URLs,
   non-gateway servers, and any case where streaming is not viable.
7. Keep the command entrypoints user-facing behavior simple:
   - `read-selected-text` should stream when gateway streaming is available.
   - `read-text-editor` should use the same streaming-capable path.
   - Success toasts should still include the gateway engine when available.
8. Add a small test harness if needed. Prefer Vitest around pure decision logic
   and mocked `fetch`/`spawn` seams rather than brittle Raycast UI tests.

## Design hints

- A useful shape is a higher-level `speakText(text)` function that returns
  `{ warnings, completion, engine? }`.
- The existing file-buffer helpers can remain for fallback.
- Avoid reading or printing `.env` files.
- Avoid broad rewrites of onboarding, screenshots, metadata, or unrelated
  preferences.
- If speed support is straightforward for `ffplay`, preserve it. If it adds too
  much risk, fall back to buffered playback when speed is not `1.0` and include a
  clear warning in code/tests.
- If save-audio is enabled, it is acceptable to fall back to buffered playback
  instead of trying to tee the stream.

## Validation

Run these in `/Users/andypai/Projects/raycast-tts-reader`:

```bash
pnpm lint
pnpm build
pnpm test
```

If you add a new test script or dependency, update `package.json` and
`pnpm-lock.yaml` consistently.

Do not commit, push, or open a PR. Leave the diff in the workspace for Codex to
review.
