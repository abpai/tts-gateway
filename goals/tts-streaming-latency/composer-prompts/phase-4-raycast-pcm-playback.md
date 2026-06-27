# Composer Task: Phase 4 Raycast PCM Playback

You are implementing the Raycast half of Step 4 in
`/Users/andypai/Projects/tts-gateway/goals/tts-streaming-latency/goal.md`.

Work in this repo:

`/Users/andypai/Projects/raycast-tts-reader`

## Why

`tts-gateway` now exposes `/tts/stream/pcm`, which streams raw PCM with metadata
headers. Raycast should prefer that endpoint for detected gateway servers so it
can avoid concatenated per-chunk MP3 decoding artifacts and start playback with
simple raw-audio `ffplay` flags.

## Current context

The worktree already contains uncommitted Phase 2 changes:

- `src/speak.ts`
- `src/playback-controller.ts`
- `src/playback-mode.ts`
- `src/server-url.ts`
- Vitest setup and tests

Preserve that work. Do not revert unrelated changes.

## Gateway PCM contract

`POST {baseUrl}/tts/stream/pcm` accepts JSON `{ text, voice? }`.

Successful responses have:

- `Content-Type: audio/raw`
- `X-TTS-Mode: stream-pcm`
- `X-TTS-Primary-Engine`
- `X-TTS-Sample-Rate`
- `X-TTS-Channels`
- `X-TTS-Sample-Width`
- `X-TTS-Pcm-Format`, for example `s16le`

Raycast should play PCM with:

```bash
ffplay -nodisp -autoexit -loglevel error -f <pcm-format> -ar <sample-rate> -ac <channels> -i -
```

## Build

Implement PCM-preferred streaming playback:

1. For detected gateway servers, try `/tts/stream/pcm` first.
2. Parse and validate the PCM headers needed for `ffplay`.
3. Pass endpoint-specific stdin playback args into `startStdinPlayback`.
4. Preserve Stop Audio behavior and request abort behavior.
5. If PCM is unavailable or returns an unsupported header/status, fall back to
   existing `/tts/stream` MP3 stdin playback before falling back to buffered
   synthesis.
6. Keep custom URL and non-gateway behavior buffered.
7. Keep speed/save-audio fallback behavior unchanged unless a small pure helper
   needs adjustment.
8. Add or update Vitest tests for:
   - PCM header parsing.
   - `ffplay` args for PCM stdin playback.
   - gateway streaming chooses PCM first.
   - PCM 404/unsupported metadata falls back to MP3 stream.
   - stream-pump failure still kills `ffplay`.

## Non-goals

- Do not update screenshots, onboarding, or metadata.
- Do not remove MP3 fallback.
- Do not implement CosyVoice.
- Do not commit, push, or open a PR.

## Validation

Run these in `/Users/andypai/Projects/raycast-tts-reader`:

```bash
pnpm lint
pnpm build
pnpm test
```

Leave the diff uncommitted for Codex review.
