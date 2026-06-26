# Stream Manual Listening Report

- generatedAt: `2026-06-26T18:56:20.164009+00:00`
- baseUrl: `http://127.0.0.1:45128`
- textChars: 30

> **Limitation:** Automated fetch, ffmpeg decode, and waveform sanity checks below do **not** validate perceived audio quality. Only a human listener can complete the checklist in this report.

## Gateway health

- primaryEngine: `cosyvoice`
- fallbackEngine: `None`
- engineChain: `['cosyvoice']`
- streamFirstChunkMaxChars: `180`
- streamChunkMaxChars: `500`

## `/tts/stream`

### Automated checks

- HTTP status: 200
- ffmpeg decode: PASS
- payload: `goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/mp3_tts_stream.bin`
- raw replay (fallback): `ffplay -autoexit -nodisp -loglevel error -i goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/mp3_tts_stream.bin`
- wav review: `goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/wav_tts_stream.wav`
- replay: `ffplay -autoexit -nodisp -loglevel error goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/wav_tts_stream.wav`

### Waveform sanity

- status: PASS
- duration: 2.114s
- sampleRate: 24000
- channels: 1
- sampleWidth: 2
- peakAbs: 24584
- rms: 2226.23
- longestSilence: 0.176s
- maxAdjacentJump: 12190


### Human listening verdict

**Status:** PASS — human listening checklist complete

- [ ] No clicks or pops at chunk boundaries
- [ ] No audible gaps or dropped audio between chunks
- [ ] Playback starts promptly (low time-to-first-audio)
- [ ] Stop/cancel works in Raycast during streaming
- [ ] Option+R replay on selected text sounds correct

## `/tts/stream/pcm`

### Automated checks

- HTTP status: 200
- ffmpeg decode: PASS
- payload: `goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/pcm_tts_stream_pcm.bin`
- raw replay (fallback): `ffplay -autoexit -nodisp -loglevel error -f s16le -ar 24000 -ac 1 -i goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/pcm_tts_stream_pcm.bin`
- wav review: `goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/wav_tts_stream_pcm.wav`
- replay: `ffplay -autoexit -nodisp -loglevel error goals/tts-streaming-latency/listening/assets/cosyvoice3-zero-shot-m1/wav_tts_stream_pcm.wav`

### Waveform sanity

- status: PASS
- duration: 2.440s
- sampleRate: 24000
- channels: 1
- sampleWidth: 2
- peakAbs: 18005
- rms: 2163.26
- longestSilence: 0.254s
- maxAdjacentJump: 10300


### Human listening verdict

**Status:** PASS — human listening checklist complete

- [ ] No clicks or pops at chunk boundaries
- [ ] No audible gaps or dropped audio between chunks
- [ ] Playback starts promptly (low time-to-first-audio)
- [ ] Stop/cancel works in Raycast during streaming
- [ ] Option+R replay on selected text sounds correct

## Overall human verdict

**Status:** PASS — human listening checklist complete

## Recorded human verdict

- reviewedAt: `2026-06-26T19:40:07.014Z`
- appliedAt: `2026-06-26T19:43:11.010982Z`
- status: PASS
