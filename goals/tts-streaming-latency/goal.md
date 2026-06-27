# TTS Streaming Latency Goal

Improve the `tts-gateway` and `raycast-tts-reader` speech path so the
Option+R selected-text workflow starts audible playback much faster and can be
optimized against repeatable benchmark data. The work includes benchmark
scripts, Raycast streaming playback, gateway first-chunk tuning, automated
streaming tests, a streaming-capable engine abstraction, and a CosyVoice
sidecar-backed streaming backend.

Use `facts.md` as the shared understanding of required outcomes.

Use `plan.md` as the execution plan.

Done means the accepted facts are implemented or explicitly resolved, the
planned automated checks pass, manual streaming audio quality is validated, and
CosyVoice is benchmarked against the tuned Kokoro streaming baseline before any
default-engine recommendation changes.
