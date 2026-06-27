# Facts

- The project includes a repeatable latency benchmark script that records JSON results for warm and cold TTS runs across short, sentence-length, medium, and long text fixtures.
- The benchmark script measures /v1/speech total time, /tts/stream time to first byte, /tts/stream total time, and any new streaming transport or CosyVoice endpoint added by this goal.
- The Raycast Option+R selected-text flow starts playback from the gateway stream without waiting for the full generated audio file when streaming is available.
- Backward compatibility is not required; breaking API or preference behavior is acceptable when it reduces complexity or improves maintainability, as long as the release impact is documented.
- The gateway has a stream-specific first-chunk strategy so long selections can produce first audio faster without relying only on the global buffered-synthesis chunk size.
- The streaming transport is validated for multi-chunk playback quality, with no obvious pauses, clicks, or decoder stalls in the target Raycast playback path.
- Automated tests cover stream route behavior, ordered streamed output, first-chunk planning, and the Raycast streaming playback path.
- The gateway engine abstraction supports engines that can stream audio incrementally while still allowing non-streaming engines to work through an adapter or simpler compatibility path.
- CosyVoice is implemented as a first-class streaming backend, preferably through a sidecar-backed engine boundary that keeps tts-gateway responsible for routing, health, metrics, and client-facing API behavior.
- CosyVoice latency and quality are benchmarked against the tuned Kokoro streaming baseline before any default-engine recommendation changes.
