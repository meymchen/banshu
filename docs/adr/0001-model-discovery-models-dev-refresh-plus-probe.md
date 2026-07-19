# Dynamic model discovery: models.dev refresh as backbone, vendor /models as additive probe

We want providers to discover models at runtime instead of relying only on the
bundled catalog. The obvious design — query each vendor's `/v1/models` — was
rejected as the primary mechanism because those endpoints return bare ids with
no cost, context window, or capability metadata, and vendor support is spotty
(as of 2026-07: DeepSeek and Moonshot document it; Z.AI, Kimi For Coding, and
Xiaomi MiMo don't; MiniMax confirmed it has none). Instead, `Models::refresh()`
fetches models.dev's `api.json` once (full metadata, covers all six built-in
vendors — the same source our bundled catalog is generated from) and merges it
over the catalog by id; vendor `/models` probes run second and only *append*
unknown ids as zero-metadata models — their real value is custom
OpenAI/Anthropic-compatible endpoints (llama.cpp, vLLM, gateways) that have no
catalog at all. This mirrors pi, whose dynamic discovery is a remote-catalog
overlay, not per-vendor listing.

## Consequences

- `refresh()` is infallible and best-effort: it returns a per-provider report
  (refreshed / skipped-no-key / endpoint-unsupported / failed) and never
  disturbs the existing overlay on failure; offline, the bundled catalog still
  works. MiniMax's guaranteed 404 is reported as endpoint-unsupported, not an
  error.
- No TTL, no disk persistence, no force flag in the library — refresh cadence
  and caching are application-layer policy (pi does its 4h TTL + store in the
  coding-agent layer, not the ai package).
- Probe-synthesized models follow zero-means-unknown: costs and window sizes
  are `0`, never guessed, so cost accounting shows nothing rather than lies.
- The xtask-generated bundled catalog stays as the offline baseline, and the
  models.dev provider-id mapping (moonshot→moonshotai, kimi→kimi-for-coding)
  moves into the library as a per-provider field so custom providers can opt
  out of the models.dev layer entirely.
