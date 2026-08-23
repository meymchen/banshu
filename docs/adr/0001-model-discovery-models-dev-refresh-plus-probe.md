# Dynamic model discovery: models.dev refresh as backbone, vendor /models as additive probe

Status: amended by issue #51 on 2026-08-23.

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
- Issue #51 amends the original no-persistence decision. The PRD requires
  offline startup and last-known-good recovery in the ai package, so
  applications may inject a `ModelsStore` adapter and select network,
  freshness, force, and cancellation policy through `RefreshOptions`. The
  library owns the policy vocabulary and precedence rules; the application
  still owns durable storage through its adapter. `force` bypasses freshness,
  but never overrides `allow_network=false`.
- Catalog Refresh persists and reuses ETag and Last-Modified validators. A 304
  advances `checked_at` while retaining the restored Overlay. Failed or
  cancelled Catalog Refresh and Probe work never replaces the corresponding
  last-known-good layer.
- Probe-synthesized models follow zero-means-unknown: costs and window sizes
  are `0`, never guessed, so cost accounting shows nothing rather than lies.
- The xtask-generated bundled catalog stays as the offline baseline, and the
  models.dev provider-id mapping (moonshot→moonshotai, kimi→kimi-for-coding)
  moves into the library as a per-provider field so custom providers can opt
  out of the models.dev layer entirely.
