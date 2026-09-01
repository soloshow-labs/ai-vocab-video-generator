# Roadmap

This roadmap lists possible future work, not promised features or release
dates. Work should begin only after the expected behavior, security boundary,
licensing, and test plan are clear.

## Near-term candidates

### Dictionary-backed phonetics

Automatic phonetics currently use the configured OpenAI-compatible LLM. Manual
phonetics remain the predictable offline option.

Before adding a dictionary mode:

- select a redistributable dictionary source and document its license;
- define British and American pronunciation selection;
- define precedence between manual input, dictionary results, and LLM fallback;
- normalize IPA display without changing user-entered manual phonetics;
- include dictionary identity, pronunciation variant, and fallback behavior in
  cache fingerprints;
- add bilingual controls only after all choices affect saved entries and video
  output.

The feature would need tests for known and missing words, multiple
pronunciations, punctuation, provider failures, cache invalidation, and
preservation of manual input.

### Dependency maintenance

Keep routine patch upgrades separate from source migrations. Moving to
MoviePy 2 is an API migration, not a lockfile refresh, because the project uses
MoviePy 1 editor imports, clip mutation methods, effects, and frame callbacks.

Revisit MoviePy 2 only when a stable release is compatible with the project's
security-patched Pillow version. Perform the migration on a dedicated branch
with portrait and landscape frame comparisons, audio-order tests, material
video tests, music ducking tests, and short MP4 integration tests. Never
downgrade Pillow merely to satisfy the video dependency.

Keep PyTorch and TorchAudio as a matched pair when updating optional ASR
dependencies. Upgrade static-analysis tools independently so new diagnostics
are reviewed rather than globally suppressed.

## Medium-term candidates

### Long-task control and resource use

Generation currently runs synchronously. Possible improvements include:

- a cancellation control that marks the job failed safely and removes partial
  output;
- progress within image retrieval, speech synthesis, card rendering, and
  composition rather than only between stages;
- bounded parallel image and speech work with deterministic output ordering;
- a maximum final artifact size and file-backed or streaming result delivery
  instead of reading a large MP4 into one in-memory byte string.

Concurrency must remain bounded per provider and must not expose credentials in
worker errors. Cancellation must preserve already completed reusable artifacts.

### Runtime storage lifecycle

Jobs and session uploads are intentionally local and persistent. If real usage
shows storage growth is confusing, add:

- per-session cleanup of upload copies after an immutable job snapshot;
- cumulative storage quotas and a free-space safety threshold;
- task storage usage display;
- confirmation before deleting completed jobs;
- an age-based cleanup policy that never removes an active or locked job.

These changes should be designed together so the storage totals shown in the
UI always match the filesystem and its deletion rules.

### Windows privacy enforcement

POSIX platforms apply owner-only modes to runtime directories and files.
Before claiming equivalent protection on shared Windows computers, add Windows
ACL enforcement and tests for `.env`, storage roots, manifests, uploads,
generated artifacts, transcripts, and model caches. If the expected ACL cannot
be applied, the app should show a clear warning or refuse to persist sensitive
data.

## Later candidates

### Offline TTS

Edge TTS is the only current speech backend. An offline provider should be
considered only after choosing a model with acceptable voice quality,
redistribution terms, download size, supported platforms, and Chinese/English
coverage.

The implementation should use the existing speech provider protocol and keep
the same voice, rate, volume, repeat, caching, and error contracts. Model
weights belong in an ignored runtime cache and must not be committed to the
repository or bundled into release archives.

### Module splitting

`webui.py`, `pipeline.py`, and `storage.py` are large, but their responsibilities
and tests are currently clear. Split them only when a real feature would
otherwise mix unrelated responsibilities or make focused testing harder. Do
not reorganize them solely to reduce their line counts.

Likely future seams are WebUI section renderers, pipeline stage services, and
storage manifest/media validators. Preserve public behavior and cache identity
during any split.

## Not planned

- importing unsupported early job manifests or private configuration;
- a public unauthenticated network mode;
- bundled provider credentials, proprietary fonts, music, model weights, or
  downloaded stock media;
- settings that appear in the UI without affecting generated output;
- parameter preset import/export without a demonstrated workflow that is not
  already served by defaults and `config.toml`.

## Promoting an item into development

Before implementation, create a focused design that states:

1. the user problem and visible behavior;
2. supported and unsupported cases;
3. data, credential, filesystem, network, and licensing boundaries;
4. cache and regeneration effects;
5. failure and cancellation behavior;
6. focused tests plus the full release checks.

Update this roadmap when a feature is released or intentionally rejected. Do
not leave completed implementation checklists here.
