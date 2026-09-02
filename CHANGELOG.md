# Changelog

All notable changes to AI Vocab Video Generator are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bilingual Simplified Chinese and English Streamlit interface.
- Editable AI-generated or manual vocabulary lessons with optional phonetics.
- OpenAI, DeepSeek, Moonshot, Qwen, Ollama, and custom compatible LLM providers.
- Word media from local files, Pexels, or Pixabay, with deterministic previews.
- Portrait and landscape card rendering with configurable narration, question
  segments, progress bars, music, and task regeneration.
- Optional FunASR topic transcription on Python 3.11 and 3.12.
- Loopback-only launchers, isolated task storage, secret scanning, dependency
  auditing, and release archive checks.
- A manually triggered GitHub Actions workflow that reruns the complete CI
  suite before creating a version-checked draft source release.
- A compact GitHub repository link in the application header with bilingual
  accessible labels and safe new-tab navigation.

### Changed

- Changed default fast English narration to one playback per word, preserving
  explicitly configured repeat counts in existing tasks.
- Raised local background and word-image limits to 32 MiB across uploads,
  previews, and task snapshots. Upload controls state the image/video limits,
  and size errors include the actual size and allowed maximum. Remote image
  downloads remain limited to 10 MiB.
- Replaced internal implementation plans with current architecture and roadmap
  documentation for maintainers.
- Bundled a project-owned countdown cue.
- Bounded image and video previews according to output aspect ratio.
- Rejected LLM endpoint fragments and query parameters other than the non-secret
  `api-version` option before task manifests are written.
- Enforced the decoded-pixel safety budget for newly uploaded preview
  backgrounds as well as final card rendering.
- Limited each Streamlit upload to 128 MiB before the app applies stricter
  media-specific limits.
- Created the optional ASR model cache with owner-only permissions on POSIX.
- Replaced internal historical compatibility names with current domain names.
- Added per-word remote material previews with optional local overrides and
  explicit empty-result guidance.
- Removed the MoviePy/ImageIO static-image deprecation path while retaining
  the security-patched Pillow 12 dependency.

### Fixed

- Retry temporary Edge TTS failures up to twice and report the affected word,
  narration track, failure reason, and attempt count in the selected language.
  Retry attempts use fresh connections and discard partial audio.
- Kept preview and material word selectors in sync after vocabulary edits,
  safely handling stale browser labels instead of comparing strings with indices.
- Prevented remote image provider response identifiers from reflecting an
  active API key into persisted job manifests.

[Unreleased]: https://github.com/soloshow-labs/ai-vocab-video-generator/commits/main
