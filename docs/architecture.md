# Architecture

This document describes the application's current structure and the boundaries
that future changes must preserve. If this guide conflicts with the code or
tests, the code and tests are authoritative.

## System overview

AI Vocab Video Generator is a local Streamlit app. It turns a topic or an
editable vocabulary list into narrated question-and-answer segments and
encodes them as an MP4.

```text
Streamlit WebUI
    │
    ├── validated GenerationRequest
    ├── session-only provider credentials
    │
    ▼
JobStorage ── snapshots uploads and creates a schema-v3 manifest
    │
    ▼
GenerationPipeline
    ├── vocabulary and phonetics ── OpenAI-compatible LLM provider
    ├── word materials ──────────── local / Pexels / Pixabay provider
    ├── narration ───────────────── Edge TTS provider
    ├── cards ───────────────────── Pillow renderer
    └── MP4 ─────────────────────── MoviePy / FFmpeg composer
    │
    ▼
Isolated job directory with inputs, reusable artifacts, manifest, and videos
```

Optional topic transcription uses FunASR with SenseVoiceSmall. It is loaded
only when the optional ASR dependencies are installed and the user invokes
speech input.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `domain.py` | Validated models for requests, layouts, narration, media, jobs, and progress. It does not perform network or UI work. |
| `config.py` | Non-secret preferences, provider presets, and environment-backed secrets. |
| `providers/base.py` | Protocols for vocabulary, image, speech, and transcription providers. |
| `providers/llm.py` | OpenAI-compatible vocabulary generation and automatic phonetic completion. |
| `providers/images.py` | Local media validation and size-bounded Pexels/Pixabay image retrieval. |
| `providers/tts.py` | Edge TTS synthesis with bounded transient-error retries, safe failure categories, and bounded output validation. |
| `providers/asr.py` | Lazy optional FunASR transcription. |
| `preview.py` | Typed per-word candidate, selection and upload state; non-reversible remote search fingerprints independent of layout. |
| `storage.py` | Immutable input snapshots, schema-v3 manifests, job locks, privacy checks, and atomic writes. |
| `pipeline.py` | Generation and regeneration orchestration, cache fingerprints, progress events, and failure handling. |
| `rendering/cards.py` | Deterministic Pillow question and answer layers. |
| `rendering/video.py` | Timed material overlays, narration, music ducking, progress animation, and H.264/AAC composition. |
| `webui.py` | Bilingual Streamlit presentation, input collection, provider construction, previews, generation, and regeneration controls. |

## Request and generation flow

1. The WebUI collects a topic or vocabulary entries, a background image, and
   the selected layout, material, narration, music, and output settings.
2. `GenerationRequest` validates the complete combination. A request must have
   content, a valid background at generation time, and at least one enabled
   answer narration track with a positive repeat count.
3. `JobStorage.create_job()` copies uploaded inputs into a staging directory,
   validates the manifest, and makes the job directory visible atomically.
4. `GenerationPipeline` obtains vocabulary, prepares deterministic materials,
   synthesizes narration, renders cards, and composes a temporary MP4.
5. The temporary MP4 replaces the pending output atomically, but only after
   composition succeeds. The manifest then records the active video, all
   previous completed videos, a safe status message, and cache fingerprints.
6. On failure, the pipeline removes the partial MP4, records a redacted error,
   and preserves valid inputs and completed intermediate artifacts for
   diagnosis or regeneration.

Progress stages are `preparing`, `vocabulary`, `images`, `speech`, `cards`,
`composing`, and `complete`. The WebUI localizes their messages instead of
persisting translated UI text in the pipeline.

## Job storage

Each job uses a random 32-character hexadecimal identifier and is isolated
under the configured storage root:

```text
storage/<job-id>/
├── inputs/
│   ├── background.<ext>
│   ├── materials/
│   ├── pins/
│   ├── replacements/
│   ├── music/
│   └── fonts/
├── artifacts/
│   ├── materials/
│   ├── audio/
│   ├── cards/
│   └── videos/video-0001.mp4
├── manifest.json
└── .job.lock
```

Only schema version 3 is supported. Manifest paths must be relative to the job
and point to an allowlisted directory. The app rejects unknown fields, path
traversal, unsafe symlinks, absolute private paths, credentials, invalid fonts,
and malformed artifact references before reading or writing a manifest.

Manifest writes use a same-directory temporary file, filesystem sync, and
atomic replacement. A per-job lock prevents concurrent generation or
regeneration from mutating the same job.

## Caching and regeneration

The pipeline fingerprints the inputs that affect each stage:

- vocabulary: topic, entries, phonetic mode, model endpoint, model name, and
  prompt version;
- materials: source, provider, selection mode, seed, query, aspect, and local
  file bytes where applicable;
- speech: text, track, voice, repeats, rate, and volume;
- cards: content, background, material bytes, renderer version, and styles;
- composition: ordered visual and audio artifacts, durations, music, render
  settings, and material-video offsets.

An unchanged fingerprint may reuse its completed artifact. A changed input
invalidates only the affected stage and downstream stages. Regeneration can
replace selected material indices and always creates the next numbered MP4;
it does not overwrite a previously completed video.

Replacement files are immutable snapshots in `inputs/replacements/`. Only after
composition succeeds are they committed as the request's pinned materials in the
same manifest update as the completed video. A failed replacement run restores
the previous material assignments; subsequent runs use the last successful choices.

The WebUI keeps per-word search keywords, candidate metadata, the selected full-size file, and a local
override together in each `WordMaterialState`. Hiding materials does not clear
that state. Search identity excludes layout-only settings, while card-preview
identity includes visual settings and the currently selected inputs. A changed
card-preview identity clears only the rendered preview, not the selected image.
Editing only a word's search keywords clears its candidate gallery but retains
the selected file or upload. `GenerationRequest.material_queries` stores bounded,
normalized keywords by entry index, independently of `WordEntry` and speech.
Remote acquisition fingerprints include the effective query; local assignment
and speech caches do not depend on it.

The material overview derives its status directly from each `WordMaterialState`:
an upload takes precedence over a selected file, then a saved pin, then an empty
gallery or pending automatic search. Navigation changes only the current gallery
index; it does not acquire media or change the selected source.

Recent-task discovery scans direct task directories and validates current-schema
metadata without loading fonts or media bytes. Results are capped at 20 in the UI
and cached per session, storage location, and credential fingerprint. Refreshing
or completing a generation invalidates the cache. The displayed update time is
the manifest's filesystem modification time; no schema migration is introduced.
Invalid tasks are skipped without repair or deletion. Manifest reads reject
symlinks, non-regular files, and files larger than 4 MiB.

Video history only exposes files declared in `artifacts.videos`, newest first.
The selected reference is checked again against the manifest and confined to the
task's video directory at read time. Browsing or downloading a version is read-only:
it never changes the task request, current editor, or latest generation result.

Static previews select a word and a question/answer card independently of the
generation request. Their fingerprints include the preview index, card type,
question text/style and material inputs. Local material assignment and video
frame extraction use the same word index and seed as generation. A remote word
without a pin uses a labelled placeholder rather than issuing a network request.
Question previews are available only while the question segment is enabled;
audio and animated progress bars are outside static-preview scope.

## Provider boundaries

Provider implementations are dependency-injected through protocols so the
pipeline can be tested without Streamlit or network access.

- LLM endpoints must use HTTPS, except exact loopback HTTP endpoints. Userinfo,
  fragments, credential-bearing components, and query parameters other than
  the non-secret `api-version` option are rejected.
- LLM and image metadata responses are streamed under explicit byte limits.
- Remote image downloads accept only HTTPS URLs from the selected provider's
  allowlisted CDN hosts. Redirects, content type, encoded size, decoded pixel
  count, and provider-supplied source IDs are validated before a file is saved.
- Local images and videos are decoded and checked for file size, dimensions,
  duration, and frame rate before use.
- `media_limits.py` supplies shared local byte budgets for the WebUI, media
  validation, and task snapshots: 32 MiB for images/audio and 128 MiB for videos.
  Remote image downloads retain their separate 10 MiB limit. All images retain
  the 50-million-pixel budget. Size errors carry only numeric sizes, never paths.
- Provider errors expose a safe user message and diagnostic category, never
  raw authorization headers, URLs containing keys, or response bodies.

Connection tests use the same provider adapters as generation. Every new
provider needs mock-transport tests for success, rejection, timeout, oversized
responses, and credential redaction.

## Rendering boundary

`CardRenderer` creates separate base and foreground layers. This lets video
materials animate between the background and text while static images use the
same visual ordering. Material fitting supports `contain`, `cover`, and
`stretch`, followed by circle or rectangle masking.

`VideoComposer` builds the narration timeline, animates question progress,
loops muted material videos, mixes optional background music, and ducks music
during foreground narration. Output uses H.264 video and AAC audio at the
request FPS. MoviePy resources are explicitly closed after composition.

## Security invariants

- The checked-in launchers and Streamlit configuration bind to `127.0.0.1`.
  Public or LAN exposure requires a separate authenticated deployment design.
- Provider credentials stay in environment-backed settings or current
  Streamlit session memory. They must never enter logs, manifests, filenames,
  previews, provider source IDs, or generated media metadata.
- Uploaded inputs are copied into private job storage before generation or
  regeneration can use them.
- Runtime data, `.env`, model caches, generated media, and build output are
  ignored by Git and excluded from source distributions where applicable.
- File and timeline limits are enforced before expensive decoding or encoding.
- User-facing errors are safe summaries; raw exceptions remain detached from
  persisted and displayed messages.

See [SECURITY.md](../SECURITY.md) for reporting and local-data guidance.

## Safe extension rules

When adding or changing a capability:

1. Put validation and defaults in typed domain models, not in widget-only code.
2. Keep network behavior behind a provider protocol.
3. Snapshot external files before a job depends on them.
4. Include every output-affecting value in the closest cache fingerprint.
5. Add focused failure and secrecy tests before exposing a new UI control.
6. Keep every visible control functional; do not ship placeholder settings.
7. Run the full checks documented in [CONTRIBUTING.md](../CONTRIBUTING.md).
