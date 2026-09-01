# AI Vocab Video Generator

English | [中文](README.zh.md)

Turn a topic or an editable word list into a narrated English vocabulary video.
This local Streamlit app supports Simplified Chinese and English, opens in
Simplified Chinese by default, and lets you customize the layout, media,
typography, progress indicators, and narration.

Maintained by [SoloShow Labs](https://github.com/soloshow-labs).

WeChat Official Account: 一人独角show

## What it does

- Builds a vocabulary list from a topic, or accepts an editable list written as
  paired Chinese and English lines.
- Supports OpenAI, DeepSeek, Moonshot, Qwen, Ollama, and custom
  OpenAI-compatible chat endpoints.
- Uses a required local background image, with optional word media from local
  files, Pexels, or Pixabay.
- Uses Edge TTS for configurable Chinese, fast English, and slow English
  narration.
- Builds question and answer cards in 9:16 or 16:9 format.
- Stores each run in its own job directory, together with the inputs and
  settings needed for regeneration.
- Reuses valid media, audio, and cards during regeneration while keeping every
  previously completed MP4.
- Optionally transcribes a recorded topic with FunASR/SenseVoiceSmall.

No API keys, fonts, music, model weights, demo media, or generated videos are
included in the repository.

## Requirements

- Python 3.11 or 3.12 (uv can install a compatible Python when needed)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [FFmpeg](https://ffmpeg.org/download.html), with both `ffmpeg` and `ffprobe`
  available on `PATH`

macOS with Homebrew:

```bash
brew install uv ffmpeg
```

Debian or Ubuntu (install uv with the
[official installer](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
sudo apt-get update
sudo apt-get install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows users can install uv with `winget install --id=astral-sh.uv -e` and
download an FFmpeg build from the
[official FFmpeg download page](https://ffmpeg.org/download.html). Extract the
archive, add its `bin` directory to the user or system `PATH`, and open a new
terminal before running the verification commands below.

Verify the tools before continuing:

```bash
uv --version
ffmpeg -version
ffprobe -version
```

## Quick start

Clone the repository, then run all commands from the project directory:

```bash
git clone https://github.com/soloshow-labs/ai-vocab-video-generator.git
cd ai-vocab-video-generator
```

### 1. Install the locked dependencies

Choose one command according to whether you need the **Voice Input Topic**
feature. You do not need to run both commands:

```bash
# Standard installation without voice input
uv sync --frozen

# Install the optional local ASR transcription dependencies
uv sync --frozen --extra asr
```

Both commands are suitable for a first installation. They create `.venv` when
it does not exist and install the exact versions recorded in `uv.lock`.
`--extra asr` adds the larger local speech-transcription dependencies, while
`--frozen` prevents uv from changing the lockfile. Use plain `uv sync` only when
you intentionally changed project dependencies and want uv to check or update
the lockfile.

### 2. Optional: create a local `.env` file

Copying `.env.example` is optional. On macOS or Linux:

```bash
cp .env.example .env
chmod 600 .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

- Skip this step if you will enter a key in the WebUI for the current session,
  use a manual word list, or run a local Ollama model.
- Create `.env` if you want the app to reload your provider keys after a
  restart. Fill in only the providers you use and leave the rest blank.
- `.env` is ignored by Git. Never commit or paste a real key into a README,
  source file, task script, or `config.toml`.

### 3. Start the WebUI

macOS or Linux:

```bash
./webui.sh
```

Windows:

```bat
webui.bat
```

The launchers find the project directory automatically and use the locked
environment. To run on a different port:

```bash
./webui.sh --server.port 8503
```

The equivalent direct command is:

```bash
uv run --frozen streamlit run --server.address 127.0.0.1 src/ai_vocab_video_generator/webui.py
```

Open <http://localhost:8501> if the browser does not open automatically. Stop
the server with `Ctrl+C` in the terminal.

The bundled launchers and Streamlit configuration bind to `127.0.0.1`. This is
a trusted single-user local application and has no built-in authentication.
Do not expose it to a LAN or the public internet without an authenticated
reverse proxy and separate per-user storage and credentials.

The interface supports `zh-CN` and `en-US` and opens in Simplified Chinese. Use
the language selector at the top of the page to switch languages without
changing saved media or rendering settings.

### 4. Make a first video

You can create a video without an LLM or image-search API key:

1. Enter the vocabulary manually. Enable **Manual Phonetics** and paste
   repeating three-line groups:

   ```text
   苹果
   apple
   /ˈæpəl/
   香蕉
   banana
   /bəˈnɑːnə/
   ```

   Alternatively, turn both phonetic modes off and use only Chinese/English
   two-line groups.
2. Upload a background image. A background is required for every video.
3. Turn **Image Materials** off, or select local materials and upload images or
   videos that you may use. This avoids needing Pexels or Pixabay.
4. Select **Generate Video**, then preview or download the MP4.

Edge TTS still needs network access. Output and uploaded files are stored under
the ignored `storage/` directory.

To generate the word list with AI, select one LLM provider, enter that
provider's key, choose a topic and word count, and then select **Use AI to
Generate Vocabulary**. You still need to upload a background image.

## Choose only the configuration you need

| Goal | Required configuration | Not required |
| --- | --- | --- |
| Paste a complete manual list | Background image; manual phonetics with three-line groups, or both phonetic modes off with two-line groups | LLM key |
| Generate vocabulary from a topic | One LLM provider and its key, or a running local Ollama model | Keys for every other LLM provider |
| Complete missing phonetics automatically | Automatic phonetics plus the selected LLM | Manual phonetics |
| Use local word images/videos | Enable image materials, choose local source, upload files | Pexels/Pixabay key |
| Search remote still images | Enable image materials, choose remote source, configure the selected Pexels or Pixabay key | Local material upload |
| Run the LLM locally | Install Ollama, run it, and pull `qwen3.5:9b` | Cloud LLM key |
| Speak the topic instead of typing | `uv sync --frozen --extra asr`; allow the first SenseVoiceSmall model download | ASR extra for typed topics |
| Add music or a custom font | Upload a licensed music file or enter a local font path | Either item for normal generation |
| Regenerate an earlier result | Its 32-character task ID; replacement files only for entries you want to change | Original browser upload paths |

## Provider and secret configuration

Keys entered in password fields remain available only for the current
Streamlit session. To load keys automatically after a restart, copy
`.env.example` to `.env` and fill in only the providers you use:

```dotenv
AIVVG_OPENAI_API_KEY=
AIVVG_DEEPSEEK_API_KEY=
AIVVG_MOONSHOT_API_KEY=
AIVVG_QWEN_API_KEY=
AIVVG_CUSTOM_API_KEY=
AIVVG_PEXELS_API_KEY=
AIVVG_PIXABAY_API_KEY=
```

| Variable | Used when | Where to obtain it |
| --- | --- | --- |
| `AIVVG_OPENAI_API_KEY` | OpenAI is selected for AI generation or automatic phonetics | [OpenAI API keys](https://platform.openai.com/api-keys) |
| `AIVVG_DEEPSEEK_API_KEY` | DeepSeek is selected | [DeepSeek API keys](https://platform.deepseek.com/api_keys) |
| `AIVVG_MOONSHOT_API_KEY` | Moonshot is selected | [Kimi API Platform](https://platform.kimi.com/console/api-keys) |
| `AIVVG_QWEN_API_KEY` | Qwen is selected | [Alibaba Cloud Model Studio](https://help.aliyun.com/en/model-studio/get-api-key) |
| `AIVVG_CUSTOM_API_KEY` | A custom OpenAI-compatible endpoint is selected | The selected provider's console |
| `AIVVG_PEXELS_API_KEY` | Remote image materials use Pexels | [Pexels API](https://www.pexels.com/api/) |
| `AIVVG_PIXABAY_API_KEY` | Remote image materials use Pixabay | [Pixabay API](https://pixabay.com/api/docs/) |

For the current session, a key entered in the WebUI overrides the matching
`.env` value. WebUI keys are never written to `.env`, `config.toml`, job
manifests, or generated files, and they disappear when the session ends.
Environment variables set by the operating system take precedence over values
loaded from `.env`.

The LLM presets and their startup defaults are:

| Provider | Base URL | Default model | Notes |
| --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-5.6-terra` | Cloud key required |
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` | Cloud key required |
| Moonshot | `https://api.moonshot.cn/v1` | `kimi-k2.6` | Cloud key required; thinking is disabled for the structured task |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.7-flash` | Cloud key required |
| Ollama | `http://localhost:11434/v1` | `qwen3.5:9b` | Local installation normally needs no key |
| Custom | User supplied | User supplied | OpenAI-compatible HTTPS endpoint; loopback HTTP is allowed |

For Ollama, install it from the [official download page](https://ollama.com/download),
then prepare the default model:

```bash
ollama pull qwen3.5:9b
```

Start Ollama if it is not already running, select **Ollama** in the WebUI, and
keep the default local URL. The application does not download Ollama models.

## Persistent non-secret defaults

Use `.env` for secrets and `config.toml` for non-secret local defaults. You only
need `config.toml` when the built-in defaults do not suit your setup:

```bash
cp config.example.toml config.toml
```

| Setting | Purpose | Configure it when |
| --- | --- | --- |
| `app.storage_dir` | Stores job manifests, uploaded copies, caches, and generated MP4 files | You want outputs on another local disk or directory |
| `app.model_cache_dir` | Stores the optional FunASR/SenseVoiceSmall model cache | You enabled voice input or want model files on another disk |
| `app.font_path` | Supplies the initial font path used by text-style controls | The system fallback does not render Chinese consistently |
| `llm.preset` | Selects the LLM shown at startup | You normally use a provider other than OpenAI |
| `llm.base_url` | Supplies the startup OpenAI-compatible endpoint | You use a proxy, a custom service, or non-default endpoint |
| `llm.model` | Supplies the startup model ID | Your account uses a different authorized model |

Never put API keys in `config.toml`. Values currently shown in the WebUI take
precedence over these startup defaults. A saved task records the rendering
request needed for regeneration, but never records provider credentials.

## WebUI settings: what they affect

| Group | Main controls | When to change them |
| --- | --- | --- |
| Basic Settings | Interface language; LLM provider, key, URL, model; remote image provider and key | Change only the providers you use. Preset URL/model values normally need no editing. |
| Word Content | Topic, word count, AI generation, voice input, phonetic mode, editable vocabulary | Always review the final editable list. Voice input requires the optional ASR install. |
| Canvas and Materials | 9:16/16:9 output, required background, image-material switch, local/remote source | Choose the aspect and background for every new video. Disable materials for text-only cards. |
| Lesson Flow | Question segment/text/style, question voice, countdown progress bar | Enable these when you want a quiz pause before revealing each answer. Progress and question narration only take effect with the question segment. |
| Narration | Chinese, fast English, and slow English switches, repeat counts, voices, rates, and volumes | By default, fast English plays once. Enable the other tracks only when needed. A disabled track, or a track with zero repeats, produces no audio. |
| Visual Style | English/phonetic/Chinese font, size, fill, outline and offsets; material assignment, candidate pool, size, mask, fit and offsets | Change when adapting branding or layout. `-1` means automatic centering; `cover` fills by cropping, `contain` shows the whole asset, and `stretch` may distort it. |
| Audio and Output | Optional local music, music volume, narration ducking, output FPS | Add only licensed music. Raise FPS for smoother motion at the cost of rendering time and file size. |
| Regenerate Existing Task | Load a task by ID, review its saved words, optionally choose replacement materials by word | Use to revise a saved job while keeping every previous completed MP4. |

Custom LLM endpoints must use HTTPS; loopback HTTP is allowed for local
services. Credentials are kept separate by provider, and custom-endpoint
credentials are isolated by HTTPS origin. URL fragments are rejected, and
`api-version` is the only supported query parameter. Put all authentication
data in the API key field. Pexels and Pixabay provide still images, not videos
or music.

## Script formats

Automatic phonetics use repeating two-line groups:

```text
苹果
apple
香蕉
banana
```

Manual phonetics use repeating three-line groups:

```text
苹果
apple
/ˈæpəl/
```

Blank lines are ignored. Manual mode never requires an LLM. In automatic mode,
an edited script with missing phonetics uses the configured compatible LLM to
complete only the phonetic field; responses that change the entered Chinese or
English are rejected.

## Advanced layout and narration

New videos created in the WebUI use a fixed 1080×1920 portrait canvas or a
1920×1080 landscape canvas. Question, media, progress, English, phonetic, and
Chinese elements all support top, bottom, left, and right offsets. Enter `-1`
to center an element automatically. Setting one edge anchors the element to
that edge; setting two opposite edges centers it within the remaining space.

Question segments play the bundled, project-owned countdown cue before the
question narration. The progress bar follows the combined audio duration.
Answer narration always plays in this order: Chinese, fast English, then slow
English, including all configured repeats. Disabled tracks and tracks with zero
repeats produce no audio. Chinese and slow English narration therefore start
disabled with zero repeats, while fast English starts enabled and plays once.
The bundled Edge TTS catalog contains all 14 Chinese and 47 English voices
returned by the service when the list was last updated; existing tasks can
still load voices that are no longer in the catalog. Question narration has
its own switch, repeat count, voice, rate, volume, and preview. It does not
reuse the fast English settings and defaults to one repeat with the default
fast English voice.

Edge TTS automatically retries temporary connection failures, timeouts, service
unavailability, and missing audio responses up to twice (three attempts total,
with delays of one and two seconds). Invalid settings, certificate failures,
and file-write errors are not retried. If synthesis still fails, the interface
reports the word number, narration track, reason, and attempt count in the
selected language. Question narration and voice previews identify their track
as well. Persistent failures still require checking your network or settings.

### Supported local media and rendering defaults

All uploads are decoded before use. A matching filename extension alone does
not make an invalid file acceptable. Local backgrounds and word images are limited
to 32 MiB; remote image downloads remain limited to 10 MiB. All images are limited
to 50 million decoded pixels. Local videos are limited to 128 MiB, five minutes,
3840 pixels per dimension, and
60 FPS, audio uploads to 32 MiB, and one generated timeline to one hour.

The background uploader shows a 32 MB limit. Mixed image/video uploaders show
128 MB, but images still have a 32 MiB limit; both limits are stated beside the
control. The uploader labels sizes as MB but measures them in MiB
(1 MiB = 1024×1024 bytes). The same limits apply to previews and regeneration.

| Input | Accepted types | Notes |
| --- | --- | --- |
| Background | `.png`, `.jpg`, `.jpeg`, `.webp` | Required; copied into the job. |
| Local vocabulary material | `.png`, `.jpg`, `.jpeg`, `.webp`, `.mp4`, `.mov`, `.m4v`, and `.webm` | Images and silent-in-output video clips may be mixed. |
| Background music | `.mp3`, `.wav`, `.m4a`, `.aac`, and `.ogg` | Optional and user-provided only. |
| Regeneration replacement | The same image and video types as local material | Load the task first, then choose a different word for each replacement file. |

Pexels and Pixabay remain still-image providers; they never supply local video
or music. Local video is looped or truncated to the card segment, begins at a
deterministic saved offset, and its source audio is always removed.

Unsplash is not currently integrated. Its API requires download tracking and
photographer/Unsplash attribution; support should only be added when those
credits can be carried through previews, jobs, and exported results. See the
[official Unsplash API guidelines](https://help.unsplash.com/en/articles/2511245-unsplash-api-guidelines).

Output may be set from 12 through 60 FPS and defaults to 24 FPS. Portrait
is fixed at 1080×1920 and landscape at 1920×1080 for new videos made in the
WebUI. The material mask is applied after one of these fit modes:

- `contain` preserves the aspect ratio, shows the whole asset, and letterboxes
  the unused part of the material box;
- `cover` preserves the aspect ratio and crops overflow to fill the box; it is
  the default;
- `stretch` fills both dimensions without preserving the aspect ratio.

Rectangle and circle masks use the same fitted pixels in preview and final
output. Video material moves only inside that masked box; card text and the
progress layer remain above it.

### Background music and ducking

Music is disabled by default. When enabled, the default volume is 12% and the
default narration-ducking amount is 65%. The selected local track loops
continuously across the full output. During countdown, question narration, or
answer narration, its gain is multiplied by
`1 - ducking_percent / 100`, with a short transition at speech boundaries. At
50% ducking, for example, music plays at half of its configured music volume
during those intervals. The application never searches for or downloads music.

## Choose images by word

Before previewing Pexels or Pixabay media, parse at least one vocabulary entry
and enter the key for the selected provider. Expand **Choose Images by Word**, then use
the word selector or **Previous word / Next word**, then click **Search candidates**.
The gallery shows up to **8 candidates by default**; adjust this from 1 to 20 in
**Visual Styles → Image Material Settings → Remote Candidate Pool**. Only the
current word is searched, and the gallery loads thumbnails first.
This optional panel is collapsed by default. Collapsing it keeps your choices;
if you skip it, generation uses automatic material assignment.

The compact overview at the top shows each word's material status: local upload,
manual selection, automatic selection, saved material, no results, or pending
automatic search. Click a word to jump to its gallery without changing its image.
Pending automatic search is normal; you can generate without reviewing every word.
Even if a search returns no results, you can upload material for that word.

**Image search keywords** defaults to the current English word. For a more
specific meaning, change `bank` to `river bank`, for example. Leave it blank to
use the original word. Keywords affect image search only, not vocabulary,
on-screen text, or narration. Each word keeps its own keywords, which are also
saved with the generated task for regeneration. Click **Search candidates**
after editing. Your selected image or upload stays in place until you choose
another image. Words without a selected material use their current keywords
for automatic search during generation.

The first search selects an image using your sequential or random assignment
setting. Click **Use this image** to change it; the selected image gets a blue
border and is downloaded at full size. Choices stay in place when you switch
words. **Search again** refreshes only the current word's candidates, without
changing any selections or the task seed. It may return the same images.

Changing material size, position, shape, or fit keeps the selected image.
Temporarily disabling materials or switching to local materials also preserves
your remote choices for when you switch back. Changing the word, provider,
candidate count, aspect ratio, or assignment mode resets the remote choices;
editing only the search keywords does not. Uploading or removing an override, or changing
visual settings, clears an outdated card preview; click **Preview** to update it.

If none fits, upload a local image or video beneath the gallery. This replaces
only the current word's material; choosing a candidate switches back to remote.
If a search or download fails, your previous selection is kept. Words you leave
untouched still use automatic remote search during generation. No results?
Upload your own material, or leave the word to use the neutral fallback.

Selected files are copied byte-for-byte into the task and reused during
regeneration. These session choices are not parameter presets: keep the generated
task directory to retain its materials after the session ends.

Preview cache files are session artifacts under ignored local storage. They are
not part of the source distribution and may be removed after the session ends.

## Preview individual cards

Above the generation buttons, choose a **Word to preview** and **Card type**,
then click **Preview** to check any word's layout. The default is the answer
card. Enable the question segment to preview a question card; disabling the
segment switches the preview back to the answer card. These controls affect
only the preview, not which words appear in the final video.

The preview uses the selected word's pinned image or local override. For a
local material pool, it follows the same sequential or seeded-random assignment
as generation. Video materials use a still frame selected with the same seed
and word index. If a remote image has not been selected, a labelled placeholder
is shown; previewing does not trigger an image search.

Changing the word, card type, or relevant visual settings clears the old preview.
Click **Preview** again to refresh it. Previews stay compact and are still images:
they do not generate audio or simulate the animated progress bar. Check the
generated video for sound and animation.

## Job manifest and architecture

`GenerationRequest` validates vocabulary, layout, FPS, media type and fit,
narration, music, and pinned previews. `JobStorage` copies uploaded inputs into
an isolated job and writes its manifest. The generation pipeline coordinates
providers, speech, Pillow card layers, and MoviePy/FFmpeg composition;
Streamlit remains the presentation layer. Remote provider adapters stream
HTTPS responses under a strict size limit, reject redirects to non-HTTPS URLs,
and validate content type, dimensions, and image decoding before saving a
file. Session credentials are never persisted.

Only schema version 3 is supported. Jobs with an earlier or unknown manifest
schema are rejected without changing their manifest or generated files.
Schema-v3 jobs can be regenerated from their task ID and saved inputs.

## Regeneration

After generation, the result panel shows a compact video preview, a download
button, and a copyable 32-character task ID. Download filenames include the task
ID and video version. Keep both the ID and its folder in `storage/`: the ID alone
cannot restore a deleted task.

To update a saved video:

1. Expand **Regenerate Existing Task** and choose a **Recent task** to load it.
   Alternatively, enter its ID and click **Load Task**.
2. Check the saved word list. This flow uses the task's saved vocabulary and
   settings; it neither reads nor overwrites the main editor.
3. Optionally upload replacement images or videos. For each file, choose the
   word by name from the dropdown. Multiple files cannot target the same word.
   If materials were disabled in the saved task, replacement controls are unavailable.
4. Click **Replace Material and Regenerate**. The result panel immediately shows
   the new video and download. No replacement files are required to regenerate.

Changing the ID requires loading the task again. A failed load leaves the main
editor and saved task files unchanged, and disables regeneration until a valid
task is loaded. Each successful regeneration creates the next numbered file,
such as `video-0002.mp4` or `video-0003.mp4`; earlier videos remain in the task directory.

**Recent tasks** lists up to 20 available tasks, newest update first, with their
topic (or words), saved word count, status, update time, and abbreviated ID.
Use **Refresh list** to pick up tasks created in another session. You can still
load tasks outside the list by entering their full ID. Unreadable or unsupported
tasks are omitted; the list never imports, repairs, or deletes them.

After loading a task, **Video versions** lets you preview and download any of its
saved videos. The newest version is selected by default, with the same compact
portrait or landscape preview used for new results. Switching versions does not
roll back the task, change its materials, replace the editor, or change the latest
result. Only completed videos registered in the task are listed. Times use the
server's time zone; version times are file modification times, not an audit log.

Successful replacements become the task's saved choices. Later regenerations keep
them unless you replace those words again. A failed regeneration does not commit
its new replacements or overwrite earlier completed videos.

Before rendering, the app snapshots the background, local media, uploaded
music, supported custom fonts (TTF, TTC, OTF, or OTC, up to 64 MiB), and pinned
remote previews. It rejects LLM base URLs whose decoded components contain
userinfo, credential assignments, or strings that resemble real secrets. URL
fragments and all query parameters except the non-secret `api-version` option
are also rejected, preventing URL-based credentials from reaching a task
manifest. Regeneration uses only saved job inputs; it does not need the
original upload path or session preview directory. Replacement uploads are
copied into the job before any cache or manifest state changes.

The manifest fingerprints the inputs relevant to each stage. Vocabulary,
materials, speech, and card layers are reused independently when their
fingerprints and artifacts remain valid. Material acquisition keys contain only
provider/source selection inputs (plus the SHA-256 of a selected local source),
so fit, mask, box, or offset changes keep downloaded bytes while invalidating
downstream cards/video overlays. The composition audit identity covers ordered
card/material bytes, foreground audio, durations and offsets, music, render,
and encoder-relevant settings; the manifest also records the final MP4 SHA-256.
The composition is rendered again for every regeneration. Question voice
settings invalidate question speech; material-byte, saved-video-offset, pinned,
or replacement changes invalidate affected downstream work. A new successful
composition always receives the next numbered MP4 and never overwrites an
earlier completed video. Regeneration replacements accept the
same supported local image and video formats as initial material uploads.

## 中文快速说明

- 界面默认使用简体中文 (`zh-CN`)，也可切换为英文 (`en-US`)。
- 背景图支持 PNG、JPG/JPEG、WebP；本地单词素材还支持 MP4、MOV、M4V、WebM，
  视频原声不会进入成片。
- 帧率范围为 12–60，默认 24；素材可选择完整显示 (`contain`)、裁切填满
  (`cover`，默认) 或拉伸 (`stretch`)，之后再应用圆形或矩形遮罩。
- 背景音乐只能由用户上传，支持 MP3、WAV、M4A、AAC、OGG；可设置音量与朗读
  压低比例。问题朗读有独立的开关、次数、声音、语速、音量和试听。
- 远程素材支持逐词浏览候选、手动选图和本地覆盖；已选文件会保存到任务中，未操作的词条
  仍在生成时自动搜索。
- 每次成功重新生成都会保留旧 MP4 并新增编号文件；请妥善保存 32 位任务 ID。

## Optional voice input

Voice transcription is deliberately excluded from the default environment
because its ML dependencies are large. If you initially used the standard
installation and later want voice input, run:

```bash
uv sync --frozen --extra asr
```

For a fresh installation, you can run this command directly instead of first
running `uv sync --frozen`. Keep `--extra asr` on later environment syncs so uv
does not remove the optional ASR dependencies.

The first transcription downloads SenseVoiceSmall and its `fsmn-vad` helper
model into the ignored `model_cache/` directory. Review the models' own
licenses and usage terms before redistributing them.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv: command not found` | Install uv from its official installation page, open a new terminal, and rerun `uv --version`. |
| FFmpeg or `ffprobe` is missing | Install FFmpeg, then confirm both `ffmpeg -version` and `ffprobe -version` work in the same terminal. |
| Port 8501 is already in use | Run `./webui.sh --server.port 8503` and open the printed URL. |
| An API returns 401 or 403 | Confirm the selected provider matches the key. A non-empty key entered in the UI overrides `.env` for that session. |
| Ollama connection fails | Start Ollama, run `ollama list`, and pull the exact model name shown in the WebUI if it is missing. |
| Edge TTS or remote images time out | Check network access to the selected service and retry; local material avoids the image-provider request but narration still uses Edge TTS. |
| Voice transcription is unavailable | Run `uv sync --frozen --extra asr`, restart the app, and allow the first model download to finish. |

The terminal usually contains the most useful diagnostic details. Generated
jobs and logs are stored under the configured `storage_dir`; credentials are
redacted from user-facing errors and task manifests.

## Fonts and media rights

The renderer tries common system CJK fonts and otherwise falls back to
Pillow's default font. For consistent Chinese typography, configure an
installed OFL-licensed font such as Noto Sans CJK. Fonts are referenced by
local path, snapshotted into each saved job for regeneration, and never copied
into the source repository.

You are responsible for confirming that uploaded background images, local image
or video materials, music, fonts, and media fetched through third-party
providers may be used, edited, synchronized with audio, published, and
redistributed for your intended purpose. Provider availability is not a license
grant. No fonts, music, demo media, or provider downloads are bundled with the
repository.

## Development

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run mypy src
uv pip check --python .venv/bin/python
uv run pip-audit --local --skip-editable
```

See [CHANGELOG.md](CHANGELOG.md) for release history. Before contributing or
redistributing the project, read [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), and [NOTICE.md](NOTICE.md).

Maintainers can also read the [architecture guide](docs/architecture.md) and
[roadmap](docs/roadmap.md). The architecture guide describes the current code;
the roadmap records criteria for possible future work.

## License and attribution

The project is licensed under the [MIT License](LICENSE). Third-party notices
and the licensing boundary for dependencies, assets, and user-provided media
are documented in [NOTICE.md](NOTICE.md).
