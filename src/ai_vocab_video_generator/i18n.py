"""Small complete translation catalog for the local Streamlit UI."""

from enum import StrEnum


class Locale(StrEnum):
    ZH_CN = "zh-CN"
    EN_US = "en-US"


DEFAULT_LOCALE = Locale.ZH_CN


_EN_US = {
    "title": "AI Vocab Video Generator",
    "caption": "Create narrated vocabulary learning videos.",
    "public_demo_notice": (
        "Public demo: temporary session-isolated storage, up to 5 words per video. "
        "Enter your own provider keys and download the result before leaving; keys and "
        "generated files are not retained after the session or server restarts."
    ),
    "public_demo_credentials_help": (
        "This public demo never uses the operator's API keys. Keys entered here stay only "
        "in your current browser session and are not written to task files."
    ),
    "public_demo_asr_disabled": (
        "Voice input is unavailable in the public demo because its local ASR model exceeds "
        "the free hosting budget. Enter the topic as text instead."
    ),
    "public_demo_word_limit": "The public demo supports at most {limit} words per video.",
    "public_demo_busy": ("Another video is being generated. Please wait a moment and try again."),
    "public_demo_result_help": (
        "This result is temporary. Download the MP4 before closing the page."
    ),
    "language": "Language",
    "basic_settings": "Basic Settings (:blue[click to expand])",
    "basic_settings_help": (
        "Choose service providers here. Keys entered in the UI stay only in the current "
        "session, override matching `.env` values, and are never written to task files."
    ),
    "llm_preset": "LLM Provider",
    "llm_openai_help": (
        "OpenAI preset: :red[API key required]. Get one from [OpenAI API keys]"
        "(https://platform.openai.com/api-keys); the preset fills the endpoint and model."
    ),
    "llm_deepseek_help": (
        "DeepSeek preset: :red[API key required]. Get one from the [DeepSeek platform]"
        "(https://platform.deepseek.com/api_keys); the preset fills the endpoint and model."
    ),
    "llm_moonshot_help": (
        "Moonshot preset: :red[API key required]. Get one from the [Kimi API Platform]"
        "(https://platform.kimi.com/console/api-keys); the preset fills the endpoint and "
        "flagship model."
    ),
    "llm_qwen_help": (
        "Qwen preset: :red[API key required]. Get one from [Alibaba Cloud Model Studio]"
        "(https://help.aliyun.com/en/model-studio/get-api-key); the preset fills the "
        "compatible endpoint and model."
    ),
    "llm_ollama_help": (
        "Ollama runs locally and normally needs no real key. Make sure the selected model is "
        "already installed and the endpoint is reachable."
    ),
    "llm_custom_help": (
        "Custom: :red[API key required]. It accepts an OpenAI-compatible HTTPS endpoint, or "
        "loopback HTTP for a local service. The key is bound to that endpoint's origin."
    ),
    "llm_openai_setup": (
        "**OpenAI setup**\n\n"
        "- **API Key:** [create one on the official platform]"
        "(https://platform.openai.com/api-keys)\n"
        "- **Base URL:** preset default `https://api.openai.com/v1`\n"
        "- **Model:** preset default `gpt-5.6-terra`; [view model documentation]"
        "(https://platform.openai.com/docs/models)"
    ),
    "llm_deepseek_setup": (
        "**DeepSeek setup**\n\n"
        "- **API Key:** [create one on the official platform]"
        "(https://platform.deepseek.com/api_keys)\n"
        "- **Base URL:** preset default `https://api.deepseek.com`\n"
        "- **Model:** preset default `deepseek-v4-flash`; [view the model list]"
        "(https://api-docs.deepseek.com/api/list-models/)"
    ),
    "llm_moonshot_setup": (
        "**Moonshot (Kimi) setup**\n\n"
        "- **API Key:** [create one on the Kimi API Platform]"
        "(https://platform.kimi.com/console/api-keys)\n"
        "- **Base URL:** preset default `https://api.moonshot.cn/v1`\n"
        "- **Model:** preset default cost-effective general model `kimi-k2.6` "
        "(thinking disabled); [view the model list]"
        "(https://platform.kimi.com/docs/models)"
    ),
    "llm_qwen_setup": (
        "**Qwen setup**\n\n"
        "- **API Key:** [get one from Alibaba Cloud Model Studio]"
        "(https://help.aliyun.com/en/model-studio/get-api-key)\n"
        "- **Base URL:** preset default "
        "`https://dashscope.aliyuncs.com/compatible-mode/v1`\n"
        "- **Model:** preset default `qwen3.7-flash`; [view the model list]"
        "(https://help.aliyun.com/en/model-studio/list-models)"
    ),
    "llm_ollama_setup": (
        "**Ollama setup**\n\n"
        "- **API Key:** not required for a default local installation\n"
        "- **Base URL:** preset default `http://localhost:11434/v1`\n"
        "- **Model:** preset default `qwen3.5:9b`; [view the Ollama model library]"
        "(https://ollama.com/library)"
    ),
    "llm_custom_setup": (
        "**Custom provider setup**\n\n"
        "- **API Key:** obtain it from your provider\n"
        "- **Base URL:** enter an OpenAI-compatible HTTPS endpoint; local loopback HTTP "
        "is allowed\n"
        "- **Model:** enter the exact model ID authorized by your provider"
    ),
    "api_key": "API Key",
    "api_key_openai": (
        "API Key (:red[required]; [get an OpenAI API key](https://platform.openai.com/api-keys))"
    ),
    "api_key_deepseek": (
        "API Key (:red[required]; [get a DeepSeek API key](https://platform.deepseek.com/api_keys))"
    ),
    "api_key_moonshot": (
        "API Key (:red[required]; [get a Moonshot API key]"
        "(https://platform.kimi.com/console/api-keys))"
    ),
    "api_key_qwen": (
        "API Key (:red[required]; [get an Alibaba Cloud Model Studio API key]"
        "(https://help.aliyun.com/en/model-studio/get-api-key))"
    ),
    "api_key_ollama": "API Key (optional for local Ollama)",
    "api_key_custom": "API Key (:red[required; obtain it from your provider console])",
    "pexels_key_help": (":red[required]; [request a Pexels API key](https://www.pexels.com/api/)"),
    "pixabay_key_help": (":red[required]; [get a Pixabay API key](https://pixabay.com/api/docs/)"),
    "base_url": "Base URL",
    "model": "Model",
    "remote_provider": "Image Material Provider",
    "remote_provider_help": (
        ":red[API key required] when using remote materials. Pexels and Pixabay currently "
        "provide still images only. Apply through [Pexels](https://www.pexels.com/api/) or "
        "[Pixabay](https://pixabay.com/api/docs/), then select the matching provider."
    ),
    "pexels_key": "Pexels API Key",
    "pixabay_key": "Pixabay API Key",
    "test_llm_connection": "Test LLM Connection",
    "test_llm_connection_help": (
        "Sends one small real request that generates a structured test entry. The provider may "
        "charge a minimal amount."
    ),
    "test_image_connection": "Test Image Service",
    "test_image_connection_help": (
        "Runs one lightweight metadata search to validate the key; no image is downloaded."
    ),
    "record_topic": "Enter Topic by Voice (optional)",
    "topic_settings": "Topic and Vocabulary",
    "topic_settings_help": (
        ":red[AI can generate] vocabulary from a topic. You can also edit or paste line-based "
        "vocabulary directly; word count affects AI generation only."
    ),
    "transcribe": "Transcribe and Fill Topic",
    "topic": "Video Topic (enter a topic to have :red[AI generate] the vocabulary)",
    "word_count": "Word Count",
    "word_count_help": "Used only when AI generates vocabulary; it does not limit manual input.",
    "generate_script": "Use AI to generate vocabulary from the topic",
    "script": (
        "Vocabulary (:blue[generate it from the topic with AI or edit it manually for the "
        "current phonetic mode])"
    ),
    "background_settings": "Background Settings",
    "background_settings_help": (
        "You :red[must upload] a PNG, JPG, or WebP background; it is cropped to the selected "
        "standard output aspect."
    ),
    "background_image": "Background Image",
    "phonetic_settings": "Phonetic Settings",
    "phonetic_settings_help": (
        "Automatic phonetics uses the configured LLM only for missing phonetics. Manual mode "
        "expects Chinese, English, and phonetic lines; the two modes are mutually exclusive."
    ),
    "automatic_phonetic": "Automatic Phonetics",
    "manual_phonetic": "Manual Phonetics",
    "phonetic_disabled_help": (
        "Turn both options off to hide phonetics. Automatic and manual modes are mutually "
        "exclusive."
    ),
    "canvas_material_settings": "Canvas and Materials",
    "canvas_material_settings_help": (
        "Choose the output aspect, required background, and the material source used for each "
        "word. Less common size and placement controls remain under Visual Styles."
    ),
    "lesson_flow_settings": "Question and Progress",
    "narration_group_settings": "Narration Settings",
    "visual_style_settings": "Visual Styles",
    "audio_output_settings": "Audio and Output",
    "video_settings": "Video Settings",
    "advanced_video_settings": "Advanced Video Settings (:blue[click to expand])",
    "video_settings_help": "Choose the fixed output aspect. Frame rate is under advanced settings.",
    "aspect": "Output Aspect",
    "aspect_help": (
        "Portrait is fixed at 1080 × 1920 and landscape at 1920 × 1080. Switching aspect also "
        "loads matching layout defaults and remote-search orientation."
    ),
    "portrait": "Portrait 9:16",
    "landscape": "Landscape 16:9",
    "material_source": "Material Source",
    "material_source_help": (
        "Local uploads use your selected images or videos. Remote search downloads one still "
        "image per English word from the provider selected in Basic Settings."
    ),
    "local_uploads": "Local Uploads",
    "remote_search": "Remote Search",
    "word_material_review": "Choose Images by Word (Optional)",
    "word_material_review_help": (
        "Choose from the image candidates, or upload your own material if none fits."
    ),
    "local_material_override": "None of these fit? Use your own material",
    "material_current_word": "Current word",
    "material_previous_word": "Previous word",
    "material_next_word": "Next word",
    "material_candidate_count": "{count} candidates",
    "material_search": "Search candidates",
    "material_search_query": "Image search keywords",
    "material_search_query_help": (
        "Defaults to this word. Try a more specific phrase, such as 'river bank'. "
        "Leave blank to use the word. Only image search is affected, not vocabulary or narration. "
        "Searching again keeps your current image until you choose another."
    ),
    "material_search_again": "Search again",
    "material_search_help": (
        "Refresh this word's candidates without changing any selected material. "
        "Results may be unchanged."
    ),
    "material_searching": "Searching image candidates…",
    "material_search_prompt": "Click Search candidates to browse images for this word.",
    "material_candidate": "Candidate {number}",
    "material_use": "Use this image",
    "material_selected": "✓ Selected",
    "material_thumbnail_failed": "Thumbnail unavailable. You can still try selecting this image.",
    "material_upload_prompt": "Upload an image or video",
    "material_upload_help": (
        "An upload replaces material for {word} only; other words are unaffected."
    ),
    "material_remove_upload": "Remove local upload",
    "material_using_candidate": "Using candidate {number} · {mode}",
    "material_manual": "Selected by you",
    "material_auto": "Selected automatically",
    "material_untouched_help": "Words you leave unchanged will use automatic image selection.",
    "remote_material_automatic": "No material is pinned; generation will search this word.",
    "remote_material_pinned": "The current material is pinned for generation.",
    "remote_material_missing": (
        "No suitable remote image was found. Search again or upload your own material."
    ),
    "local_material_selected": "This local file overrides the remote result for this word.",
    "selection_mode": "Material Assignment",
    "selection_mode_help": (
        "Sequential cycles through candidates by entry order. Stable random assignment uses "
        "the saved task seed, so regeneration keeps the same choices."
    ),
    "sequential": "Sequential",
    "random": "Random",
    "question_settings": "Question Settings",
    "question_settings_help": (
        "When disabled, question text, question narration, and the progress bar do not take effect."
    ),
    "show_question": "Enable Question Segment",
    "question_text": "Question Text",
    "advanced_question_settings": "Advanced Question Settings (:blue[click to expand])",
    "progress_settings": "Progress Bar Settings",
    "progress_settings_help": (
        "Only takes effect when Enable Question Segment is on. The bar appears only during "
        "each question segment and follows the combined duration of the countdown and narration."
    ),
    "show_progress": "Enable Progress Bar",
    "advanced_progress_settings": "Advanced Progress Bar Settings (:blue[click to expand])",
    "start_color": "Start Color",
    "end_color": "End Color",
    "material_settings": "Image Material Settings",
    "material_settings_help": (
        "When disabled, material source, uploads or search, assignment, size, shape, and "
        "position do not take effect. Local files may be images or videos; Pexels and Pixabay "
        "currently return still images only."
    ),
    "show_material": "Enable Image Materials",
    "advanced_material_settings": "Advanced Material Settings (:blue[click to expand])",
    "material_uploads": "Material Images",
    "material_uploads_mixed": "Material Images and Videos",
    "circle": "Circle",
    "rectangle": "Rectangle",
    "shape": "Shape",
    "chinese_narration": "Chinese Narration",
    "fast_english_narration": "Fast English Narration",
    "slow_english_narration": "Slow English Narration",
    "narration_settings_help": (
        ":red[voice must match the text language]. A disabled track or zero repeats produces "
        "no audio. The voice list includes all bundled Edge TTS voices for the track language; "
        "volume and rate range from -100% to +100%, and Play Voice is only a preview."
    ),
    "advanced_narration_settings": "Advanced Narration Settings (:blue[click to expand])",
    "enabled": "Enabled",
    "repeats": "Repeats",
    "voice": "Voice",
    "voice_gender_female": "Female",
    "voice_gender_male": "Male",
    "volume": "Volume",
    "rate": "Rate",
    "play_voice": "Play Voice",
    "play_voice_help": (
        "Synthesizes a short sample to test the selected Edge TTS connection and voice."
    ),
    "english_text": "English Text",
    "phonetic_text": "Phonetic Text",
    "chinese_text": "Chinese Text",
    "text_settings_help": (
        "Disable a text layer to hide it. Font path must name a supported local font; size, "
        "color, weight, stroke, and offsets are applied to preview and final video. Use -1 on "
        "an offset for automatic positioning."
    ),
    "advanced_text_settings": "Advanced Text Settings (:blue[click to expand])",
    "font_path": "Font Path",
    "font_size": "Font Size",
    "fill_color": "Fill Color",
    "weight": "Visual Weight",
    "stroke_color": "Stroke Color",
    "stroke_width": "Stroke Width",
    "top": "Top",
    "bottom": "Bottom",
    "left": "Left",
    "right": "Right",
    "width": "Width",
    "height": "Height",
    "question_label": "Question",
    "english_label": "English",
    "phonetic_label": "Phonetic",
    "chinese_label": "Chinese",
    "progress_label": "Progress Bar",
    "material_label": "Image Material",
    "fast_english_label": "Fast English",
    "slow_english_label": "Slow English",
    "font_suffix": "Font",
    "font_size_suffix": "Font Size",
    "font_color_suffix": "Font Color",
    "font_weight_suffix": "Font Weight",
    "stroke_color_suffix": "Stroke Color",
    "stroke_weight_suffix": "Stroke Weight",
    "top_margin_suffix": "Top Margin",
    "bottom_margin_suffix": "Bottom Margin",
    "left_margin_suffix": "Left Margin",
    "right_margin_suffix": "Right Margin",
    "width_suffix": "Width",
    "height_suffix": "Height",
    "shape_suffix": "Shape",
    "start_color_suffix": "Start Color",
    "end_color_suffix": "End Color",
    "narration_repeats_suffix": "Narration Repeats",
    "narration_voice_suffix": "Narration Voice",
    "narration_volume_suffix": "Narration Volume (+100 means +100%)",
    "narration_rate_suffix": "Narration Rate (+100 means +100%)",
    "generate_video": "Generate Video",
    "preview": "Preview",
    "preview_word": "Word to preview",
    "preview_card_type": "Card type",
    "preview_answer": "Answer card",
    "preview_question": "Question card",
    "preview_sample_word": "Sample word (no vocabulary entered)",
    "preview_question_help": "Enable the question segment to preview its card.",
    "preview_help": (
        "Select a word and card type, then click Preview. This is a still image without audio "
        "or the animated progress bar; it does not change the video content."
    ),
    "preview_placeholder": (
        "This word has no selected material yet. The preview uses a placeholder; "
        "choose an image or upload your own to check the actual layout."
    ),
    "result": "Result",
    "download": "Download MP4",
    "task_id": "Task ID",
    "result_task_help": (
        "Copy this ID and keep its folder in storage/. The ID alone is not a backup."
    ),
    "result_version": "Video file",
    "material_overview_help": (
        "Click a word to review its image. Pending automatic search is normal; "
        "you do not need to choose every image manually."
    ),
    "material_status_upload": "Local upload",
    "material_status_manual": "Manually selected",
    "material_status_auto": "Auto-selected",
    "material_status_saved": "Saved image",
    "material_status_empty": "No results · upload available",
    "material_status_pending": "Pending auto-search",
    "recent_tasks": "Recent tasks",
    "refresh_tasks": "Refresh list",
    "recent_tasks_placeholder": "Choose a task to load",
    "recent_tasks_empty": "No available tasks",
    "recent_tasks_help": (
        "Shows up to 20 tasks, most recently updated first (server time). "
        "Refresh to include tasks created elsewhere. You can also enter any task ID below. "
        "The count is the number of saved words."
    ),
    "recent_tasks_unavailable": (
        "The task list is unavailable. You can still try loading a task by ID."
    ),
    "recent_task_option": "{title} · Words: {count} · {status} · {time} · {task_id}",
    "untitled_task": "Untitled task",
    "task_status_queued": "Queued",
    "task_status_running": "Running",
    "task_status_complete": "Complete",
    "task_status_failed": "Failed",
    "video_history": "Video versions",
    "video_history_version": "Version to preview",
    "video_history_current": "Current output",
    "video_history_empty": "This task has no saved video yet.",
    "video_history_help": (
        "Preview or download a saved version. This does not roll back the task "
        "or change your vocabulary, settings, or latest result. "
        "Times are file modification times in the server's time zone."
    ),
    "video_history_unavailable": (
        "This video is unavailable. Its file may be missing or the task may have changed. "
        "Try loading the task again."
    ),
    "download_history": "Download this version",
    "load_task": "Load Task",
    "task_load_required": "Enter a task ID and load its saved vocabulary before regenerating.",
    "task_load_failed": (
        "Could not load this task. Check the ID and make sure its folder is intact in storage/. "
        "Your current vocabulary and saved files have not been changed."
    ),
    "task_no_vocabulary": "This task has no saved vocabulary yet and cannot be regenerated.",
    "task_loaded": "Loaded task {task_id} · Words: {count}",
    "saved_chinese": "Chinese",
    "saved_phonetic": "Phonetic",
    "saved_task_help": (
        "Confirm the saved words above. Regeneration uses this task's vocabulary and settings, "
        "not the editor above. Replacement files are optional; each must target a different word. "
        "Previous videos are kept."
    ),
    "task_materials_disabled": (
        "Materials were disabled in this task. You can regenerate it, "
        "but replacement materials would have no effect."
    ),
    "replacement_word": "Replace material for",
    "replacement_word_option": "Word {number}: {word}",
    "replacement_images": "Replacement Materials",
    "replacement_index": "Entry Index",
    "regenerate": "Replace Material and Regenerate",
    "task_directory": "Task Directory",
    "open_folder": "Open Folder",
    "preparing": "Preparing job",
    "progress_vocabulary": "Vocabulary is ready",
    "progress_materials": "Vocabulary materials are ready",
    "progress_narration": "Narration is ready",
    "progress_cards": "Vocabulary cards are ready",
    "progress_composing": "Composing video",
    "progress_complete": "Video is ready",
    "missing_topic": "Enter a video topic before using AI to generate vocabulary.",
    "missing_content": "Enter a video topic or provide valid vocabulary directly.",
    "missing_background": "Upload a background image before generation.",
    "missing_llm_key": "Enter an API key before AI generation or automatic phonetic completion.",
    "missing_materials": "Upload at least one local image or video material.",
    "missing_provider_key": "Enter the selected image material provider key.",
    "script_invalid": "The vocabulary format is invalid; check it against the phonetic mode.",
    "generation_complete": "Video generated successfully.",
    "regeneration_complete": "Video regenerated successfully.",
    "llm_connection_success": "LLM connected. The key, endpoint, and model are working.",
    "image_connection_success": ("Image service connected. The current key can search for images."),
    "error_speech_generic": "Speech generation failed. Check your network and narration settings.",
    "error_speech_summary": "Speech generation failed. {reason} Attempts: {attempts}.",
    "error_speech_word": (
        "Speech generation failed for word {number} ({track}). {reason} Attempts: {attempts}."
    ),
    "error_speech_track": "Speech generation failed during {track}. {reason} Attempts: {attempts}.",
    "error_speech_track_question": "question narration",
    "error_speech_track_zh": "Chinese narration",
    "error_speech_track_fast": "English narration",
    "error_speech_track_slow": "slow English narration",
    "error_speech_connection": "Could not connect to Edge TTS. Check your network and try again.",
    "error_speech_timeout": "Edge TTS timed out. Check your network and try again.",
    "error_speech_service": "Edge TTS is temporarily unavailable or rate-limited. Try again later.",
    "error_speech_rejected": (
        "Edge TTS rejected the request. Check the voice and service availability."
    ),
    "error_speech_empty": "Edge TTS returned no audio. Check the voice and text, then try again.",
    "error_speech_settings": "Check the voice, rate, volume, and text settings.",
    "error_speech_file": (
        "Could not save the audio file. Check available disk space and storage permissions."
    ),
    "error_speech_output": "The audio output is missing, empty, or exceeds the size limit.",
    "error_speech_certificate": (
        "The secure connection to Edge TTS failed. Check your network and certificates."
    ),
    "error_speech_unknown": "Check your network and narration settings, then try again.",
    "error_llm_auth": (
        "The vocabulary provider rejected the API key. Check the credential for the selected "
        "provider."
    ),
    "error_llm_timeout": (
        "The vocabulary provider request timed out. Try again and check the network or service "
        "status."
    ),
    "error_llm_unavailable": (
        "Cannot reach the vocabulary provider. If you use Ollama, make sure it is running; "
        "otherwise check the network."
    ),
    "error_llm_not_found": (
        "The configured model or API endpoint was not found. Check the model name and provider "
        "URL. For Ollama, run ollama pull <model>."
    ),
    "error_image_auth": "The image service rejected the API key. Check the selected provider key.",
    "error_image_timeout": "The image service timed out. Try again and check the network.",
    "error_image_unavailable": "Cannot reach the image service. Check the network and try again.",
    "error_image_invalid": "The image service returned an invalid response. Try again later.",
    "error_upload_too_large": "The upload is too large. Choose a smaller file.",
    "error_upload_size_details": (
        "The file is {actual:.2f} MiB, exceeding the {limit:g} MiB limit for this file type."
    ),
    "image_upload_limits": "Each image: up to 32 MiB and 50 million decoded pixels.",
    "mixed_upload_limits": (
        "Per file: images up to 32 MiB; videos up to 128 MiB. "
        "The uploader's 128 MB label is the video limit, not the image limit."
    ),
    "error_material_image_decode": (
        "The selected material image could not be read. Check that it is not damaged and uses a "
        "supported format."
    ),
    "error_material_video_decode": (
        "The selected material video could not be read. Check that it is not damaged, too long, "
        "or in an unsupported format."
    ),
    "error_material_type": (
        "The selected material file type is unsupported. Upload a supported image or video."
    ),
    "error_generation_interrupted": (
        "Video generation was interrupted. The incomplete video file was removed."
    ),
    "duplicate_replacements": "Each replacement file must target a different word.",
    "regeneration_settings": "Regenerate Existing Task",
    "regeneration_settings_help": (
        "Load a saved task and confirm its vocabulary. Replacement materials are optional; "
        "choose which words to update. The task folder must still exist in storage/."
    ),
    "render_settings": "Render Settings",
    "fps": "Frame Rate (FPS)",
    "fps_help": (
        "The final H.264 video is encoded at 12–60 FPS. The default 24 FPS balances smoothness "
        "and render time."
    ),
    "fit_mode": "Material Fill Mode",
    "fit_mode_help": (
        "Contain shows the whole asset with empty space; cover fills the box and may crop edges; "
        "stretch fills the box and may distort the asset."
    ),
    "candidate_pool_size": "Remote Candidate Pool",
    "candidate_pool_size_help": (
        "Remote search requests this many candidates for each word before applying sequential "
        "or stable-random assignment. It does not affect local uploads."
    ),
    "contain": "Contain",
    "cover": "Cover",
    "stretch": "Stretch",
    "video_upload_help": "Upload images and videos (PNG, JPG, WebP, MP4, MOV, M4V, WebM).",
    "background_music_settings": "Background Music",
    "background_music_settings_help": (
        "Music is optional and local-only. It loops for the full video; narration ducking lowers "
        "it only while countdown or speech audio is active."
    ),
    "background_music_enabled": "Enable Background Music",
    "background_music_file": "Local Music File",
    "advanced_music_settings": "Advanced Music Settings (:blue[click to expand])",
    "music_volume": "Music Volume (%)",
    "music_ducking": "Narration Ducking (%)",
    "missing_music": "Upload a supported local music file when background music is enabled.",
    "question_narration": "Question Narration",
    "question_narration_settings": "Question Narration Settings",
    "question_narration_settings_help": (
        "Only takes effect when Enable Question Segment is on. Disabling narration or setting "
        "repeats to 0 removes spoken question audio but keeps the countdown cue."
    ),
}

_ZH_CN = {
    "title": "AI 单词视频生成器",
    "caption": "一键生成带朗读的单词学习视频。",
    "public_demo_notice": (
        "公开体验版：每个浏览器会话使用独立的临时存储，每段视频最多 5 个单词。"
        "请填写你自己的服务商密钥，并在离开前下载结果；会话结束或服务器重启后，"
        "密钥和生成文件都不会保留。"
    ),
    "public_demo_credentials_help": (
        "公开体验版不会使用部署者的 API 密钥。你在这里填写的密钥只保留在当前浏览器"
        "会话中，不会写入任务文件。"
    ),
    "public_demo_asr_disabled": (
        "公开体验版不提供语音输入，因为本地 ASR 模型会超出免费托管资源；请直接输入文字主题。"
    ),
    "public_demo_word_limit": "公开体验版每段视频最多支持 {limit} 个单词。",
    "public_demo_busy": "当前已有视频正在生成，请稍等片刻后重试。",
    "public_demo_result_help": "该结果仅临时保留，请在关闭页面前下载 MP4。",
    "language": "语言",
    "basic_settings": "基础设置 (:blue[点击展开])",
    "basic_settings_help": (
        "在这里选择服务商。界面输入的密钥只保留在当前会话，优先于对应的 `.env` "
        "配置，并且不会写入任务文件。"
    ),
    "llm_preset": "大模型服务商",
    "llm_openai_help": (
        "OpenAI 预设：:red[需要 API 密钥]，可前往 [OpenAI API 密钥]"
        "(https://platform.openai.com/api-keys)申请；接口地址和模型会自动填入。"
    ),
    "llm_deepseek_help": (
        "DeepSeek 预设：:red[需要 API 密钥]，可前往 [DeepSeek 平台]"
        "(https://platform.deepseek.com/api_keys)申请密钥；接口地址和模型会自动填入。"
    ),
    "llm_moonshot_help": (
        "Moonshot 预设：:red[需要 API 密钥]，可前往 [Kimi API 平台]"
        "(https://platform.kimi.com/console/api-keys)申请；接口地址和默认模型会自动填入。"
    ),
    "llm_qwen_help": (
        "Qwen 预设：:red[需要 API 密钥]，可前往 [阿里云百炼 API Key]"
        "(https://help.aliyun.com/zh/model-studio/get-api-key)获取；接口地址和模型会自动填入。"
    ),
    "llm_ollama_help": (
        "Ollama 在本机运行，通常不需要真实密钥；请先确认所选模型已安装，并且接口地址可访问。"
    ),
    "llm_custom_help": (
        "自定义服务商:red[需要 API 密钥]，支持兼容 OpenAI 的 HTTPS 接口，也允许本机回环"
        "地址使用 HTTP；密钥只绑定到当前接口来源。"
    ),
    "llm_openai_setup": (
        "**OpenAI 配置说明**\n\n"
        "- **API Key：** [前往官网申请](https://platform.openai.com/api-keys)\n"
        "- **接口地址：** 预设默认 `https://api.openai.com/v1`\n"
        "- **模型：** 预设默认 `gpt-5.6-terra`；"
        "[查看模型文档](https://platform.openai.com/docs/models)"
    ),
    "llm_deepseek_setup": (
        "**DeepSeek 配置说明**\n\n"
        "- **API Key：** [前往官网申请](https://platform.deepseek.com/api_keys)\n"
        "- **接口地址：** 预设默认 `https://api.deepseek.com`\n"
        "- **模型：** 预设默认 `deepseek-v4-flash`；"
        "[查看模型列表](https://api-docs.deepseek.com/api/list-models/)"
    ),
    "llm_moonshot_setup": (
        "**Moonshot 配置说明**\n\n"
        "- **API Key：** [前往 Kimi API 平台申请]"
        "(https://platform.kimi.com/console/api-keys)\n"
        "- **接口地址：** 预设默认 `https://api.moonshot.cn/v1`\n"
        "- **模型：** 预设默认高性价比通用模型 `kimi-k2.6`，关闭思考模式；"
        "[查看模型列表](https://platform.kimi.com/docs/models)"
    ),
    "llm_qwen_setup": (
        "**通义千问 Qwen 配置说明**\n\n"
        "- **API Key：** [前往阿里云百炼申请]"
        "(https://help.aliyun.com/zh/model-studio/get-api-key)\n"
        "- **接口地址：** 预设默认 "
        "`https://dashscope.aliyuncs.com/compatible-mode/v1`\n"
        "- **模型：** 预设默认 `qwen3.7-flash`；"
        "[查看模型列表](https://help.aliyun.com/zh/model-studio/list-models)"
    ),
    "llm_ollama_setup": (
        "**Ollama 配置说明**\n\n"
        "- **API Key：** 本地默认安装不需要填写\n"
        "- **接口地址：** 预设默认 `http://localhost:11434/v1`\n"
        "- **模型：** 预设默认 `qwen3.5:9b`；"
        "[查看 Ollama 模型库](https://ollama.com/library)"
    ),
    "llm_custom_setup": (
        "**自定义服务商配置说明**\n\n"
        "- **API Key：** 请到所用服务商后台获取\n"
        "- **接口地址：** 填写兼容 OpenAI 的 HTTPS 地址；本机回环地址允许使用 HTTP\n"
        "- **模型：** 填写服务商已授权的准确模型 ID"
    ),
    "api_key": "API 密钥",
    "api_key_openai": (
        "API 密钥 (:red[必填]；[获取 OpenAI API 密钥](https://platform.openai.com/api-keys))"
    ),
    "api_key_deepseek": (
        "API 密钥 (:red[必填]；[获取 DeepSeek API 密钥](https://platform.deepseek.com/api_keys))"
    ),
    "api_key_moonshot": (
        "API 密钥 (:red[必填]；[获取 Moonshot API 密钥](https://platform.kimi.com/console/api-keys))"
    ),
    "api_key_qwen": (
        "API 密钥 (:red[必填]；[获取阿里云百炼 API Key]"
        "(https://help.aliyun.com/zh/model-studio/get-api-key))"
    ),
    "api_key_ollama": "API 密钥 (本地 Ollama 可不填)",
    "api_key_custom": "API 密钥 (:red[必填，请到所用服务商后台获取])",
    "pexels_key_help": ":red[必填]；[申请 Pexels API 密钥](https://www.pexels.com/api/)",
    "pixabay_key_help": ":red[必填]；[获取 Pixabay API 密钥](https://pixabay.com/api/docs/)",
    "base_url": "接口地址",
    "model": "模型",
    "remote_provider": "图片素材服务商",
    "remote_provider_help": (
        "使用远程素材时:red[需要 API 密钥]。当前 Pexels 和 Pixabay 只提供静态图片；"
        "可前往 [Pexels](https://www.pexels.com/api/) 或 "
        "[Pixabay](https://pixabay.com/api/docs/) 申请，并选择与当前密钥对应的服务商。"
    ),
    "pexels_key": "Pexels API 密钥",
    "pixabay_key": "Pixabay API 密钥",
    "test_llm_connection": "测试大模型连接",
    "test_llm_connection_help": (
        "发送一次真实的小请求并生成一条结构化测试词条；服务商可能收取极少量费用。"
    ),
    "test_image_connection": "测试图片服务连接",
    "test_image_connection_help": "只执行一次轻量元数据搜索以验证密钥，不会下载图片。",
    "record_topic": "语音输入主题 (可选)",
    "topic_settings": "主题与单词信息",
    "topic_settings_help": (
        "输入主题后，可让:red[AI 自动生成]单词信息；你也可以直接编辑或粘贴分行内容。"
        "单词数量只影响 AI 生成。"
    ),
    "transcribe": "识别并填入主题",
    "topic": "视频主题 (输入主题后，可让 :red[AI 自动生成]单词信息)",
    "word_count": "单词数量",
    "word_count_help": "仅用于 AI 根据主题生成单词信息，不限制手动输入的数量。",
    "generate_script": "使用 AI 根据主题生成单词信息",
    "script": "单词信息 (:blue[可由 AI 根据主题生成，也可按当前音标模式手动编辑])",
    "background_settings": "背景设置",
    "background_settings_help": (
        ":red[必须上传] PNG、JPG 或 WebP 背景图，程序会按所选标准输出比例进行裁切。"
    ),
    "background_image": "背景图片",
    "phonetic_settings": "音标设置",
    "phonetic_settings_help": (
        "自动音标只在缺少音标时调用已配置的大模型；手动模式使用中文、英文、音标三行格式。"
        "两种模式互斥。"
    ),
    "automatic_phonetic": "自动音标",
    "manual_phonetic": "手动音标",
    "phonetic_disabled_help": "两个选项都关闭时不显示音标；自动音标与手动音标互斥。",
    "canvas_material_settings": "画面与素材",
    "canvas_material_settings_help": (
        "在这里选择输出比例、必需的背景图片和每个单词使用的素材来源；不常调整的尺寸与位置"
        "统一放在“画面样式”中。"
    ),
    "lesson_flow_settings": "问答与进度",
    "narration_group_settings": "朗读设置",
    "visual_style_settings": "画面样式",
    "audio_output_settings": "音频与输出",
    "video_settings": "视频设置",
    "advanced_video_settings": "高级视频参数 (:blue[点击展开])",
    "video_settings_help": "设置固定输出比例；帧率位于高级参数中。",
    "aspect": "输出比例",
    "aspect_help": (
        "竖屏固定输出 1080 × 1920，横屏固定输出 1920 × 1080；切换比例也会加载对应的"
        "布局默认值和远程搜索方向。"
    ),
    "portrait": "竖屏 9:16",
    "landscape": "横屏 16:9",
    "material_source": "素材来源",
    "material_source_help": (
        "本地上传使用你选择的图片或视频；远程搜索会按每个英文单词，从基础设置中的服务商"
        "下载一张静态图片。"
    ),
    "local_uploads": "本地上传",
    "remote_search": "远程搜索",
    "word_material_review": "逐词选图 (可选)",
    "word_material_review_help": ("先从候选图片中选择，不满意再上传自己的素材。"),
    "local_material_override": "候选都不合适？使用本地素材",  # noqa: RUF001
    "material_current_word": "当前单词",
    "material_previous_word": "上一词",
    "material_next_word": "下一词",
    "material_candidate_count": "{count} 张候选",
    "material_search": "搜索候选",
    "material_search_query": "搜图关键词",
    "material_search_query_help": (
        "默认使用当前单词，也可改成更具体的词组，例如 river bank；留空则使用原单词。"
        "只影响图片搜索，不改变单词内容或朗读。重新搜索后，只有选用其他图片才会替换当前素材。"
    ),
    "material_search_again": "重新搜索",
    "material_search_help": "仅更新当前单词的候选列表，不改变已选素材；搜索结果可能与之前相同。",
    "material_searching": "正在搜索候选图片…",
    "material_search_prompt": "点击“搜索候选”，查看这个单词的图片。",
    "material_candidate": "候选 {number}",
    "material_use": "使用此图",
    "material_selected": "✓ 已选用",
    "material_thumbnail_failed": "缩略图加载失败，仍可尝试选用此图。",
    "material_upload_prompt": "上传图片或视频",
    "material_upload_help": "上传后仅替换 {word} 的素材，其他单词不受影响。",
    "material_remove_upload": "移除本地素材",
    "material_using_candidate": "当前使用：候选 {number} · {mode}",
    "material_manual": "手动选择",
    "material_auto": "自动选择",
    "material_untouched_help": "未手动选择的单词，将按当前设置自动选图。",
    "remote_material_automatic": "尚未固定素材，生成时会自动搜索该单词。",
    "remote_material_pinned": "已固定当前素材，生成时将直接使用。",
    "remote_material_missing": "没有找到合适的远程图片，请重新搜索或上传本地素材。",
    "local_material_selected": "已选择本地素材，将覆盖该单词的远程结果。",
    "selection_mode": "素材分配方式",
    "selection_mode_help": (
        "顺序会按词条轮换素材；稳定随机会结合已保存的任务种子进行分配，因此重新生成仍保持选择。"
    ),
    "sequential": "顺序",
    "random": "随机",
    "question_settings": "问题设置",
    "question_settings_help": "关闭后，问题文本、问题朗读和进度条均不生效。",
    "show_question": "启用问题片段",
    "question_text": "问题文本",
    "advanced_question_settings": "问题高级参数 (:blue[点击展开])",
    "progress_settings": "进度条设置",
    "progress_settings_help": (
        "仅在“启用问题片段”后生效；进度条只在每个问题片段中显示，播放时长等于倒计时与"
        "问题朗读的总时长。"
    ),
    "show_progress": "启用进度条",
    "advanced_progress_settings": "进度条高级参数 (:blue[点击展开])",
    "start_color": "起始颜色",
    "end_color": "结束颜色",
    "material_settings": "图片素材设置",
    "material_settings_help": (
        "关闭后，素材来源、上传或搜索、分配方式、尺寸、形状与位置均不生效。本地来源支持"
        "图片和视频；Pexels、Pixabay 远程来源当前只获取静态图片。"
    ),
    "show_material": "启用图片素材",
    "advanced_material_settings": "图片素材高级参数 (:blue[点击展开])",
    "material_uploads": "素材图片",
    "material_uploads_mixed": "素材图片和视频",
    "circle": "圆形",
    "rectangle": "矩形",
    "shape": "形状",
    "chinese_narration": "中文朗读",
    "fast_english_narration": "快速英语朗读",
    "slow_english_narration": "慢速英语朗读",
    "narration_settings_help": (
        ":red[音色必须与文本语言一致]。关闭音轨或重复次数为 0 时不会生成该音轨；下拉框"
        "包含该语言全部内置 Edge TTS 音色，音量和语速范围为 -100% 到 +100%，播放声音只用于试听。"
    ),
    "advanced_narration_settings": "朗读高级参数 (:blue[点击展开])",
    "enabled": "启用",
    "repeats": "重复次数",
    "voice": "音色",
    "voice_gender_female": "女声",
    "voice_gender_male": "男声",
    "volume": "音量",
    "rate": "语速",
    "play_voice": "播放声音",
    "play_voice_help": "合成一段简短示例，用于测试当前 Edge TTS 连接和所选音色。",
    "english_text": "英文文本",
    "phonetic_text": "音标文本",
    "chinese_text": "中文文本",
    "text_settings_help": (
        "关闭后隐藏该文字层；字体路径必须指向受支持的本地字体。字号、颜色、粗细、描边和位置"
        "会同时作用于预览和最终视频，位置填写 -1 表示自动。"
    ),
    "advanced_text_settings": "文字高级参数 (:blue[点击展开])",
    "font_path": "字体路径",
    "font_size": "字号",
    "fill_color": "文字颜色",
    "weight": "文字粗细",
    "stroke_color": "描边颜色",
    "stroke_width": "描边宽度",
    "top": "上",
    "bottom": "下",
    "left": "左",
    "right": "右",
    "width": "宽度",
    "height": "高度",
    "question_label": "问题",
    "english_label": "英文",
    "phonetic_label": "音标",
    "chinese_label": "中文",
    "progress_label": "进度条",
    "material_label": "图片素材",
    "fast_english_label": "英文快读",
    "slow_english_label": "英文慢读",
    "font_suffix": "字体",
    "font_size_suffix": "字体大小",
    "font_color_suffix": "字体颜色",
    "font_weight_suffix": "字体粗细",
    "stroke_color_suffix": "描边颜色",
    "stroke_weight_suffix": "描边粗细",
    "top_margin_suffix": "上边距",
    "bottom_margin_suffix": "下边距",
    "left_margin_suffix": "左边距",
    "right_margin_suffix": "右边距",
    "width_suffix": "宽度",
    "height_suffix": "高度",
    "shape_suffix": "形状",
    "start_color_suffix": "开始颜色",
    "end_color_suffix": "结束颜色",
    "narration_repeats_suffix": "朗读次数",
    "narration_voice_suffix": "朗读声音",
    "narration_volume_suffix": "朗读音量 (100表示+100%)",
    "narration_rate_suffix": "朗读语速 (100表示+100%)",
    "generate_video": "生成视频",
    "preview": "预览",
    "preview_word": "预览单词",
    "preview_card_type": "卡片类型",
    "preview_answer": "答案卡",
    "preview_question": "问题卡",
    "preview_sample_word": "示例单词 (尚未填写单词信息)",
    "preview_question_help": "启用问题片段后，才可选择问题卡。",
    "preview_help": (
        "选择单词和卡片类型后点击“预览”。静态预览不含朗读和动态进度条，不改变整段视频的内容。"
    ),
    "preview_placeholder": (
        "该单词尚未选定素材，当前使用占位图预览。选图或上传本地素材后，可查看实际排版。"
    ),
    "result": "生成结果",
    "download": "下载 MP4",
    "task_id": "任务 ID",
    "result_task_help": (
        "请复制并保存任务 ID，同时保留 storage/ 中对应的任务文件夹。只有 ID 无法恢复任务。"
    ),
    "result_version": "视频文件",
    "material_overview_help": "点击单词可跳转选图。「待自动搜索」是正常状态，无需逐个手动选图。",
    "material_status_upload": "本地上传",
    "material_status_manual": "手动选图",
    "material_status_auto": "已自动选图",
    "material_status_saved": "已保存素材",
    "material_status_empty": "无结果·可上传",
    "material_status_pending": "待自动搜索",
    "recent_tasks": "最近任务",
    "refresh_tasks": "刷新列表",
    "recent_tasks_placeholder": "选择后自动加载任务",
    "recent_tasks_empty": "暂无可用任务",
    "recent_tasks_help": (
        "按最近更新时间显示最多 20 个任务，时间以运行本程序的电脑为准。"
        "在其他页面生成任务后，可刷新列表；也可以在下方手动输入任务 ID。"
        "单词数表示已保存的单词数量。"
    ),
    "recent_tasks_unavailable": "暂时无法读取任务列表，仍可尝试通过任务 ID 加载。",
    "recent_task_option": "{title} · {count} 个单词 · {status} · {time} · {task_id}",
    "untitled_task": "未命名任务",
    "task_status_queued": "待生成",
    "task_status_running": "生成中",
    "task_status_complete": "已完成",
    "task_status_failed": "失败",
    "video_history": "历史视频版本",
    "video_history_version": "选择预览版本",
    "video_history_current": "当前成品",
    "video_history_empty": "该任务暂时没有已保存的视频。",
    "video_history_help": (
        "这里只预览和下载，不会回退任务，也不会改动当前单词、参数或最新生成结果。"
        "时间为文件修改时间，以运行本程序的电脑为准。"
    ),
    "video_history_unavailable": (
        "暂时无法预览此视频，文件可能已丢失或任务发生了变化，请尝试重新加载任务。"
    ),
    "download_history": "下载此版本",
    "load_task": "加载任务",
    "task_load_required": "请先输入任务 ID 并加载任务，确认单词后再重新生成。",
    "task_load_failed": (
        "无法加载任务，请检查 ID 是否正确，以及 storage/ 中的任务文件夹是否完整。"
        "当前编辑的单词和已保存的文件均未改变。"
    ),
    "task_no_vocabulary": "该任务尚未保存单词信息，暂时无法重新生成。",
    "task_loaded": "已加载任务 {task_id} · 共 {count} 个单词",
    "saved_chinese": "中文释义",
    "saved_phonetic": "音标",
    "saved_task_help": (
        "请确认上方词表。重新生成使用该任务保存的单词和参数，不会读取或覆盖主编辑区的内容。"
        "替换素材可选，每个文件需选择不同的单词；原视频会保留。"
    ),
    "task_materials_disabled": "该任务未启用图片素材，可以重新生成，但替换素材不会生效。",
    "replacement_word": "替换哪个单词的素材",
    "replacement_word_option": "第 {number} 个：{word}",
    "replacement_images": "替换素材",
    "replacement_index": "词条序号",
    "regenerate": "替换素材并重新生成",
    "task_directory": "任务目录",
    "open_folder": "打开目录",
    "preparing": "正在准备生成任务",
    "progress_vocabulary": "单词信息已准备完成",
    "progress_materials": "单词素材已准备完成",
    "progress_narration": "朗读音频已准备完成",
    "progress_cards": "单词画面已准备完成",
    "progress_composing": "正在合成视频",
    "progress_complete": "视频已生成完成",
    "missing_topic": "请先输入视频主题，再使用 AI 生成单词信息。",
    "missing_content": "请输入视频主题，或直接填写有效的单词信息。",
    "missing_background": "生成前请上传背景图片。",
    "missing_llm_key": "使用 AI 生成单词信息或自动补全音标前，请先填写 API 密钥。",
    "missing_materials": "使用本地素材时，请至少上传一张图片或一个视频。",
    "missing_provider_key": "请输入所选图片素材服务商的 API 密钥。",
    "script_invalid": "“单词信息”格式有误，请按当前音标模式检查内容。",
    "generation_complete": "视频生成成功。",
    "regeneration_complete": "视频重新生成成功。",
    "llm_connection_success": "大模型连接成功，当前密钥、接口地址和模型均可用。",
    "image_connection_success": "图片服务连接成功，当前密钥可以正常搜索图片。",
    "error_speech_generic": "语音生成失败，请检查网络和朗读设置后重试。",
    "error_speech_summary": "语音生成失败。{reason}已尝试 {attempts} 次。",
    "error_speech_word": "第 {number} 个单词的{track}失败。{reason}已尝试 {attempts} 次。",
    "error_speech_track": "{track}失败。{reason}已尝试 {attempts} 次。",
    "error_speech_track_question": "问题片段朗读",
    "error_speech_track_zh": "中文朗读",
    "error_speech_track_fast": "英语快读",
    "error_speech_track_slow": "英语慢读",
    "error_speech_connection": "无法连接 Edge TTS，请检查网络后重试。",
    "error_speech_timeout": "Edge TTS 请求超时，请检查网络后重试。",
    "error_speech_service": "Edge TTS 暂时不可用或请求过于频繁，请稍后重试。",
    "error_speech_rejected": "Edge TTS 拒绝了请求，请检查音色和服务是否可用。",
    "error_speech_empty": "Edge TTS 未返回音频，请检查音色和朗读文字后重试。",
    "error_speech_settings": "请检查音色、语速、音量和朗读文字是否有效。",
    "error_speech_file": "无法保存语音文件，请检查磁盘剩余空间和存储目录权限。",
    "error_speech_output": "生成的音频不存在、为空或超过大小限制。",
    "error_speech_certificate": "无法与 Edge TTS 建立安全连接，请检查网络和证书配置。",
    "error_speech_unknown": "请检查网络和朗读设置后重试。",
    "error_llm_auth": "大模型服务商拒绝了 API 密钥，请检查当前服务商的密钥配置。",
    "error_llm_timeout": "大模型请求超时，请稍后重试并检查网络或服务状态。",
    "error_llm_unavailable": (
        "无法连接大模型服务；使用 Ollama 时请确认服务已启动，否则请检查网络。"
    ),
    "error_llm_not_found": (
        "找不到配置的大模型或接口；请检查模型名称和服务地址。使用 Ollama 时，"
        "请先运行 ollama pull <模型名>。"
    ),
    "error_image_auth": "图片服务拒绝了 API 密钥，请检查当前服务商的密钥配置。",
    "error_image_timeout": "图片服务连接超时，请稍后重试并检查网络。",
    "error_image_unavailable": "无法连接图片服务，请检查网络后重试。",
    "error_image_invalid": "图片服务返回了无效响应，请稍后重试。",
    "error_upload_too_large": "上传文件超过大小限制，请选择更小的文件。",
    "error_upload_size_details": (
        "文件大小为 {actual:.2f} MiB，超过此类型允许的 {limit:g} MiB 上限。"
    ),
    "image_upload_limits": "单张图片不超过 32 MiB，解码后不超过 5000 万像素。",
    "mixed_upload_limits": (
        "单个文件：图片不超过 32 MiB，视频不超过 128 MiB。"
        "上传框显示的 128 MB 是视频上限，不是图片上限。"
    ),
    "error_material_image_decode": ("无法读取所选素材图片，请检查文件是否损坏或格式是否受支持。"),
    "error_material_video_decode": (
        "无法读取所选素材视频，请检查文件是否损坏、过长或格式是否受支持。"
    ),
    "error_material_type": "不支持所选素材文件类型，请上传受支持的图片或视频。",
    "error_generation_interrupted": "视频生成已中断，未完成的视频文件已清理。",
    "duplicate_replacements": "多个替换文件不能选择同一个单词，请为每个文件选择不同的单词。",
    "regeneration_settings": "重新生成已有任务",
    "regeneration_settings_help": (
        "先加载已有任务并确认词表。替换素材可选，可以只更换某个单词的素材。"
        "请保留 storage/ 中对应的任务文件夹。"
    ),
    "render_settings": "渲染设置",
    "fps": "帧率 (FPS)",
    "fps_help": "最终 H.264 视频支持 12–60 FPS；默认 24 FPS，兼顾流畅度与生成速度。",
    "fit_mode": "素材填充方式",
    "fit_mode_help": (
        "完整显示会保留全部素材并可能留空；覆盖填满可能裁切素材边缘；拉伸填满可能使素材变形。"
    ),
    "candidate_pool_size": "远程候选数量",
    "candidate_pool_size_help": (
        "远程搜索会为每个单词请求这么多候选，再按顺序或稳定随机方式分配；本地上传不受影响。"
    ),
    "contain": "完整显示",
    "cover": "覆盖裁切",
    "stretch": "拉伸填充",
    "video_upload_help": "支持图片和视频 (PNG、JPG、WebP、MP4、MOV、M4V、WebM)。",
    "background_music_settings": "背景音乐",
    "background_music_settings_help": (
        "背景音乐可选且只能本地上传，会循环覆盖整段视频；朗读压低只在倒计时或语音播放期间降低音量。"
    ),
    "background_music_enabled": "启用背景音乐",
    "background_music_file": "本地音乐文件",
    "advanced_music_settings": "背景音乐高级参数 (:blue[点击展开])",
    "music_volume": "音乐音量 (%)",
    "music_ducking": "朗读时压低比例 (%)",
    "missing_music": "启用背景音乐时，请上传受支持的本地音乐文件。",
    "question_narration": "启用问题朗读",
    "question_narration_settings": "问题朗读设置",
    "question_narration_settings_help": (
        "仅在“启用问题片段”后生效；关闭问题朗读或将重复次数设为 0 时，只会移除问题语音，"
        "仍会保留倒计时提示音。"
    ),
}

_CATALOGS = {Locale.ZH_CN: _ZH_CN, Locale.EN_US: _EN_US}


def catalog_keys(locale: Locale) -> frozenset[str]:
    return frozenset(_CATALOGS[locale])


def translate(locale: Locale, key: str) -> str:
    try:
        return _CATALOGS[locale][key]
    except KeyError as exc:
        raise KeyError(f"Missing translation key: {key}") from exc
