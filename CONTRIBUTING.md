# Contributing

Thank you for helping improve AI Vocab Video Generator.

## Development setup

Install Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg. Then
run the following commands from the repository root:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run mypy src
uv run pip-audit --local --skip-editable
```

Install the optional local transcription dependencies only when working on
speech input or ASR:

```bash
uv sync --extra asr
```

## Change guidelines

- Add or update a failing test before changing behavior.
- Keep Streamlit in the presentation layer. Provider and pipeline logic must
  remain usable without Streamlit.
- Use typed application errors with a safe user-facing message. Never include
  API keys, authorization headers, signed URLs, or raw third-party responses.
- Do not commit generated media, model weights, fonts, music, `.env`, local
  configuration, job storage, or credentials.
- Do not add a new network provider without tests that use a mock transport,
  explicit timeouts, TLS URLs, and redacted failures.
- Document the license of any new dependency or asset source.

Keep pull requests focused. In the description, explain the user-visible
behavior, how the change was verified, and any licensing implications.

## Releasing

Releases are created manually through the `Release` GitHub Actions workflow:

1. Update `project.version` in `pyproject.toml` and finish the corresponding
   changelog entry.
2. Merge the release changes into the default branch and confirm that CI is
   passing.
3. Open **Actions → Release → Run workflow**, select the default branch, and
   enter the matching `vX.Y.Z` tag.
4. Keep **Create a draft release** enabled. The workflow reruns the complete CI
   suite before it creates the tag and draft GitHub Release.
5. Review the generated release notes, then publish the draft from the GitHub
   Releases page.

The workflow intentionally publishes a source release only. GitHub provides the
source ZIP and TAR.GZ automatically; it does not build desktop installers or
publish to PyPI. Existing tags and releases are rejected to prevent accidental
replacement.
