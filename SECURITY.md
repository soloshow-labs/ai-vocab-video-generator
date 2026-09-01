# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Security** tab and
the [private vulnerability reporting form](https://github.com/soloshow-labs/ai-vocab-video-generator/security/advisories/new).
Do not open a public issue containing an exploit, credential, private endpoint,
or user-generated media.

Include the affected revision, reproduction steps, expected impact, and any
suggested mitigation. Maintainers will acknowledge complete reports as soon as
reasonably possible and coordinate disclosure once a fix is available.

## Secrets and local data

- Keep provider keys in session-only password fields, environment variables,
  or the ignored `.env` file.
- Never put a credential in `config.toml`, a screenshot, a job manifest, or a
  bug report.
- Treat `storage/`, uploaded media, generated narration, and transcripts as
  private local data.
- Review downloaded models and user-provided media under their own licenses.

If a key is accidentally exposed, revoke or rotate it with the provider first.
Then remove it from the working tree and Git history before publishing.
Rewriting Git history does not invalidate a credential by itself.

## Supported versions

Security fixes are applied to the latest revision on the default branch while
the project is in its initial development stage.
