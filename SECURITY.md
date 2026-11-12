# Security policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private reporting instead:
<https://github.com/cb-Sigma/triton-cachekit/security/advisories/new>.

Include what you did, what happened, and the version of triton-cachekit and Triton. You should get
an acknowledgement within a week.

## What to keep in mind when using this tool

- **A kernel cache is executable code.** Triton loads the binaries it finds in its cache directory
  onto the GPU without checking where they came from. `unpack` verifies the SHA-256 of every file
  against the archive's own manifest, which detects corruption but **not** a malicious archive: the
  manifest travels with the data and is not signed. Only unpack archives you built or trust, and
  sign them with your usual tooling (for example cosign on the OCI layout) if they cross a trust
  boundary.
- **`prewarm` imports and runs your kernel modules.** A manifest or recording names Python files
  or modules; they are imported in a subprocess, which executes their top-level code. Treat
  manifests and recordings from others like scripts.
- **Recordings can contain sensitive data**: absolute paths, host name, command line, `TRITON_*`
  environment variables and (unless `--no-source` is used) kernel source text. Review before
  sharing them in an issue.
- `unpack` refuses archive members outside `cache/<key>/<file>` (no absolute paths, no `..`).
- The tool makes no network connections.
