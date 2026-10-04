# Artifact storage

- Do not store large BFS states, downloaded GPU evidence, or synthetic SSD stress payloads on the user's C: drive. Keep them on the GPU host and Hugging Face.
- Keep only source, small logs, manifests, configuration, checksum receipts, and concise validation reports locally. Upload existing large files directly without making another archive or cache copy on C:.
- Before removing unique local evidence, preserve it on HF with its existing privacy, pin the revision, and verify remote size, SHA-256, and authenticated access. Reproducible synthetic payloads may be removed after preserving reports and reproduction source.
- Delete owned temporary payloads promptly after verification. Coordinate ownership with other active project chats; preserve active launch files and never clean another chat's worktree or live files.
- Avoid large local HF downloads and caches. Read state payloads and perform SSD stress on the GPU host; local tests must use small fixtures.
