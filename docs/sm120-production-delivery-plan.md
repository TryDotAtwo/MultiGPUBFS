# Shared SM120 production and owned-process cleanup

The package default is now the shared CUDA-runtime library compiled for SM75,
SM86 and SM120, using the same CUDA source tree as the previous diagnostic
contrasts. The Rust executable and all BFS algorithms remain unchanged.

- Wait for the owned process group after its leader exits, with a bounded
  TERM/KILL cleanup deadline. Save cleanup.json and a post-cleanup snapshot
  on failure; stop further GPU probes only after that deadline.
- Perform cleanup once per probe and close the stdout reader.
- Complete production API/kernel, resident and exact S3 BFS tests first.
  Run production modes, multi-GPU probes and the finite CayleyPy catalog
  before risky legacy-build contrasts. Preserve all contrast failures.
- Preserve original binary and CUDA source provenance, per-library hashes,
  actual loader resolution, native cubin inventories and private runtime linkage.
- Check the installed wheel on real 2xT4 GPUs, including ordinary doctor and
  single/multi-GPU generic, matrix and specialized BFS. This validates the
  new library on its SM75 images; original-server SM120 execution requires
  a fresh report from that server. Never label T4 acceptance as SM120 acceptance.
- Publish the pinned wheel, checksum manifest, tested command and finite
  acceptance evidence on GitHub. No collaborator diagnostic archive or
  user state payload is published.

Regression RED: Actions 38685911739 reproduces delayed descendants, duplicate
cleanup, the old ordering and the old production preference.
