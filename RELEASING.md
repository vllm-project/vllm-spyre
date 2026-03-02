# Release process

This repository contains two packages with independent versioning:

- **vllm-spyre** (tags: `vX.Y.Z`) — the main Spyre hardware plugin
- **vllm-spyre-next** (tags: `spyre-next-vX.Y.Z`) — next iteration on the torch-spyre stack

## Releasing vllm-spyre

When ready to make a new vllm-spyre release:

1. Create and push a tag on main for the next version, following the convention `vX.Y.Z`.
2. Create a new release on GitHub for this tag. Use the "Generate Release Notes" button to draft release notes based on the changes since the last release. **Make sure to select the previous `v*` tag** (not a `spyre-next-v*` tag) as the base for release notes.
3. Once the release is ready, publish it by clicking the "Publish release" button.
4. The `build-and-publish.yaml` workflow will trigger when the release is published, and push a new wheel to pypi.

## Releasing vllm-spyre-next

When ready to make a new vllm-spyre-next release:

1. Create and push a tag on main for the next version, following the convention `spyre-next-vX.Y.Z`.
2. Create a new release on GitHub for this tag. Use the "Generate Release Notes" button to draft release notes based on the changes since the last release. **Make sure to select the previous `spyre-next-v*` tag** (not a `v*` tag) as the base for release notes.
3. Once the release is ready, publish it by clicking the "Publish release" button.
4. The `build-and-publish-next.yaml` workflow will trigger when the release is published, and push a new wheel to pypi.

See [`vllm_spyre_next/RELEASING.md`](vllm_spyre_next/RELEASING.md) for more details on the vllm-spyre-next release process, including how to do a test release.
