# Release process

Currently, we only have a single release process for pushing releases off `main`. In the future we may need to maintain multiple release streams to handle compatibility with multiple released versions of vllm.

When ready to make a new release:

1. Create and push a tag on main for the next version, following the convention `vX.Y.Z`.
2. Create a new release on GitHub for this tag. Use the "Generate Release Notes" button to draft releast notes based on the changes since the last release.
3. Once the release is ready, publish it by clicking the "Publish release" button.
4. The `build-and-publish.yaml` workflow will trigger when the release is published, and push a new wheel to pypi

We could automate the process of creating the release on GitHub as well, however there is a slight snag that github actions cannot trigger from events that were performed by github actions. See: https://github.com/semantic-release/semantic-release/discussions/1906
