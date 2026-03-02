# Releasing vllm-spyre-next

## Tag format

All vllm-spyre-next releases use tags with the prefix `spyre-next-v`:

```
spyre-next-v0.1.0
spyre-next-v0.2.0
spyre-next-v1.0.0
```

This prefix distinguishes vllm-spyre-next tags from vllm-spyre tags (`vX.Y.Z`) in the
shared git history, allowing each package to version independently in the monorepo.

## Release process

1. Create and push a tag on main for the next version:

   ```bash
   git checkout main && git pull origin main
   git tag spyre-next-v0.1.0
   git push origin spyre-next-v0.1.0
   ```

2. Create a new release on GitHub for this tag. Use the "Generate Release Notes" button
   to draft release notes. **In the "Previous tag" dropdown, select the previous
   `spyre-next-v*` tag** — not a `v*` tag — so the notes only cover changes relevant
   to this package.

3. Review and curate the release notes, then publish the release.

4. The `build-and-publish-next.yaml` workflow will trigger and push a new wheel to PyPI.

## Test releases

Before cutting a production release, you can publish a release candidate to
[test.pypi.org](https://test.pypi.org/p/vllm-spyre-next) to validate the build and install process:

1. Push a pre-release tag:

   ```bash
   git tag spyre-next-v0.1.0-rc1
   git push origin spyre-next-v0.1.0-rc1
   ```

2. The `publish-to-test-pypi-next.yaml` workflow will trigger automatically and publish to test.pypi.org.

3. Install and test from test.pypi.org:

   ```bash
   pip install -i https://test.pypi.org/simple/ vllm-spyre-next==0.1.0rc1
   ```
