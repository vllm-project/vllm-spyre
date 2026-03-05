# Release process

This repository contains two packages with independent versioning:

- **vllm-spyre** (tags: `vX.Y.Z`) — the main Spyre hardware plugin
- **vllm-spyre-next** (tags: `spyre-next-vX.Y.Z`) — next iteration on the torch-spyre stack

Both packages follow [Semantic Versioning](https://semver.org/): use **patch** for bug fixes, **minor** for new features, **major** for breaking changes.

## Release steps

1. Go to the [Releases page](https://github.com/vllm-project/vllm-spyre/releases) and click "Draft a new release"
2. Click "Choose a tag" and type the new version:
   - For vllm-spyre: `vX.Y.Z` (e.g., `v1.2.0`)
   - For vllm-spyre-next: `spyre-next-vX.Y.Z` (e.g., `spyre-next-v0.2.0`)
3. Click "Generate release notes" and select the previous tag **of the same package** as the base:
   - For vllm-spyre: select a `v*` tag (not `spyre-next-v*`)
   - For vllm-spyre-next: select a `spyre-next-v*` tag (not `v*`)
4. Review the changes to decide on the version bump, edit release notes, then publish
5. The appropriate workflow will automatically trigger and publish to PyPI

**Alternative**: Create the tag locally first, then create the release on GitHub.

## Test releases

**vllm-spyre**: Automatically published to [test.pypi.org](https://test.pypi.org/project/vllm-spyre/) on every push to main.

**vllm-spyre-next**: Create a pre-release tag (e.g., `spyre-next-v0.1.0-rc.1`) to trigger automatic publication to [test.pypi.org](https://test.pypi.org/p/vllm-spyre-next). Install with:

```bash
pip install -i https://test.pypi.org/simple/ vllm-spyre-next==0.1.0rc1
```

**Note**: setuptools_scm automatically converts SemVer pre-release tags to [PEP 440](https://peps.python.org/pep-0440/) format (e.g., `v1.0.0-rc.1` → `1.0.0rc1`).
