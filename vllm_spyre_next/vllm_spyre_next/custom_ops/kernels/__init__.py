# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre IR provider registrations.

Importing this package triggers @register_impl decorators,
registering Spyre-specific implementations for vLLM IR ops.
"""
from . import rms_norm as _rms_norm  # noqa: F401
