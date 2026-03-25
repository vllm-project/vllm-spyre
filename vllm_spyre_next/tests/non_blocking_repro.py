"""Reproduce non_blocking behavior on Spyre device.

Tests whether torch-spyre supports non_blocking=True for:
1. tensor.copy_(src, non_blocking=True) — H2D (CPU→Spyre)
2. tensor.to("cpu", non_blocking=True) — D2H (Spyre→CPU)
3. tensor.copy_(src, non_blocking=True) with dtype mismatch (int32→int64)

Results:
- If all pass silently, non_blocking=True is ignored (safe to leave in parent code)
- If any fail, we need to override specific methods to use non_blocking=False

Use this script to file a torch-spyre issue if non_blocking is not supported.
"""

import sys

import torch
import torch_spyre  # noqa: F401

device = torch.device("spyre")


def test(name, fn):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    sys.stdout.flush()
    try:
        fn()
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    sys.stdout.flush()


# --- Test 1: H2D copy_ with non_blocking=True ---
def test_h2d_copy_non_blocking():
    src = torch.arange(64, dtype=torch.float16, device="cpu")
    dst = torch.zeros(64, dtype=torch.float16, device=device)
    dst.copy_(src, non_blocking=True)
    back = dst.to("cpu")
    assert torch.equal(src, back), f"Data mismatch: {src[:5]} vs {back[:5]}"

test("H2D copy_ (float16, non_blocking=True)", test_h2d_copy_non_blocking)


# --- Test 2: D2H .to("cpu", non_blocking=True) ---
def test_d2h_to_non_blocking():
    src = torch.arange(64, dtype=torch.float16, device="cpu")
    t = torch.zeros(64, dtype=torch.float16, device=device)
    t.copy_(src)
    back = t.to("cpu", non_blocking=True)
    assert torch.equal(src, back), f"Data mismatch: {src[:5]} vs {back[:5]}"

test("D2H .to('cpu', non_blocking=True)", test_d2h_to_non_blocking)


# --- Test 3: H2D copy_ with dtype mismatch (int32→int64) ---
def test_h2d_copy_dtype_mismatch():
    src = torch.arange(64, dtype=torch.int32, device="cpu")
    dst = torch.zeros(64, dtype=torch.int64, device=device)
    dst.copy_(src, non_blocking=True)
    back = dst.to("cpu")
    expected = src.to(torch.int64)
    assert torch.equal(expected, back), f"Data mismatch: {expected[:5]} vs {back[:5]}"

test("H2D copy_ (int32→int64, non_blocking=True)", test_h2d_copy_dtype_mismatch)


# --- Test 4: H2D copy_ with int64 non_blocking ---
def test_h2d_int64_non_blocking():
    src = torch.arange(64, dtype=torch.int64, device="cpu")
    dst = torch.zeros(64, dtype=torch.int64, device=device)
    dst.copy_(src, non_blocking=True)
    back = dst.to("cpu")
    assert torch.equal(src, back), f"Data mismatch: {src[:5]} vs {back[:5]}"

test("H2D copy_ (int64, non_blocking=True)", test_h2d_int64_non_blocking)


print("\n" + "=" * 60)
print("All tests complete!")
