"""Two small Triton kernels used by the example prewarm manifest.

They are deliberately simple; what matters for the cache is how they are
specialised (dtypes, constexpr values, alignment), not what they compute.
"""

import triton
import triton.language as tl

RMS_EPS = tl.constexpr(1e-5)


@triton.jit
def mean_square(x, n_cols):
    return tl.sum(x * x, axis=0) / n_cols


@triton.jit
def rms_norm_rows(x_ptr, weight_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    """out[row, :] = x[row, :] / sqrt(mean(x[row, :] ** 2) + eps) * weight"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = 1.0 / tl.sqrt(mean_square(x, n_cols) + RMS_EPS)
    tl.store(out_ptr + row * n_cols + cols, (x * scale * w).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = x * sigmoid(x)"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = x / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)
