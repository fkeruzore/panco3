"""Beam + transfer-function filtering of the model map (JAX).

Ports ``panco2/panco2/filtering.py``. Two operations, applied in this order
(matching panco2's ``Filter.__call__``):

1. **Transfer function**: zero-pad, FFT, multiply by a precomputed 2-D transfer
   function, inverse FFT, crop. Direct ``jnp.fft`` port.
2. **Gaussian beam**: ``scipy.ndimage.gaussian_filter`` is not in JAX, so we
   reimplement it exactly -- separable, truncated-at-4-sigma kernel with the
   ``reflect`` (half-sample-symmetric, == ``numpy`` ``symmetric``) boundary
   that ``scipy.ndimage`` uses by default. This is differentiable and matches
   scipy to machine precision (see ``tests/test_filtering.py``).

The transfer-function *setup* (interpolating a 1-D/2-D TF onto the map's
Fourier grid) is done once in NumPy; only the per-call map operations are
in JAX.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array


# --------------------------------------------------------------------------- #
# scipy.ndimage.gaussian_filter, reimplemented in JAX
# --------------------------------------------------------------------------- #
def _gaussian_kernel1d(sigma: float, truncate: float = 4.0):
    """1-D Gaussian kernel identical to
    ``scipy.ndimage._gaussian_kernel1d``."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / sigma**2 * x**2)
    phi /= phi.sum()
    return jnp.asarray(phi, dtype=jnp.float64), radius


def _correlate_axis(arr: Array, phi: Array, radius: int, axis: int) -> Array:
    """Correlate ``arr`` with the symmetric 1-D kernel along ``axis``.

    Uses ``symmetric`` padding (== scipy.ndimage 'reflect'). The kernel is
    symmetric so correlation and convolution coincide.
    """
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    arr = jnp.pad(arr, pad, mode="symmetric")
    arr_m = jnp.moveaxis(arr, axis, -1)
    shp = arr_m.shape
    rows = arr_m.reshape(-1, shp[-1])
    conv = jax.vmap(lambda r: jnp.convolve(r, phi, mode="valid"))(rows)
    conv = conv.reshape(shp[:-1] + (conv.shape[-1],))
    return jnp.moveaxis(conv, -1, axis)


def gaussian_filter(
    image: Array, sigma: float, truncate: float = 4.0
) -> Array:
    """JAX equivalent of ``scipy.ndimage.gaussian_filter`` (mode='reflect')."""
    phi, radius = _gaussian_kernel1d(sigma, truncate)
    out = image
    for axis in range(image.ndim):
        out = _correlate_axis(out, phi, radius, axis)
    return out


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #
class Filter:
    """Beam + transfer-function filter, matching
    ``panco2.filtering.Filter``."""

    def __init__(self, beam_sigma_pix: float = 0.0, tf=None, pad_pix: int = 0):
        self.has_beam = beam_sigma_pix != 0.0
        self.beam_sigma_pix = beam_sigma_pix
        self.has_tf = tf is not None
        self.transfer_function = None if tf is None else jnp.asarray(tf)
        self.pad_pix = int(pad_pix)

    def __call__(self, in_map: Array) -> Array:
        if self.has_tf:
            pad = self.pad_pix
            in_map = jnp.pad(in_map, pad, mode="constant", constant_values=0.0)
            in_map_fourier = jnp.fft.fft2(in_map)
            in_map = jnp.real(
                jnp.fft.ifft2(in_map_fourier * self.transfer_function)
            )
            in_map = in_map[pad:-pad, pad:-pad] if pad > 0 else in_map
        if self.has_beam:
            in_map = gaussian_filter(in_map, self.beam_sigma_pix)
        return in_map


class Filter1d(Filter):
    """Isotropic 1-D transfer function interpolated onto the 2-D Fourier grid.

    Mirrors ``panco2.filtering.Filter1d``. The TF grid is built once in NumPy.
    """

    def __init__(
        self, npix, pix_size, k_1d, tf_1d, beam_sigma_pix=0.0, pad_pix=0
    ):
        pad = int(pad_pix)
        k_1d = np.asarray(k_1d)
        tf_1d = np.asarray(tf_1d)

        k_map_1d = np.fft.fftfreq(npix + 2 * pad, pix_size)
        k_map_2d = np.hypot(*np.meshgrid(k_map_1d, k_map_1d))

        i_sort = np.argsort(k_1d)
        tf_smallk, tf_highk = tf_1d[i_sort][0], tf_1d[i_sort][-1]
        tf_map_2d = np.interp(
            k_map_2d,
            k_1d[i_sort],
            tf_1d[i_sort],
            left=tf_smallk,
            right=tf_highk,
        )

        super().__init__(
            beam_sigma_pix=beam_sigma_pix, tf=tf_map_2d, pad_pix=pad
        )


class Filter2d(Filter):
    """Anisotropic 2-D transfer function on a ``(kx, ky)`` grid.

    Mirrors ``panco2.filtering.Filter2d``, but uses
    ``scipy.interpolate.RegularGridInterpolator`` (panco2 relied on the
    now-removed ``scipy.interpolate.interp2d``), so it is *not* guaranteed
    to be bit-for-bit identical to panco2's 2-D path.
    """

    def __init__(
        self, npix, pix_size, kx, ky, tf, beam_sigma_pix=0.0, pad_pix=0
    ):
        from scipy.interpolate import RegularGridInterpolator

        pad = int(pad_pix)
        k_map = np.fft.fftfreq(npix + 2 * pad, pix_size)
        kxx, kyy = np.meshgrid(k_map, k_map, indexing="ij")
        pts = np.stack([kxx.ravel(), kyy.ravel()], axis=-1)

        interp = RegularGridInterpolator(
            (np.asarray(kx), np.asarray(ky)),
            np.asarray(tf),
            bounds_error=False,
            fill_value=0.0,
        )
        tf_map_2d = interp(pts).reshape(kxx.shape)

        super().__init__(
            beam_sigma_pix=beam_sigma_pix, tf=tf_map_2d, pad_pix=pad
        )
