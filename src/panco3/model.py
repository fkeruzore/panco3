"""Differentiable forward model: binned pressure profile -> tSZ map (JAX).

Ports ``panco2.model.ModelBinned`` (``panco2/panco2/model.py``). The
pipeline is

    par_vec --> pressure bins P_i
            --> Compton-y profile y(R)      (panco3.integrate, LOS quadrature)
            --> 2-D Compton-y map           (panco3.interp.prof2map)
            --> filtered SZ map             (panco3.filtering.Filter)
            --> * conv + zero               (+ point sources)

Every step is a pure JAX function of ``par_vec`` (and static config held on the
instance as JAX arrays / Python ints), so ``jax.grad``/``jax.jit`` apply.

Parameter-vector layout (same indices dict as panco2):
``P_0 .. P_{n-1}`` (pressure bins), ``conv``, ``zero`` (optional), then point
source fluxes ``F_1 .. F_k``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import Array

from . import interp, integrate
from .cluster import SZ_FACT
from .filtering import Filter


class ModelBinned:
    def __init__(
        self,
        r_bins,
        radii: dict,
        zero_level: bool = True,
        sz_fact: float = SZ_FACT,
        n_nodes: int = 32,
        r_max_integ: float | None = None,
    ):
        self.r_bins = jnp.asarray(r_bins)
        self.n_bins = int(self.r_bins.shape[0])
        self.zero_level = zero_level
        self.sz_fact = float(sz_fact)
        self.n_nodes = int(n_nodes)
        self.r_max_integ = r_max_integ

        # Static radii grids as JAX arrays.
        self.r_x = jnp.asarray(radii["r_x"])
        self.r_xy = jnp.asarray(radii["r_xy"])

        # Parameter index layout.
        self.indices: dict[str, int] = {}
        self.i_press = slice(0, self.n_bins)
        for i in range(self.n_bins):
            self.indices[f"P_{i}"] = i
        self.indices["conv"] = self.n_bins
        self.i_conv = self.n_bins
        if zero_level:
            self.indices["zero"] = self.n_bins + 1
            self.i_zero = self.n_bins + 1
        else:
            self.i_zero = None

        # Point sources (configured via add_point_sources).
        self.n_ps = 0
        self.i_ps = slice(self.n_params, self.n_params)
        self._ps_x = None
        self._ps_y = None
        self.ps_size = None

        self._filter = Filter()

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return len(self.indices)

    @property
    def params(self) -> list[str]:
        return list(self.indices.keys())

    @property
    def filter(self) -> Filter:
        return self._filter

    @filter.setter
    def filter(self, value: Filter):
        self._filter = value

    def par_vec2dic(self, vec) -> dict:
        return {k: vec[i] for k, i in self.indices.items()}

    def par_dic2vec(self, dic) -> Array:
        vec = np.zeros(self.n_params)
        for p, i in self.indices.items():
            vec[i] = dic[p]
        return jnp.asarray(vec)

    # ------------------------------------------------------------------ #
    def add_point_sources(self, coords, wcs, beam_sigma_pix):
        """Register Gaussian point sources at sky ``coords`` (see panco2)."""
        from astropy.wcs.utils import skycoord_to_pixel

        n_pix = wcs.array_shape[0]
        self.n_ps = len(coords)
        old = self.n_params
        self.i_ps = slice(old, old + self.n_ps)

        xmaps, ymaps = [], []
        for i, c in enumerate(coords):
            px, py = skycoord_to_pixel(c, wcs)
            xmap, ymap = np.meshgrid(
                np.arange(n_pix) - px, np.arange(n_pix) - py
            )
            xmaps.append(xmap)
            ymaps.append(ymap)
            self.indices[f"F_{i + 1}"] = old + i

        self._ps_x = jnp.asarray(np.stack(xmaps))  # (n_ps, npix, npix)
        self._ps_y = jnp.asarray(np.stack(ymaps))
        self.ps_size = float(beam_sigma_pix)

    # ------------------------------------------------------------------ #
    def compute_slopes(self, P_i: Array) -> Array:
        """Power-law slopes of the binned profile (Romero et al. 2018)."""
        lr = jnp.log(self.r_bins)
        lp = jnp.log(P_i)
        alphas = -jnp.diff(lp) / jnp.diff(lr)
        return jnp.concatenate([alphas[:1], alphas, alphas[-1:]])

    def pressure_profile(self, r, par_vec) -> Array:
        return interp.interp_powerlaw(self.r_bins, par_vec[self.i_press], r)

    def compton_prof(self, P_i: Array, radarr: Array) -> Array:
        """Compton-y profile by differentiable LOS quadrature."""
        return integrate.compton_y_profile(
            P_i,
            self.r_bins,
            radarr,
            self.sz_fact,
            r_max_integ=self.r_max_integ,
            n_nodes=self.n_nodes,
        )

    def compton_map(self, par_vec) -> Array:
        P_i = par_vec[self.i_press]
        y_prof = self.compton_prof(P_i, self.r_x[1:])
        return interp.prof2map(y_prof, self.r_x[1:], self.r_xy)

    def sz_map(self, par_vec) -> Array:
        y_map = self.compton_map(par_vec)
        sz = self._filter(y_map) * par_vec[self.i_conv]
        if self.i_zero is not None:
            sz = sz + par_vec[self.i_zero]
        return sz

    def ps_map(self, par_vec) -> Array:
        if self.n_ps == 0:
            return jnp.zeros_like(self.r_xy)
        fluxes = par_vec[self.i_ps]  # (n_ps,)
        gauss = jnp.exp(
            -0.5
            * (
                (self._ps_x / self.ps_size) ** 2
                + (self._ps_y / self.ps_size) ** 2
            )
        )  # (n_ps, npix, npix)
        return jnp.sum(fluxes[:, None, None] * gauss, axis=0)

    def model_map(self, par_vec) -> Array:
        """Full model map: filtered SZ + point sources."""
        return self.sz_map(par_vec) + self.ps_map(par_vec)

    # ------------------------------------------------------------------ #
    def init_integ_Y(self, r_max):
        rb = np.asarray(self.r_bins)
        self.r_integ_Y = jnp.asarray(
            np.concatenate([[0.0], rb[rb < r_max], [r_max, -1.0]])
        )

    def integ_Y(self, par_vec) -> Array:
        """Spherically integrated Compton-Y within ``r_integ_Y`` (see
        panco2).

        The ``alpha -> 3`` case of ``(r1^(3-a) - r0^(3-a)) / (3-a)`` is
        removable (-> ``log(r1/r0)``); guarded with a double-``where`` for
        finite gradients.
        """
        P_i = par_vec[self.i_press]
        alphas = self.compute_slopes(P_i)
        r = self.r_integ_Y
        total = 0.0
        for i in range(r.shape[0] - 2):
            a = alphas[i]
            r0, r1 = r[i], r[i + 1]
            exp_a = 3.0 - a
            near3 = jnp.abs(exp_a) < 1e-8
            safe_exp = jnp.where(near3, 1.0, exp_a)
            normal = (r1**safe_exp - r0**safe_exp) / safe_exp
            # r0 may be 0; guard log(r0/...) -> use the normal branch there.
            limit = jnp.where(
                r0 > 0, jnp.log(r1) - jnp.log(jnp.where(r0 > 0, r0, 1.0)), 0.0
            )
            shell = P_i[i] * r1**a * jnp.where(near3, limit, normal)
            total = total + shell
        return 4.0 * jnp.pi * self.sz_fact * total
