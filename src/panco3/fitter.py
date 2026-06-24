"""PressureProfileFitter: orchestrates data, model, filtering, and priors.

JAX port of ``panco2.PressureProfileFitter`` (``panco2/panco2/panco2.py``),
keeping the same user-facing workflow:

    ppf = PressureProfileFitter(fits_file, hdu_data, hdu_rms, z, M_500, ...)
    ppf.define_model(r_bins)
    ppf.add_filtering(beam_fwhm=..., ell=..., tf=...)
    ppf.define_priors(P_bins=..., conv=..., zero=...)

The map and RMS are held as JAX arrays (differentiable boundary).
Cosmology is a ``jax_cosmo.Cosmology`` (see :mod:`panco3.cluster`).
Inference (BlackJAX NUTS) is built from this object in
:mod:`panco3.inference`.
"""

from __future__ import annotations

import dill
import numpy as np
import jax.numpy as jnp

from . import io, geometry, filtering
from .cluster import Cluster, default_cosmo
from .model import ModelBinned


class PressureProfileFitter:
    def __init__(
        self,
        sz_map_file,
        hdu_data,
        hdu_rms,
        z,
        M_500,
        coords_center=None,
        map_size=None,
        cosmo=None,
    ):
        if cosmo is None:
            cosmo = default_cosmo()
        self.cluster = Cluster(z, M_500=M_500, cosmo=cosmo)

        data = io.load_sz_fits(
            sz_map_file,
            hdu_data,
            hdu_rms,
            coords_center=coords_center,
            map_size=map_size,
        )
        # Differentiable boundary: data as JAX arrays.
        self.sz_map = jnp.asarray(data.sz_map)
        self.sz_rms = jnp.asarray(data.sz_rms)
        self.wcs = data.wcs
        self.pix_size = data.pix_size  # arcsec
        self.map_size = data.map_size  # arcmin
        self.coords_center = data.coords_center

        self.model = None
        self.radii = None
        self.covmat = None
        self.inv_covmat = None
        self.has_covmat = False
        self.has_integ_Y = False
        self.has_mask = False
        self.priors = None

    # ------------------------------------------------------------------ #
    @classmethod
    def load_from_file(cls, file_name):
        with open(file_name, "rb") as f:
            return dill.load(f)

    def dump_to_file(self, file_name):
        with open(file_name, "wb") as f:
            f.write(dill.dumps(self))

    # ------------------------------------------------------------------ #
    def define_model(
        self, r_bins, zero_level=True, n_nodes=32, r_max_integ=None
    ):
        """Define the binned-profile forward model (see panco2
        define_model)."""
        npix = int(np.asarray(self.sz_map).shape[0])
        self.radii = geometry.make_radii(npix, self.pix_size, self.cluster.d_a)
        self.model = ModelBinned(
            r_bins,
            self.radii,
            zero_level=zero_level,
            sz_fact=self.cluster.sz_fact,
            n_nodes=n_nodes,
            r_max_integ=r_max_integ,
        )
        return self.model

    # ------------------------------------------------------------------ #
    def add_filtering(self, beam_fwhm=0.0, pad=0, ell=None, tf=None):
        """Add beam + transfer-function filtering (see panco2
        add_filtering)."""
        self.beam_fwhm = beam_fwhm
        beam_sigma_pix = (
            beam_fwhm / (2 * np.sqrt(2 * np.log(2))) / self.pix_size
        )
        npix = int(np.asarray(self.sz_map).shape[0])

        if tf is None:
            self.model.filter = filtering.Filter(beam_sigma_pix)
        elif isinstance(ell, np.ndarray):
            assert tf.shape == ell.shape, (
                "tf and ell don't have the same shape"
            )
            self.model.filter = filtering.Filter1d(
                npix,
                self.pix_size,
                ell / (360.0 * 3600),
                tf,
                beam_sigma_pix=beam_sigma_pix,
                pad_pix=pad,
            )
        elif isinstance(ell, (tuple, list)):
            assert tf.shape == (len(ell[0]), len(ell[1])), (
                "ell and tf shapes incompatible"
            )
            self.model.filter = filtering.Filter2d(
                npix,
                self.pix_size,
                ell[0] / (360.0 * 3600),
                ell[1] / (360.0 * 3600),
                tf,
                beam_sigma_pix=beam_sigma_pix,
                pad_pix=pad,
            )
        else:
            raise ValueError("Could not interpret (ell, tf); check inputs.")

    # ------------------------------------------------------------------ #
    def add_point_sources(self, coords, beam_fwhm):
        beam_sigma_pix = (
            beam_fwhm / (2 * np.sqrt(2 * np.log(2))) / self.pix_size
        )
        self.model.add_point_sources(coords, self.wcs, beam_sigma_pix)

    def add_integ_Y(self, Y, dY, r):
        self.integ_Y = (Y, dY)
        self.r_integ_Y = r
        self.model.init_integ_Y(r)
        self.has_integ_Y = True

    def add_covmat(self, covmat=None, inv_covmat=None):
        """Provide a noise covariance (or its inverse) for correlated noise."""
        if inv_covmat is None:
            inv_covmat = np.linalg.inv(np.asarray(covmat))
        self.covmat = None if covmat is None else jnp.asarray(covmat)
        self.inv_covmat = jnp.asarray(inv_covmat)
        self.has_covmat = True

    # ------------------------------------------------------------------ #
    def define_priors(self, P_bins=None, conv=None, zero=None, ps_fluxes=None):
        """Store the prior specification.

        Each entry is a (loc/scale or bounds) spec consumed by
        :mod:`panco3.priors` to build differentiable log-densities. Accepts
        the same kinds of per-parameter priors as panco2 (one per pressure
        bin, plus ``conv``, ``zero``, and one per point-source flux).
        """
        self.priors = {
            "P_bins": P_bins,
            "conv": conv,
            "zero": zero,
            "ps_fluxes": ps_fluxes if ps_fluxes is not None else [],
        }

    # ------------------------------------------------------------------ #
    def model_map(self, par_vec):
        """Full model map (filtered SZ + point sources)."""
        return self.model.model_map(par_vec)
