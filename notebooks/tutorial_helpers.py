"""Small shared helpers for the Roman GRS disperser tutorials.

These are *not* part of ``roman_disperser`` — they are thin convenience
wrappers that package up boilerplate the tutorials would otherwise repeat
(turning a synphot template into a count-rate spectrum, loading a
sensitivity curve). The first notebook (``01_spectra_to_counts``) builds
these up step by step and explains the units; everything after that just
imports them from here so the narrative can stay focused on the disperser.

Units convention used throughout the tutorials
-----------------------------------------------
- Wavelengths are **microns** for anything that touches the optical model
  / PSF / disperser (``optical_model_jax``, ``psf_model``,
  ``star_disperser``, ``galaxy_disperser``).
- Wavelengths are **Angstroms** for synphot and for the sensitivity FITS
  files (and the production catalog's SED grid).
- Fluxes are FLAM = erg / s / cm^2 / Angstrom.
- Count rates are electrons / s (per wavelength bin).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import astropy.units as u
import synphot as syn
from astropy.io import fits

from roman_disperser import paths


def grism_wavelength_grid(lam_min_um=0.9, lam_max_um=2.0, dlam_angstrom=2.0):
    """Uniform wavelength grid covering the grism band.

    Returns ``(wl_um, wl_angstrom, dlam_angstrom)``. The default (0.9-2.0 um,
    2 A spacing) matches the production catalog's SED grid. Use a coarser
    ``dlam_angstrom`` to make the dispersers run faster for quick experiments.
    """
    wl_a = np.arange(lam_min_um * 1e4, lam_max_um * 1e4 + dlam_angstrom / 2, dlam_angstrom)
    return wl_a / 1e4, wl_a, dlam_angstrom


def load_sensitivity(sca, order, wl_angstrom, sensitivity_dir=None):
    """Load a grism sensitivity curve and interpolate onto ``wl_angstrom``.

    ``sca`` is the 1-18 detector number, ``order`` is a string ("0"/"1"/"2").
    The curve converts a FLAM spectrum into a per-Angstrom count rate; off the
    tabulated range the sensitivity is taken to be zero. Returns a NumPy array
    the same length as ``wl_angstrom``.
    """
    sensitivity_dir = Path(sensitivity_dir) if sensitivity_dir else paths.sensitivity_dir()
    smap = yaml.safe_load((sensitivity_dir / "sensitivity_map.yaml").read_text())
    sfile = sensitivity_dir / smap[f"SCA{sca}"][str(order)]
    with fits.open(sfile) as hdul:
        s_wl = hdul[1].data["WAVELENGTH"]
        s_val = hdul[1].data["SENSITIVITY"]
    return np.interp(wl_angstrom, s_wl, s_val, left=0.0, right=0.0)


def template_to_counts(template_name, mag, sca, order, wl_um=None, band=None,
                       redshift=0.0, sensitivity_dir=None):
    """Turn a bundled spectral template into a count-rate spectrum.

    Steps (all explained in notebook 01):
      1. load the synphot template by name and (optionally) redshift it;
      2. normalise it to ``mag`` (AB) in the given bandpass (``band``
         defaults to F158);
      3. sample it onto the wavelength grid as FLAM;
      4. multiply by the grism sensitivity (SCA, order) and the bin width
         to get a count rate per wavelength bin.

    ``redshift`` matters for the galaxy templates: the Kinney-Calzetti
    atlas spectra are rest-frame UV-optical (~1235-9945 A) and only fall in
    the grism band once redshifted, so a star uses ``redshift=0`` while a
    galaxy needs e.g. ``redshift=1.5``. ``wl_um`` defaults to
    :func:`grism_wavelength_grid`. Returns ``(wl_um, counts)`` where
    ``counts`` is electrons/s per bin (a plain NumPy array ready to hand to
    a disperser after ``jnp.asarray``).
    """
    from roman_disperser import refdata

    if wl_um is None:
        wl_um, wl_a, dlam_a = grism_wavelength_grid()
    else:
        wl_um = np.asarray(wl_um)
        wl_a = wl_um * 1e4
        dlam_a = np.diff(wl_a).mean()
    if band is None:
        band = refdata.get_f158_band()

    spec = refdata.get_template(template_name)
    if redshift:
        spec = syn.SourceSpectrum(spec.model, z=redshift)
    spec = spec.normalize(mag * u.ABmag, band=band)
    flam = spec(wl_a * u.AA, flux_unit=syn.units.FLAM).value
    sens = load_sensitivity(sca, order, wl_a, sensitivity_dir=sensitivity_dir)
    counts = flam * sens * dlam_a
    return wl_um, counts


# ---------------------------------------------------------------------------
# Minimal 1D spectral extraction (built up by hand in notebook 04). There is no
# extraction routine in roman_disperser itself; these are teaching helpers.
# Dispersion runs along the SCA y axis, so we sum cross-dispersion (x) pixels at
# each wavelength's trace position. Wavelengths are MICRONS throughout.
# ---------------------------------------------------------------------------

def spectral_trace(optical_payload, xsca, ysca, wl_um):
    """Trace (x, y) SCA pixel positions (1-indexed) for each wavelength.

    Chains the JAX optical-model transforms sca->fpa->trace->mpa->sca for an
    undispersed source at ``(xsca, ysca)``.
    """
    import jax.numpy as jnp
    import roman_disperser.optical_model_jax as omj

    wl = jnp.asarray(wl_um)
    xfpa, yfpa = omj.sca_to_fpa(optical_payload, xsca, ysca)
    xmpa, ympa = omj.trace_beam(optical_payload, jnp.broadcast_to(xfpa, wl.shape),
                                jnp.broadcast_to(yfpa, wl.shape), wl)
    tx, ty = omj.mpa_to_sca(optical_payload, xmpa, ympa)
    return np.asarray(tx).ravel(), np.asarray(ty).ravel()


def dispersion(optical_payload, xsca, ysca, wl_um):
    """dy/dlambda (pixels per micron) along the trace, via autodiff (jax.grad)."""
    import jax
    import jax.numpy as jnp
    import roman_disperser.optical_model_jax as omj

    def trace_y(wl):
        wl1 = jnp.atleast_1d(wl)
        xfpa, yfpa = omj.sca_to_fpa(optical_payload, xsca, ysca)
        xmpa, ympa = omj.trace_beam(optical_payload, jnp.broadcast_to(xfpa, wl1.shape),
                                    jnp.broadcast_to(yfpa, wl1.shape), wl1)
        _, ty = omj.mpa_to_sca(optical_payload, xmpa, ympa)
        return ty[0]

    grad = jax.jit(jax.vmap(jax.grad(trace_y)))
    return np.asarray(grad(jnp.asarray(wl_um)))


def extract_1d(image, optical_payload, xsca, ysca, wl_um, aperture=10):
    """Boxcar-extract a 1D spectrum along the trace of a source at (xsca, ysca).

    For each wavelength: find the trace pixel, sum +/- ``aperture`` pixels in the
    cross-dispersion (x) direction, then scale by ``|dy/dlambda| * dlambda`` to
    convert counts-per-pixel-row into counts per wavelength bin (same units as
    the input count-rate spectrum). Returns a NumPy array the length of wl_um.
    """
    tx, ty = spectral_trace(optical_payload, xsca, ysca, wl_um)
    dydl = dispersion(optical_payload, xsca, ysca, wl_um)
    dlam_um = float(np.diff(np.asarray(wl_um)).mean())
    raw = np.zeros(len(wl_um))
    for i in range(len(wl_um)):
        ix = int(round(float(tx[i]))) - 1     # 1-indexed FITS -> 0-indexed array
        iy = int(round(float(ty[i]))) - 1
        if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]:
            raw[i] = image[iy, max(0, ix - aperture):ix + aperture + 1].sum()
    return raw * np.abs(dydl) * dlam_um
