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
