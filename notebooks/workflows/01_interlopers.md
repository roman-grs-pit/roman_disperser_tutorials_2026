# Interloper Calibration for LSS : A Toy Model

The following sets up a toy problem to explore interloper calibration, completeness
and purity in an LSS sample. 

NOTE : There are many parameters here that should be varied for a more 
realistic exploration. We leave that open to the reader. 

Assume all objects are galaxies, but are small enough to be treated as point sources. 
The continuum is assumed to be a flat spectrum in $f_\nu$ with an F158 AB magnitude=24.
Also assume that it has two lines - H$\alpha$ and OIII; assume that the H$\alpha$ line flux 
is $10^{-16}$ erg/s/cm$^2$ and OIII flux 0.3 times weaker. The line could be assumed to be 
10 Ang in width. 

Pick detector (SCA) positions between -500 to 4500 in $x$ and $y$, and redshifts uniformly 
chosen between $z=1$ and $z=2$. Set spectra based on the redshifts.

Simulate this scene for a single SCA. Repeat for a 15 degree roll (rotate around the SCA 
center for simplicity). Simulate both the 0th and 1st order.

For each exposure, assume a exposure time of 190s. Poisson sample the resulting counts. Add
in a uniform zodiacal light background of 1 e/s/pixel, poisson sampled.

For each object, do a simple boxcar extraction and compute the spectrum. 

Take each of these spectra and do a simple redshift fitting. Explore various completeness/purity
metrics as desired, varying parameters. 