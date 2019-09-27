# HubbleVariableStars
Here will be code on analyzing the Ha variability of PMS stellar objects in the Tarantula Nebula / 30-dor
The calibrate_wfc3.py script contains functions to 
    1. Sort Hubble downloaded images into tree structure (filter -> deep/short)
    2. AstroDrizzle the single science images onto a common WCS header (defined by template.fits, made in earlier analysis)
    3. Regrid images using Lancos3 flux-conserving algorithm. There are both single core and parallel implementations of this
    4. Perform astrometric calibrations using SEX and SCAMP
    5. Perform Cosmic Ray masking using drizzled aligned framed and sigma clipping. Here I assume BG STDDEV scales linearly with flux
    6. Export CR masked, exposure time multiplied FITS or IRAF
    6. Prepare IRAF scripts to do aperture photometry
    7. Concat all the single IRAF phot files to make one huge catalogue of magnitudes. 
