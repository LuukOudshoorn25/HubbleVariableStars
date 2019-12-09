import numpy as np
from astropy.io import fits
from glob import glob
import os
from joblib import Parallel, delayed
from astropy.io import ascii
import time
tstart = time.time()
from glob import glob
import errno
import os
from astropy.table import Table
import pandas as pd
from shutil import copyfile
from sys import exit
import sys
from joblib import Parallel, delayed
import multiprocessing
from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt
from astroquery.mast import Observations

#from ccdproc import ImageFileCollection
from astropy.io import ascii
import shutil
from drizzlepac import tweakreg
from astropy.stats import SigmaClip
from photutils import StdBackgroundRMS
from astropy.wcs import WCS
from stsci.skypac import pamutils
from drizzlepac import astrodrizzle
from reproject import reproject_interp, reproject_exact
from astropy.wcs import WCS
from astropy import wcs
from drizzlepac import skytopix
from drizzlepac import pixtosky
from scipy import interpolate
folderlist = np.sort(glob('jb*10'))

for folder in folderlist: 
    os.chdir(folder) 
    refimage = '../../drz/'+folder+'_drc.fits'
    """tweakreg.TweakReg('*_flc.fits', 
        enforce_user_order=False, 
        imagefindcfg={'threshold': 80, 'conv_width': 3.5, 'dqbits': ~4096, 'use_sharp_round':True}, 
        refimage=refimage,
        refimagefindcfg={'threshold': 2, 'conv_width': 2.5}, 
        shiftfile=True, 
        outshifts='shiftfile.txt', 
        searchrad=300.0, 
        ylimit=0.6, 
        updatehdr=True, 
        wcsname='UVIS_FLT', 
        reusename=True, 
        interactive=False) """
    drz_output = folder+'_LuukDrizzle.fits'
    """astrodrizzle.AstroDrizzle('*flc*fits',
        output=drz_output,
        preserve=True,
        overwrite=True,
        clean=False,
        build=True,
        context=False,
        resetbits=0,
        #driz_separate=True,
        #median=True,
        #blot=True,
        #driz_cr=True,
        #driz_cr_corr=True,
        driz_cr_grow=1,
        driz_cr_snr= '10 10',
        final_wcs=True,
        skysub=False,
        final_refimage=refimage)"""


    mask_hdu = fits.open(drz_output).copy()
    mask_hdu[1].data = np.array(np.where(fits.open(drz_output.replace('.fits', '_med.fits'))[0].data>1e-6, 1, 0), dtype=np.float32)
    mask_hdu.writeto(folder+'_mask.fits', overwrite=True)
    os.chdir('../')







#multiply
flist=glob('*drc.fits')
for f in flist:
    mask = glob('../singles/'+f.split('_drc.fits')[0]+'/'+f.split('_drc.fits')[0]+'_mask.fits')
    mask = fits.open(mask[0])[1].data
    masked_hdu = fits.open(f)

    new_shape = masked_hdu[1].data.shape
    
    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])
    
    xnew = np.arange(new_shape[1])
    ynew = np.arange(new_shape[0])

    func = interpolate.interp2d(x, y, mask, kind='linear')
    new_mask = func(xnew, ynew)
    new_mask = np.where(new_mask>0, 1, np.nan)
    masked_hdu[1].data = 2220*masked_hdu[1].data * new_mask
    masked_hdu.writeto(f[:-5]+'_masked.fits', overwrite=True)





























