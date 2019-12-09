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
from drizzlepac import astrodrizzle
from reproject import reproject_interp, reproject_exact

import operator

tweakreg.TweakReg('jby*_drc.fits',
              enforce_user_order=False,
              imagefindcfg={'threshold': 0.9, 'conv_width': 3.5,'maxflux':50, 'dqbits': ~4096, 'use_sharp_round':True},
              refimage='../F658N_ACS.fits',
              refimagefindcfg={'threshold': 155, 'conv_width': 2.5,'maxflux':180000},
              shiftfile=True,
              outshifts='shiftfile.txt',
              searchrad=20000.0,
              ylimit=0.6,
              updatehdr=True,
              wcsname='UVIS_FLT',
              reusename=True,
              interactive=False)
              #yoffset = 0,
              #xoffset = 0)



# masked maps
flist=glob('*drc*fits')
for f in flist[:1]:
    weightmap = fits.open(f)[-3]
    binary_mask = np.where((weightmap.data>=350)&(weightmap.data!=64), 1, np.nan)
    hdul = fits.open(f)[:2]
    hdul[1].data *= binary_mask
    hdul.writeto(f.replace('_drc.fits', '_masked_drc.fits'),overwrite=True)
    
# Convert Elenas list to XY coordinates in all these frames
# Get radec list
#df = Table.read('FullCatalogue.txt', format='ascii').to_pandas()
df = pd.read_pickle('FullCatalogue.updatedWCS.pickle')
df = df[(df.M658>12)&(df.M658<35)]
df = df[df.E658<2]
df = df[(df.M555<35)&(df.M555>12)]
df = df[df.E555<2]
df = df[df.E658max<35]
df[['RA', 'DEC']].to_csv('ACS_radec.coo', index=None, header=None, sep='\t')
##############################################
## DS9 coordinate transformations          ###
##############################################

def get_xy_coords(filepath):
    outputfile = filepath.replace('_drc.fits', '_drc.acscoords')
    #if os.path.exists(outputfile):
    #    print('return')
    #    return 
    ds9_command = "ds9 "+filepath+" -regions load ACS_radec.reg -regions system image "
    ds9_command += '-regions format XY -regions save '+outputfile #+ ' -exit'
    os.system(ds9_command)

photometry_frames = np.sort(glob('./drz/jby01p010_drc.fits'))
Parallel(n_jobs=1)(delayed(get_xy_coords)(i) for i in photometry_frames)











flist=glob('*flt.fits') 
t_start = {}
for f in flist:
    t_start[f] = fits.open(f)[0].header['EXPSTART']
t_start = sorted(t_start.items(), key=operator.itemgetter(1))

flist = [w[0] for w in t_start]


for i in range(len(flist)-2):
    if not os.path.exists('./set'+str(i)):
        os.mkdir('./set'+str(i))
    for f in [flist[i], flist[i+1], flist[i+2]]:
        copyfile(f, './set'+str(i)+'/'+f)
	
for i in range(5,len(flist)-2):
    os.chdir('./set'+str(i))
    astrodrizzle.AstroDrizzle('*flt*',
        output='drz.fits',
        preserve=True,
        overwrite=True,
        clean=False,
        build=True,
        context=False,
        resetbits=0,
        driz_separate=True,
        median=True,
        blot=True,
        driz_cr=True,
        driz_cr_corr=True,
        driz_cr_grow=1,
        driz_cr_snr= '10 10',
        final_wcs=True,
        skysub=False,
        driz_sep_refimage = '../../../HST_Guido/30dorf814drz.fits',
        final_refimage='../../../HST_Guido/30dorf814drz.fits')
    os.chdir('../')


for i in range(6,len(flist)):
    os.chdir('./set'+str(i))
    hdu = fits.open('drz_med.fits')
    files = glob('*single*wht*')
    weightmaps = [fits.open(w)[0].data for w in files]
    weightmaps = np.array(weightmaps)
    weightmaps = np.clip(weightmaps, 0, 1)
    weightmap = np.sum(weightmaps, axis=0)
    weightmap = np.where(weightmap>1, 1, 0).astype(type(hdu[0].data[42,42]))
    hdu[0].data = weightmap
    hdu.writeto('weightmap'+str(i)+'.fits', overwrite=True)
    os.chdir('../')



os.chdir('./sextractor')
for i in range(6,len(flist)):
    expfile = '../set'+str(i)+'/drz_med.fits'
    	
    weightfile = '../set'+str(i)+'/weightmap'+str(i)+'.fits'
    sexcommand = 'sex ../deep_exposure/drz_med.fits,'+expfile+' -c params.sex -CATALOG_NAME '
    sexcommand += 'med'+str(i)+'.cat -WEIGHT_IMAGE '+'../deep_exposure/deep_weightmap.fits'+','+weightfile
    os.system(sexcommand)
os.chdir('../')







os.chdir('./sextractor')

 


