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

# masked maps
flist=glob('*10_drc*fits')
for f in flist:
    weightmap = fits.open(f)[-3]
    binary_mask = np.where((weightmap.data>=1), 1, np.nan)
    hdul = fits.open(f)
    dtype = type(hdul[1].data[10,10])
    hdul[1].data *= np.array(binary_mask, dtype = np.float32)
    hdul[1].data = np.array(hdul[1].data, dtype = np.float32)
    hdul.writeto(f.replace('_drc.fits', '_masked_drc.fits'),overwrite=True)

tweakreg.TweakReg('jby*masked_drc.fits',
              enforce_user_order=False,
              imagefindcfg={'threshold': 0.9, 'conv_width': 3.5,'maxflux':50, 'dqbits': ~4096, 'use_sharp_round':True},
              refimage='../F658N_ACS.fits',
              refimagefindcfg={'threshold': 155, 'conv_width': 2.5,'maxflux':220000},
              shiftfile=True,
              outshifts='shiftfile.txt',
              searchrad=500.0,
              updatehdr=True,
              wcsname='UVIS_FLT',
              reusename=True,
              interactive=False)
              #yoffset = 0,
              #xoffset = 0)




    
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
    outputfile = filepath.replace('_drc.fits', '_drc.acs.coords')
    #if os.path.exists(outputfile):
    #    print('return')
    #    return 
    print(filepath)
    ds9_command = "ds9 "+filepath+" -regions load ACS_radec.reg -regions system image "
    ds9_command += '-regions format XY -regions save '+outputfile + ' -exit'
    os.system(ds9_command)

photometry_frames = np.sort(glob('./drz/*masked*drc*fits'))
Parallel(n_jobs=20)(delayed(get_xy_coords)(i) for i in photometry_frames)


##############################################
## Photometry using IRAF                   ###
##############################################


flist = glob('./drz/*10_masked_drc.fits')

def run_iraf(fitsfile, chunk_id):
    hdu = fits.open(fitsfile)[0]
    filename = fitsfile[6:]
    exptime = hdu.header['EXPTIME']
    filter_ = hdu.header['FILTER1']
    zmag  = {'F658N':18.107}[filter_]
    print(fitsfile)
    coordfile = fitsfile.replace('_drc.fits', '_drc.acs.coords')
    target_dir = './photfiles/'+filename.replace('.fits', '.phot')
    centroid_alg = 'centroid'
    
    iraf_script_images = open('irafscript'+chunk_id+'.cl', 'a')
    iraf_script_images.write('digiphot.apphot.phot image='+fitsfile+'[1] ')
    iraf_script_images.write('coords='+coordfile+' output='+target_dir + ' ')
    iraf_script_images.write('salgori=median annulus=4 dannulus=3 apertur=3 zmag='+str(zmag))
    iraf_script_images.write(' interac=no verify=no ')
    iraf_script_images.write('calgori='+centroid_alg +' minsnra=5 maxshif=2 cbox=3 datamin=0 datamax=INDEF ')
    iraf_script_images.write('gain=CCDGAIN readnoi=3.05')
    #if wcsin=='world': 
    #    iraf_script_images.write('wcsin=world ')
    iraf_script_images.write(' itime='+str(exptime))
    iraf_script_images.write(5*'\n')
    iraf_script_images.close()


for which_chunk, files in enumerate(np.array_split(flist,20)):
    for f in files:
        run_iraf(f, str(which_chunk))


scripts = np.sort(glob('irafscript*.cl'))
for arr in np.array_split(scripts,5):
    for j in arr:
        print('cl < '+j+' &')
    print('\n')













#all_stars = Table.read('../../../HST_Guido/30dor_all_newerr.UBVIHa.rot', format='ascii').to_pandas()
#all_stars.columns = 'ID;x;y;RA;Dec;u_1;eu_2;b_1;eb_2;v_1;ev_2;i_1;ei_2;ha_1;eha_2'.split(';')
#all_stars = all_stars.set_index('ID')

#deltaX  = all_stars.x.values - all_stars.x.values[:,np.newaxis]
#deltaY  = all_stars.y.values - all_stars.y.values[:,np.newaxis]
#deltaPix = np.array(np.sqrt(deltaX**2+deltaY**2), dtype=np.float32)
# throw away mergers
#deltaPix = pd.DataFrame(deltaPix, index=np.arange(1,len(deltaPix)+1,1), columns = np.arange(1,len(deltaPix)+1,1))

#deltaPix[deltaPix==0] = 999
#mindists = deltaPix.min(axis=1)
#to_keep = mindists.index[mindists>=8]



flist = glob('./photfiles/j*phot')
concatdf = ascii.read(flist[0]).to_pandas()
concatdf = concatdf[concatdf.PERROR == 'NoError']
concatdf = concatdf.set_index(['ID', 'IMAGE'])
concatdf = concatdf[~concatdf.MERR.isna()]

for f in flist[1:]:
    df = ascii.read(f).to_pandas()
    df = df[df.PERROR == 'NoError']
    df = df.set_index(['ID', 'IMAGE'])
    df = df[~df.MERR.isna()]
    concatdf = pd.concat((concatdf, df))
    concatdf = concatdf.drop_duplicates()
        
concatdf = concatdf.sort_index()
concatdf.to_pickle('./photfiles/concat_raw.pickle')



#concatdf = concatdf.loc[to_keep]


"""

for image in concatdf.reset_index()['IMAGE'].unique():
    image_df = concatdf[concatdf.index.get_level_values(1)==image]
    # get distances
    distances = np.sqrt(((image_df.XCENTER.values - image_df.XCENTER.values[:,np.newaxis])**2 + 
                (image_df.YCENTER.values - image_df.YCENTER.values[:,np.newaxis])**2))
    distances = pd.DataFrame(distances)
    distances[distances==0]=1000
    mindist = distances.min(axis=1)
    #idxmindist = distances.idxmin(axis=1)
    to_keep = np.where(mindist.values>10, True, False)#idxmindist[mindist>0].values
    image_df = image_df[to_keep]
    try:
        concatdf_nomergers = pd.concat((concatdf_nomergers, image_df))
    except:
        concatdf_nomergers = image_df
concatdf_nomergers = concatdf_nomergers.drop_duplicates()"""

concatdf = concatdf[concatdf.MERR<=0.25]
#concatdf = concatdf[(concatdf.XCENTER>210)&(concatdf.XCENTER<3950)]
#concatdf = concatdf[(concatdf.YCENTER>500)&(concatdf.YCENTER<4400)]
concatdf = concatdf[(concatdf.MAG>14.)&(concatdf.MAG<30)] 

concatdf.to_pickle('./photfiles/concat_clean.pickle')

Nmeas = concatdf.reset_index().groupby('ID')['IMAGE'].nunique()
concatdf = concatdf.loc[Nmeas.index[Nmeas>=3].values].sort_index()


variations_df = concatdf.groupby('ID')['MAG'].max() - concatdf.groupby('ID')['MAG'].min()
variations_df = pd.DataFrame({'MaxminMin':variations_df})
variations_df = (pd.merge(variations_df, concatdf.groupby('ID')['MERR'].median(), 
                 left_index=True, right_index=True))
variations_df['MaxminMin_sigma'] = variations_df.MaxminMin / variations_df.MERR


variations_df.to_pickle('./photfiles/ACS_variations.pickle')




