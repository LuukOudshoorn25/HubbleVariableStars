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


scamp_ex = 'scamp '
### PARAMS ###
SORT           = False
CREATE_TREE    = False
DRIZZLE_IMS    = False
SPLIT_FITS     = True
DRIZZLE_4IMS   = False
DO_ASTROMETRY  = False
DO_ASTROMETRY4 = False
DO_REFERENCE   = False
DO_REGRID      = False
DO_REGRID_PAR  = False
DO_REGRID4     = False	
DO_REGRID4_PAR = False
DO_APPHOT      = False
pms_stars      = False
recenter       = False
IRAF_parallel  = False
DO_GET_NBadPIX = False
DO_APPHOT4     = False
write_DS9_reg  = False
DO_IRAF_DF     = False
DO_IRAF_NCR_DF = False
### Function definitions ###
def initialize():
    return
    #os.system('rm -rf ./working_dir/*')
    #os.system('cp ./SEXfiles/* ./working_dir/')

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    else:
        print("Successfully created the directory %s " % path)


def sort_files():

    fitslist = glob('*fl*fits')
    filterlist = []
    for enum, fitsfile in enumerate(fitslist):
        print("Going through the fits to see unique filters, now at ", enum, " of ", len(fitslist), end='\r')
        hdul = fits.open(fitsfile)
        hst_filter = hdul[0].header['FILTER']
        filterlist.append(hst_filter)
    print(5*'\n')
    unique_filters = np.unique(filterlist)

    for unique_filter in unique_filters:
        path = "./SORTED/"+str(unique_filter)+'/deep'
        mkdir(path)


    for enum, fitsfile in enumerate(fitslist):
        hdul = fits.open(fitsfile)
        hst_filter = hdul[0].header['FILTER']
        exptime = hdul[0].header['EXPTIME']
        print("Copying the fits, now at ", enum, " of ", len(fitslist), end='\r')
        if exptime < 60:
                target_dir = './SORTED/'+hst_filter+'/short/'
        else:
                target_dir = './SORTED/'+hst_filter+'/deep/'
        try:
            if not os.path.exists(target_dir+'/'+fitsfile):
                copyfile(fitsfile, target_dir+'/'+fitsfile)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)


def create_dir_tree():
    filterlist=[]
    flist = glob('./SORTED/*/*/*fl*fits')
    for enum, fitsfile in enumerate(flist):
        print("Going through the fits to see unique filters, now at ", enum, " of ", len(flist), end='\r')
        hdul = fits.open(fitsfile)
        hst_filter = hdul[0].header['FILTER']
        filterlist.append(hst_filter)
    print(5*'\n')
    unique_filters = np.unique(filterlist)

    for unique_filter in unique_filters:
        path = "./FLT_exposures/"+str(unique_filter)+'/deep'
        if not os.path.exists(path):
            os.makedirs(path)
        path = "./FLT_exposures/"+unique_filter+'/short'
        if not os.path.exists(path):
            os.makedirs(path)






def make_PAM_maps(filepath):
    pam_wfc1 = './FLT_exposures/PAM_wfc1.fits'
    pam_wfc2 = './FLT_exposures/PAM_wfc2.fits'
    pamutils.pam_from_file(filepath, ext=1, output_pam=pam_wfc2)
    pamutils.pam_from_file(filepath, ext=4, output_pam=pam_wfc1)


def split_fits(filepath):
    target_dir1 = filepath.replace('FLT_files_shiftcoords', 'FLT_exposures')[:-5]+'_wfc1.fits'
    target_dir2 = filepath.replace('FLT_files_shiftcoords', 'FLT_exposures')[:-5]+'_wfc2.fits'
    if not os.path.exists(target_dir1) * os.path.exists(target_dir2):
        hdul = fits.open(filepath)
        wfc2 = fits.HDUList([hdul[0], hdul[1]])
        wfc1 = fits.HDUList([hdul[0], hdul[4]])
        wfc2.writeto(target_dir2, overwrite=True)
        wfc1.writeto(target_dir1, overwrite=True)
        print(target_dir2)


def drizzard():
    print(glob('*_flt.fits'))
    tweakreg.TweakReg('*_flt.fits',
                      enforce_user_order=False,
                      imagefindcfg={'threshold': 50, 'conv_width': 3.5, 'dqbits': ~4096, 'use_sharp_round':True},
                      refimage='../HST_Guido/30dorf814drz.fits',
                      refimagefindcfg={'threshold': 30, 'conv_width': 2.5},
                      shiftfile=True,
                      outshifts='shiftfile.txt',
                      searchrad=300.0,
                      ylimit=0.6,
                      updatehdr=True,
                      wcsname='newheader',
                      reusename=True,
                      interactive=False,
                      updatewcs=False,
                      yoffset = -2000,
                      xoffset = -500)




def multiply_with_PAM_exptime(filepath, PAM_MAP):
    output_filepath =filepath[:-5]+'_pamcorr_exptime.fits' 
    print(filepath)
    #if os.path.exists(output_filepath):
    #    return

    hdul = fits.open(filepath)
    exptime = hdul[0].header['EXPTIME']    
    hdul[1].data = hdul[1].data * PAM_MAP * 1
    
    hdul.writeto(output_filepath, overwrite=True)


def get_xy_coords(filepath):
    # which_wfc = ('wfc1' if 'wfc1' in filepath else 'wfc2')
    skytopix.rd2xy(filepath+'[sci,1]',coordfile="radec.dat", output = filepath[:-5]+'_wfc2.coordfile')
    skytopix.rd2xy(filepath+'[sci,2]',coordfile="radec.dat", output = filepath[:-5]+'_wfc1.coordfile')
    return

def sort_files():
    os.chdir('./FLT_files_shiftcoords/')
    fitslist = glob('*flt.fits')
    print(fitslist)
    filterlist = []
    for enum, fitsfile in enumerate(fitslist):
        print("Going through the fits to see unique filters, now at ", enum, " of ", len(fitslist), end='\r')
        hdul = fits.open(fitsfile)
        hst_filter = hdul[0].header['FILTER']
        filterlist.append(hst_filter)
    print(5*'\n')
    unique_filters = np.unique(filterlist)


    for enum, fitsfile in enumerate(fitslist):
        hdul = fits.open(fitsfile)
        hst_filter = hdul[0].header['FILTER']
        exptime = hdul[0].header['EXPTIME']
        print("Copying the fits, now at ", enum, " of ", len(fitslist), end='\r')
        if exptime < 60:
                target_dir = '../FLT_exposures/'+hst_filter+'/short'
        else:
                target_dir = '../FLT_exposures/'+hst_filter+'/deep'
        try:
            if not os.path.exists(target_dir+'/'+fitsfile):
                copyfile(fitsfile, target_dir+'/'+fitsfile)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)
    os.chdir('../')




#create_dir_tree()

CREATE_PAM=False
SORT_FILES = True
SPLIT_FITS=True
MULT_PAM = True
UPDATE_WCS=True

if CREATE_PAM:
    filepath = glob('./SORTED/F8*/*/*flt*fits')[0]
    make_PAM_maps(filepath)


if SORT_FILES:
    sort_files()

if SPLIT_FITS:
    flist = glob('./FLT_exposures/*/*/*flt*fits')
    for im in flist:
        if 'F110W' in im or 'F160W' in im:
            continue
        split_fits(im)

if MULT_PAM:
    Pam_map_wfc1 = fits.open('./FLT_exposures/PAM_wfc1.fits')[0].data
    Pam_map_wfc2 = fits.open('./FLT_exposures/PAM_wfc2.fits')[0].data
    wfc1_images = glob('./FLT_exposures/*/*/*flt*wfc1.fits')
    wfc2_images = glob('./FLT_exposures/*/*/*flt*wfc2.fits')
    for im in wfc1_images:
        if 'F110W' in im or 'F160W' in im:
            continue
        multiply_with_PAM_exptime(im, Pam_map_wfc1)
    for im in wfc2_images:
        if 'F110W' in im or 'F160W' in im:
            continue
        multiply_with_PAM_exptime(im, Pam_map_wfc2)

if UPDATE_WCS:
    folderlist = np.sort(glob('./SORTED/*/*/'))
    for folder in folderlist:
        print('Starting with ', folder)
        ims = glob(folder+'*flt.fits')
        for iter_, im in enumerate(ims):
            root = im.split('/')[-1]
            copyfile(im, './working_dir/'+root)
            print("Copied {} out of {}".format(iter_, len(ims)), end='\r')
        os.chdir('./working_dir')
        print("Moved into working dir")
        drizzard()
        #for im in ims:
        #    root = im.split('/')[-1]
        #    result = root[:-8]+'single_sci.fits'
        #    drizzfile = root[:-5]+'_drz_sci.fits'
        #    print(result)
        #    copyfile(result, '.'+folder.replace('SORTED', 'DRIZZLED')+drizzfile)
        os.chdir('../')
        os.system('rm -rf ./working_dir/*') 




RADEC2XY=True
if RADEC2XY:
    splitted_multiplied_ims = glob('./FLT_exposures/*/*/*flt.fits')
    for im in splitted_multiplied_ims:
        if 'F110W' in im or 'F160W' in im:
            continue
        get_xy_coords(im)

IRAF_parallel=True
DO_APPHOT = True




if DO_APPHOT:
    """Correct the FITS files by masking cosmic rays, multiply them by their exposure time 
       and write the CR masks to FITS files"""
    os.chdir('./working_dir')
    ims = glob('../FLT_exposures/*/*/*_pamcorr_exptime.fits')
    all_files = ims.copy()

    if IRAF_parallel:
        nsplits=6
    else:
        nsplits=1
    for which_chunk, drizzled_astrom_regrid_flist in enumerate(np.array_split(all_files,nsplits)):
        iraf_script_images  = open('app_phot_script'+str(which_chunk+1)+'.cl', 'w') 
        for f_count, im in enumerate(drizzled_astrom_regrid_flist): 
            folder = '/'.join(im.split('/')[:-1])+'/'
            print(im)
            splitted_dir = im.split('/')
            hst_filter   = splitted_dir[2]
            if (hst_filter == 'F110W') or (hst_filter == 'F160W'):
                continue
            hdu        = fits.open(im)[0]
            im_exptime = im[:-5]+'_exptime.fits'
            im_crmask  = im[:-5]+'_crmask.fits'
            print('Working on file {} from {}'.format(f_count, len(drizzled_astrom_regrid_flist)), end='\r')     
            #if not 0>np.inf:#(os.path.exists(im_exptime) and os.path.exists(im_crmask)):
            #   # CR clean
            #    flist                    = glob(folder+'*drz_sci_regrid.fits')
            #    hdu.data, sigma, CRmask  = GetCRMasked_exptime(flist, hdu.data, folder, hdu.header['EXPTIME'])
            #    hdu.writeto(im_exptime, overwrite=True)
                # Save CRmask to FITS file
            #    CRmask = CRmask.astype(type(hdu.data[555,555]))
            #    CR_hdu = hdu.copy()
            #    CR_hdu.data = CRmask
            #    CR_hdu.writeto(im_crmask, overwrite=True)
            #else:
                # If CRfile and CRcleaned file already exist, read stddevs from file
            #    stddevs_df = pd.read_csv('../stddevs.txt', delimiter='\t')
            #    stddevs_df.columns = ['folder', 'stddev']
            #    stddev = float(stddevs_df.groupby('folder').median().loc[folder])
            # Define magnitude zeropoints...
            zmag  = {'F336W':23.46,'F438W':24.98,'F555W': 25.81, 'F814W': 24.67, 'F656N': 19.92}[hst_filter]-0.1
#            if DO_APPHOT4:
#                if pms_stars:
#                    target_dir = (im[:-5]+'_pmsstars.phot ').replace('DRIZZLED', 'IRAF_cats_drz')
#                else:
#                    target_dir = (im[:-5]+'.phot ').replace('DRIZZLED', 'IRAF_cats_drz')
#            else:
#            if pms_stars:
#                target_dir = (im[:-5]+'_pmsstars.phot ').replace('DRIZZLED', 'IRAF_cats')
#            else:
            target_dir = (im[:-5]+'.phot').replace('FLT_exposures', 'IRAF_cats_FLT')
            centroid_alg = 'centroid'
            coordfile = im.split('_pamcorr')[0]+'.coordfile'
            # Write task per image to do aperture photometry in IRAF
            iraf_script_images.write('digiphot.apphot.phot image='+im+'[1] ')
            iraf_script_images.write('coords='+coordfile+' output='+target_dir + ' ')
            iraf_script_images.write('salgori=mode annulus=6 dannulus=3 apertur=5 zmag='+str(zmag) + ' interac=no verify=no ')
            iraf_script_images.write('calgori='+centroid_alg +' cbox=3 datamin=0 datamax=INDEF ')
            iraf_script_images.write('gain=CCDGAIN readnoi=3.05 sigma='+str(1) + ' itime='+str(hdu.header['EXPTIME']))
            iraf_script_images.write(5*'\n')
            del hdu
        iraf_script_images.close()
    os.chdir('../')































    

