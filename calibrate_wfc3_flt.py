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
    if os.path.exists(output_filepath):
        return
    print(filepath)
    #if os.path.exists(output_filepath):
    #    return

    hdul = fits.open(filepath)
    exptime = hdul[0].header['EXPTIME']    
    hdul[1].data = hdul[1].data * PAM_MAP * 1
    
    hdul.writeto(output_filepath, overwrite=True)


def get_xy_coords(filepath):
    # which_wfc = ('wfc1' if 'wfc1' in filepath else 'wfc2')
    skytopix.rd2xy(filepath+'[sci,1]',coordfile="radec_pms.dat", output = filepath[:-5]+'_wfc2.coordfile')
    skytopix.rd2xy(filepath+'[sci,2]',coordfile="radec_pms.dat", output = filepath[:-5]+'_wfc1.coordfile')
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


def GetCRMasked_exptime(flist, this_file_path, folderpath, exptime):
#flist, this_file_path, folderpath, exptime = this_wfc_ims, im, folder, hdu.header['EXPTIME']
    storepath_median = this_file_path.replace('exptime', 'median')
    print(this_file_path, exptime)
    this_file = fits.open(this_file_path)[1].data / exptime
    offset_file = this_file_path.replace('_pamcorr_exptime.fits', '_offset.fits')
    try:
        hdu_median = fits.open(storepath_median)
        medians = hdu_median[1].data
        print('Reading from offset fits')
    except:
        if len(flist)>6:
            flist = np.random.choice(flist, 6)
        all_frames = np.zeros((*this_file.shape, len(flist)))

        
        num_cores = max(6, len(flist))
        inputs = zip(flist, len(flist)*[this_file_path])
        results = Parallel(n_jobs=num_cores)(delayed(regrid_worker)(i) for i in [w for w in inputs])
        for i,file in enumerate(flist):
            all_frames[:,:,i] = results[i] / fits.open(file)[0].header['EXPTIME']
        too_little = np.where(np.sum(all_frames==0, axis=2)>4, np.nan, 1)
        medians = np.nanmedian(all_frames, axis=2)  * too_little
        del all_frames
        hdu_median = fits.open(this_file_path)
        hdu_median[1].data = medians
        hdu_median.writeto(storepath_median, overwrite=True)
    offsets = np.abs(this_file - medians)
    if not os.path.exists(offset_file):
        offsets_hdu = fits.open(this_file_path).copy()
        offsets_hdu.data = offsets
        offsets_hdu.writeto(offset_file, overwrite=True)
    rmsmapfile = this_file_path.replace('exptime', 'rmsmap')
    bg_stddev_arr = np.median(fits.open(rmsmapfile)[0].data) * medians / np.median(this_file)
    CRmask = np.where(offsets>20*bg_stddev_arr, -9999999, 0)
    this_file_corr = (this_file.copy() * exptime) + CRmask
    this_file_corr = np.clip(this_file_corr, -1,np.inf)
    return this_file_corr, np.median(bg_stddev_arr), np.clip(np.abs(CRmask), 0, 1)


def regrid_worker(input_tuple):
    im, refim = input_tuple
    ref_hdu  = fits.open(refim)[1]
    ref_fobj = fits.open(refim.split('_wfc')[0]+'.fits')
    ref_wcs  = WCS(ref_hdu, fobj=ref_fobj)

    singleexphdu = fits.open(im)[1]
    fobj = fits.open(im.split('_wfc')[0]+'.fits')
    wcs = WCS(singleexphdu,fobj=fobj)
    
    root = im.split('/')[-1]
    target_dir = '/'.join(im.split('/')[:-1])+'/'
    target_dir = target_dir +root[:-5]+'_regrid.fits'

    print("Single exposure read")
    array, interp = reproject_interp((singleexphdu.data, wcs), ref_wcs, shape_out=singleexphdu.data.shape)
    print("Reprojection done")
    return array





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











DO_RMSMAPS=True
if DO_RMSMAPS:
    os.system('cp ./SEXfiles/* ./working_dir/')
    os.chdir('./working_dir')
    ims = glob('../FLT_exposures/*/*/*_pamcorr_exptime.fits')
    for im in ims:
        root = im.split('/')[-1][:-5]
        hdul = fits.open(im)
        hdul[1].data = hdul[1].data / hdul[0].header['EXPTIME']
        fitsfile = root.replace('exptime', 'rate')+'.fits'
        hdul.writeto(fitsfile, overwrite=True)
        rmsmapfile = fitsfile.replace('rate', 'rmsmap')
        sex_command = 'sex ' + fitsfile +' -c params.sex -DETECT_THRESH 7 -CHECKIMAGE_TYPE BACKGROUND_RMS -CHECKIMAGE_NAME '+rmsmapfile
        os.system(sex_command) 
        rmstargetdir = '/'.join(im.split('/')[:-1])+'/'+rmsmapfile
        print(rmstargetdir)
        copyfile(rmsmapfile, rmstargetdir)
        os.system('rm -rf *.cat *.fits')
os.chdir('../')







IRAF_parallel=True
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
            hdu        = fits.open(im)
            im_crmasked = im[:-5]+'_crmasked.fits'
            im_crmask  = ''.join(im[:-5].split('_pamcorr')[:-1])+'_crmask.fits'
#            print('Working on file {} from {}'.format(f_count, len(drizzled_astrom_regrid_flist)), end='\r')

            # Do CR cleaning
            if   'wfc1' in im:
                this_wfc_ims = glob(folder+'*wfc1_pamcorr_exptime.fits') 
            elif 'wfc2' in im:    
                this_wfc_ims = glob(folder+'*wfc2_pamcorr_exptime.fits') 
            if not os.path.exists(im_crmask):
                hdu[1].data, sigma, CRmask = GetCRMasked_exptime(this_wfc_ims, im, folder, hdu[0].header['EXPTIME'])
                hdu.writeto(im_crmasked, overwrite=True)
                CR_hdu = hdu.copy()
                CR_hdu[1].data = CRmask.astype(float)
                CR_hdu.writeto(im_crmask, overwrite=True)

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
            write=True
            if write:
                target_dir = (im[:-5]+'.phot').replace('FLT_exposures', 'IRAF_cats')
                centroid_alg = 'none'
                coordfile = im.split('_pamcorr')[0]+'.coordfile'
                # Write task per image to do aperture photometry in IRAF
                if not os.path.exists(target_dir):
                    iraf_script_images.write('digiphot.apphot.phot image='+im_crmasked+'[1] ')
                    iraf_script_images.write('coords='+coordfile+' output='+target_dir + ' ')
                    iraf_script_images.write('salgori=mode annulus=6 dannulus=3 apertur=5 zmag='+str(zmag) + ' interac=no verify=no ')
                    iraf_script_images.write('calgori='+centroid_alg +' cbox=3 datamin=0 datamax=INDEF ')
                    iraf_script_images.write('gain=CCDGAIN readnoi=3.05 sigma='+str(1) + ' itime='+str(hdu[0].header['EXPTIME']))
                    iraf_script_images.write(5*'\n')
            del hdu
        iraf_script_images.close()
    os.chdir('../')





if DO_GET_NBadPIX:
    """Function to get the number of masked pixels inside the annulus IRAF used to do the photometry"""
    # For each file, read in the IRAF catalogue, use the (x,y) coordinates and write those to a new iraf command file
    os.chdir('./working_dir')
    ims = glob('../FLT_exposures/*/*/*drz_sci_regrid.fits')
    if IRAF_parallel:
        nsplits=6
    else:
        nsplits=1
    for which_chunk, drizzled_astrom_regrid_flist in enumerate(np.array_split(ims,nsplits)):
        iraf_script_nbadpix = open('app_phot_script_nbadpix'+str(which_chunk+1)+'.cl', 'w') 
        for f_count, im in enumerate(drizzled_astrom_regrid_flist): 
            folder = '/'.join(im.split('/')[:-1])+'/'
            print(im)
            splitted_dir = im.split('/')
            hst_filter   = splitted_dir[2]
            im_crmask  = im[:-5]+'_crmask.fits'
            if (hst_filter == 'F110W') or (hst_filter == 'F160W'):
                continue

            iraf_cat_dir = (im[:-5]+'.phot').replace('DRIZZLED', 'IRAF_cats_FLT')
                    coordlist_dir = iraf_cat_dir.replace('.phot', '_coords.coo')
            else:
                if pms_stars:
                    iraf_cat_dir = (im[:-5]+'_pmsstars.phot').replace('DRIZZLED', 'IRAF_cats')
                    coordlist_dir = iraf_cat_dir.replace('.phot', '_pmscoords.coo')
                else:
                    iraf_cat_dir = (im[:-5]+'.phot').replace('DRIZZLED', 'IRAF_cats')
                    coordlist_dir = iraf_cat_dir.replace('.phot', '_coords.coo')    

            if not os.path.exists(coordlist_dir):
                photfile = ascii.read(iraf_cat_dir).to_pandas().set_index('ID')
                xycoords = photfile[['XCENTER', 'YCENTER']]
                xycoords.to_csv(coordlist_dir, sep='\t', index=False, header=False)
            
            target_dir = (im[:-5]+'_pmsstars_nbadpix.phot ').replace('DRIZZLED', 'IRAF_cats_drz')    
            else:
                if not pms_stars:
                    target_dir = (im[:-5]+'_nbadpix.phot ').replace('DRIZZLED', 'IRAF_cats')
                else:
                    target_dir = (im[:-5]+'_pmsstars_nbadpix.phot ').replace('DRIZZLED', 'IRAF_cats')

            im_exptime = im[:-5]+'_exptime.fits'
            iraf_script_nbadpix.write('digiphot.apphot.phot image='+im_crmask+' ')
            iraf_script_nbadpix.write('coords= '+coordlist_dir+' output='+target_dir)
            iraf_script_nbadpix.write('salgori=constant skyvalu=0 apertur=3 zmag=99 interac=no verify=no ')
            iraf_script_nbadpix.write('calgori=none datamin=0 datamax=INDEF ')
            iraf_script_nbadpix.write('gain=CCDGAIN readnoi=3.05 sigma=1 itime=1')
            iraf_script_nbadpix.write(5*'\n')
        iraf_script_nbadpix.close()




















DO_IRAF_DF=True
DO_IRAF_NCR_DF=False
if DO_IRAF_DF or DO_IRAF_NCR_DF: 
    """Concatenate all IRAF photometry catalogues into one CSV"""
    """First do this for the photometry catalogues"""
    # Get the filelists for all catalogs
    #if not pms_stars:
    #    photometry_files  = glob('./IRAF_cats/*/*/*regrid.phot')
    #    nbadpix_files = glob('./IRAF_cats/*/*/*regrid_nbadpix.phot')
    #    output_fnames = ['APP_phot_all_exps.csv', 'NBadpix_all_exps.csv']
    #elif pms_stars:
    #    photometry_files  = glob('./IRAF_cats/*/*/*regrid_pmsstars.phot')
    #    nbadpix_files = glob('./IRAF_cats/*/*/*pmsstars_nbadpix.phot')
    #    output_fnames = ['APP_phot_all_exps_pmsstars.csv', 'NBadpix_all_exps_pmsstars.csv']
    os.chdir('./working_dir')
    photometry_files = glob('../IRAF_cats_FLT/*/*/*.phot')
    #if DO_IRAF_DF and DO_IRAF_NCR_DF:
    #    todo = [photometry_files,nbadpix_files]
    #elif DO_IRAF_DF and not DO_IRAF_NCR_DF:
    #    todo = [photometry_files]
    #elif DO_IRAF_NCR_DF and not DO_IRAF_DF:
    #    todo = [nbadpix_files]
    todo = [photometry_files]    
    for apphot_files in todo:
        if apphot_files == photometry_files:
            which = 0
        elif apphot_files == nbadpix_files:
            which = 1
        # See what columns we need to store
        columns = ascii.read(apphot_files[0]).to_pandas().columns
        # Create DF with 5 indexes
        IRAF_df = pd.DataFrame({'ID':[], 'Filter':[], 'T_Start':[],'Exp_Length':[], 'DrizzleType':[]})
        IRAF_df = IRAF_df.set_index(['ID', 'Filter', 'T_Start', 'Exp_Length', 'DrizzleType'])
        # Add the columns that we fill later
        for col in [w for w in columns if w is not 'ID']:
            IRAF_df[col] = []
        # Loop over the photometry files
        for f_count, apphot_file in enumerate(apphot_files):
            splitted_dir = apphot_file.split('/')
            # Extract relevant info for the indexing
            # Due to structure of directories, most info is in the filepath
            drizzle_type = 'SingleDrizzle'#('SingleDrizzle' if splitted_dir[1] == 'IRAF_cats' else 'MultiDrizzle')
            exp_length   = splitted_dir[3]
            hst_filter   = splitted_dir[2]
            if drizzle_type=='SingleDrizzle':
                fname        = splitted_dir[4].split('.')[0]
                # Lookup associated FITS file
                hdul         = fits.open(glob('../FLT_exposures/*/*/*'+fname+'.fits')[0])
                # Extract info from its header
                t_start      = hdul[0].header['EXPSTART']
                #obs_date     = hdul[0].header['DATE-OBS']
            else:
                t_start      = 'None'
                obs_date     = 'None'
    #        print("Filter "  , hst_filter)
    #        print("Exposure ", fname)
            print('Working on file {} of {}'.format(f_count, len(apphot_files)), end='\r')
            # Read in the photometry file
            phot_df = ascii.read(apphot_file).to_pandas()
            # Add info to df
            for col, val in zip(['Filter', 'T_Start', 'Exp_Length', 'DrizzleType'], [hst_filter, t_start, exp_length, drizzle_type]):
                phot_df[col] = val
            phot_df = phot_df.set_index(['ID', 'Filter', 'T_Start', 'Exp_Length', 'DrizzleType'])
            IRAF_df = pd.concat((IRAF_df, phot_df), axis=0)
            IRAF_df = IRAF_df[IRAF_df.CERROR!='OffImage']
        IRAF_df.sort_index().to_csv(output_fnames[which])
IRAF_df.to_csv('subset_flt_photometry.csv') 















    

