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


scamp_ex = 'scamp '
### PARAMS ###
SORT           = False
CREATE_TREE    = False
DRIZZLE_IMS    = False
DRIZZLE_4IMS   = False
DO_ASTROMETRY  = False
DO_ASTROMETRY4 = False
DO_REFERENCE   = False
DO_REGRID      = False
DO_REGRID_PAR  = False
DO_REGRID4     = False	
DO_REGRID4_PAR = False
DO_APPHOT      = False
pms_stars      = True
recenter       = True
IRAF_parallel  = True
DO_GET_NBadPIX = False
DO_APPHOT4     = False
write_DS9_reg  = False
DO_IRAF_DF     = True
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
        path = "./DRIZZLED/"+str(unique_filter)+'/deep'
        if not os.path.exists(path):
            os.makedirs(path)
        path = "./DRIZZLED/"+unique_filter+'/short'
        if not os.path.exists(path):
            os.makedirs(path)

def create_dir_tree_IRAF():
    filterlist=[]
    flist = glob('./DRIZZLED/*/*/*fl*fits')
    for enum, fitsfile in enumerate(flist):
        print("Going through the fits to see unique filters, now at ", enum, " of ", len(flist), end='\r')
        hdul = fits.open(fitsfile)
        hst_filter = hdul[0].header['FILTER']
        filterlist.append(hst_filter)
    print(5*'\n')
    unique_filters = np.unique(filterlist)

    for unique_filter in unique_filters:
        path = "./IRAF_cats/"+str(unique_filter)+'/deep'
        if not os.path.exists(path):
            os.makedirs(path)
        path = "./IRAF_cats/"+unique_filter+'/short'
        if not os.path.exists(path):
            os.makedirs(path)


def drizzard():
    tweakreg.TweakReg('*_flt.fits',
                      enforce_user_order=False,
                      imagefindcfg={'threshold': 50, 'conv_width': 3.5, 'dqbits': ~4096, 'use_sharp_round':True},
                      refimage='../template_newheader.fits',
                      refimagefindcfg={'threshold': 30, 'conv_width': 2.5},
                      shiftfile=True,
                      outshifts='shiftfile.txt',
                      searchrad=500.0,
                      ylimit=0.6,
                      updatehdr=True,
                      wcsname='UVIS_FLC',
                      reusename=True,
                      interactive=False)

    astrodrizzle.AstroDrizzle('*_flt.fits',
                              output='_drz.fits',
                              preserve=False,
                              overwrite=True,
                              clean=False,
                              build=False,
                              context=False,
                              resetbits=0,
                              driz_separate=True,
                              median=False,
                              blot=False,
                              driz_cr=False,
                              final_wcs=True,
                              skysub=False)

def drizzard_multiple():
    tweakreg.TweakReg('*_flt.fits',
                      enforce_user_order=False,
                      imagefindcfg={'threshold': 50, 'conv_width': 3.5, 'dqbits': ~4096, 'use_sharp_round':True},
                      refimage='../template_newheader.fits',
                      refimagefindcfg={'threshold': 30, 'conv_width': 2.5},
                      shiftfile=True,
                      outshifts='shiftfile.txt',
                      searchrad=750.0,
                      ylimit=0.6,
                      updatehdr=True,
                      wcsname='UVIS_FLT',
                      reusename=True,
                      interactive=False)

    astrodrizzle.AstroDrizzle('*flt*',
            output='drz.fits',
            preserve=False,
            overwrite=True,
            clean=False,
            build=False,
            context=False,
            resetbits=0,
            driz_separate=False,
            median=False,
            blot=False,
            driz_cr=False,
            final_wcs=True,
            skysub=False)
    os.chdir('../')


def sex_scamp(root):
    fname       = root[:-5]
    catname     = fname + '.sex.cat'
    os.chdir('./working_dir')
    sex_command = 'sex ' + root + ' -c params.sex -CATALOG_NAME ' + catname    
    os.system(sex_command)
    scamp_command = scamp_ex+'-c default.scamp -CHECKPLOT_TYPE FGROUPS -CHECKPLOT_NAME '+fname+ ' '+ catname 
    os.system(scamp_command)

def update_header(root, reffits=False):
    fname       = root[:-5]
    catname     = fname + '.sex.cat'
    with open(fname+'.sex.head', 'r') as file :
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('é', 'e')
    # Write the file out again
    with open(fname+'.sex.head', 'w') as file:
        file.write(filedata)

    science_hdul = fits.open(root, ignore_missing_end=True)
    newheader = fits.Header.fromfile(fname+'.sex.head', endcard=False,
                padding=False, sep='\n')
    to_replace = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
    if not reffits:
        for key in to_replace:
            science_hdul[0].header[key] = newheader[key]
    elif reffits:
        for key in to_replace:
            science_hdul[0].header['O'+key] = newheader[key]
    os.system('rm -rf '+fname+'_newheader.fits')
    science_hdul.writeto(fname+'_newheader.fits')#, output_verify='ignore')

def reference_scamp(reffile, reffits=True):
    to_remove = './working_dir/*newheader* ./working_dir/*sex.cat* ./working_dir/*drz_sci*'
    os.system('rm -rf '+ to_remove)
    root = 'template.fits'
    fname       = root[:-5]
    catname     = fname + '.sex.cat'
    copyfile(reffile, './working_dir/'+root)
    os.chdir('./working_dir')
    sex_command = 'sex ' + root + ' -c params.sex -CATALOG_NAME ' + catname    
    os.system(sex_command)
    scamp_command = scamp_ex+'-c default.scamp -CHECKPLOT_TYPE FGROUPS -CHECKPLOT_NAME '+fname+ ' '+ catname 
    os.system(scamp_command)
    update_header(root, reffits)
    os.system('cp template_newheader.fits ../')


def write_ds9_regions(coordfile ='all_stars_xy_coords.coo'):
    all_stars = Table.read(coordfile, format='ascii').to_pandas()
    all_stars.columns = ['X', 'Y']
    f = open('./working_dir/iraf_regions.reg', 'w')
    for i in range(len(all_stars)):
        f.write('circle('+str(all_stars['X'][i]) + ',' + str(all_stars['Y'][i])+',3)\n')
        f.write('circle('+str(all_stars['X'][i]) + ',' + str(all_stars['Y'][i])+',4)\n')
        f.write('circle('+str(all_stars['X'][i]) + ',' + str(all_stars['Y'][i])+',7)\n')
    f.close()   

def regrid_worker(im):
        start_time = time.time()
        refhdu = fits.open('template_newheader.fits')[0]
        root = im.split('/')[-1]
        target_dir = '/'.join(im.split('/')[:-1])+'/'
        target_dir = target_dir +root[:-5]+'_regrid.fits'
        if os.path.exists(target_dir):
            return
        singleexphdu = fits.open(im)[0]
        print("Single exposure read")
        array, interp = reproject_exact(singleexphdu, refhdu.header, parallel=False)
        print("Reprojection done")
        new_header = singleexphdu.header.copy()
        to_replace = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PA_APER', 'RA_APER', 'DEC_APER']
        for key in to_replace:
            new_header[key] = refhdu.header[key]
        fits.writeto(target_dir, array, new_header, overwrite=True)
        print("Write done")
        print("Took ", (time.time()-start_time), " sec")

def GetCRMasked_exptime(flist, this_file, folderpath, exptime):
    storepath_median = folderpath + 'median.fits'
    if os.path.exists(storepath_median):
        medians       = fits.open(storepath_median)[0].data
    if os.path.exists('../stddevs.txt'):
        stddevs_df = pd.read_csv('../stddevs.txt', delimiter='\t')#.iloc[:88]
        stddevs_df.columns = ['folder', 'stddev']
        stddev = float(stddevs_df.groupby('folder').median().loc[folderpath])
    else:
        print('Could not find stddevs.txt, now calculating')
        # Load all data    
        flist = flist[:14]
        all_frames = np.zeros((*this_file.shape, len(flist)))
        for i,file in enumerate(flist):
            all_frames[:,:,i] = fits.open(file)[0].data
        medians = np.nanmedian(all_frames, axis=2)  
        del all_frames      
    # Get Background STDDEV
    if not os.path.exists('../stddevs.txt'):
        regA = this_file[this_file.shape[0]-2555:this_file.shape[0]-2525,this_file.shape[1]-2607:this_file.shape[1]-2570]
        stddev1 = np.nanstd(regA[regA<np.nanmedian(regA)])
        regB = this_file[this_file.shape[0]-965:this_file.shape[0]-923,this_file.shape[1]-1271:this_file.shape[1]-1225]
        stddev2 = np.nanstd(regB[regB<np.nanmedian(regB)])
        regC = this_file[this_file.shape[0]-757:this_file.shape[0]-729,this_file.shape[1]-1860:this_file.shape[1]-1812]
        stddev3 = np.nanstd(regC[regC<np.nanmedian(regC)])
        regD = this_file[this_file.shape[0]-507:this_file.shape[0]-445,this_file.shape[1]-3964:this_file.shape[1]-3883]
        stddev4 = np.nanstd(regD[regD<np.nanmedian(regD)])

        stddev = np.nanmedian(np.sort(np.array([stddev1, stddev2, stddev3, stddev4]))[:-1])
    print("Bg stddev: ",stddev)
    bg_stddev_arr = stddev * medians / np.nanmedian(this_file)
    if not os.path.exists(storepath_median):
        fits.writeto(storepath_median, medians, overwrite=True)
    #fits.writeto(storepath_stddev, bg_stddev_arr, overwrite=True)
    offsets = np.abs(this_file - medians)
    CRmask = np.where(exptime*offsets>9*bg_stddev_arr, True, False)
    this_file_corr = this_file.copy()
    this_file_corr *= exptime
    this_file_corr[CRmask] = -1e10
    with open('../stddevs.txt', 'a+') as stddev_list:
        stddev_list.write(folderpath + '\t'+str(stddev)+'\n')

    return this_file_corr, stddev, CRmask
### SORT ALL FILES ACCORDING TO FILTER AND EXPOSURE LENGTH ###
if SORT:
    sort_files()
    
    
### INITIALIZE: Clear working dir and copy SEX files to there ###
path = "./working_dir/"
try:
    os.mkdir(path)
except OSError:
    if not os.path.exists(path):
        os.paprint ("Creation of the directory %s failed" % path)
else:
    if verbose:
        print ("Successfully created the directory %s " % path)

initialize()

### CREATE FILETREES ###
if CREATE_TREE:
    create_dir_tree()

### START DRIZZLING ###
if DRIZZLE_IMS:
    folderlist = np.sort(glob('./SORTED/*/*/'))
    for folder in folderlist:
        print('Starting with ', folder)
        ims = glob(folder+'*flt.fits')
        if len(glob(folder.replace('SORTED', 'DRIZZLED')+'/*_flt_drz_sci*')) == len(ims):
            continue
        for iter_, im in enumerate(ims):
            root = im.split('/')[-1]
            copyfile(im, './working_dir/'+root)
            print("Copied {} out of {}".format(iter_, len(ims)), end='\r')
        os.chdir('./working_dir')
        print("Moved into working dir")
        drizzard()
        for im in ims:
            root = im.split('/')[-1]
            result = root[:-8]+'single_sci.fits'
            drizzfile = root[:-5]+'_drz_sci.fits'
            print(result)
            copyfile(result, '.'+folder.replace('SORTED', 'DRIZZLED')+drizzfile)
        os.chdir('../')
        os.system('rm -rf ./working_dir/*') 

        

### STACK 4 FRAMES ONTO EACH OTHER TO GET MULTIPLE DRIZZLES ###
if DRIZZLE_4IMS:
    folderlist = glob('./SORTED/*/*/')
    for folder in folderlist:
        ims = glob(folder+'*flt.fits')
        num_ims = len(ims)
        # Cant drizzle with less than three images
        if num_ims <3:
            continue
        if num_ims == 4:
            os.system('rm -rf ./working_dir/*')
            for im in ims:
                root = im.split('/')[-1]
                copyfile(im, './working_dir/'+root)
            os.chdir('./working_dir/')
            drizzard_multiple()
            target_dir = '/'.join(im.split('/')[:-1])
            target_dir = target_dir.replace('SORTED', 'MultiDrizzle')
            mkdir(target_dir)
            target_dir +='/drz_sci.fits'
            copyfile('./working_dir/drz_sci.fits', target_dir)
        elif num_ims > 4:
#            n_drizzles = num_ims//4
#            for n_driz in range(n_drizzles):
            for i,n_driz in enumerate(np.arange(0, len(ims), 4)):
                to_run = ims[n_driz:n_driz+4]
                os.system('rm -rf ./working_dir/*')
                for im in to_run:
                    root = im.split('/')[-1]
                    copyfile(im, './working_dir/'+root)
                os.chdir('./working_dir/')
                drizzard_multiple()  
                target_dir = '/'.join(im.split('/')[:-1])
                target_dir = target_dir.replace('SORTED', 'MultiDrizzle')
                mkdir(target_dir)
                target_dir +='/drz_sci_'+str(i)+'.fits'
                copyfile('./working_dir/drz_sci.fits', target_dir)
### REGRID ALL FILES SUCH THAT (X,Y) COORDINATES IN THE IMAGES ALIGN WITH GUIDOS LIST ###

drizzled_astrom_flist = glob('./DRIZZLED/*/*/*_drz_sci.fits')
drizzled_astrom_regrid_flist = glob('./DRIZZLED/*/*/*_drz_sci_regrid.fits')

if DO_REGRID:
    regrid_t_start = time.time()
    regrid_n_done = 0
    N_todo = len(drizzled_astrom_flist) - len(drizzled_astrom_regrid_flist)
    refhdu = fits.open('../template_newheader.fits')[0]
    regrid_n_done = 0
    for im in drizzled_astrom_flist:
        root = im.split('/')[-1]
        target_dir = '/'.join(im.split('/')[:-1])+'/'
        target_dir = target_dir +root[:-5]+'_regrid.fits'
        if os.path.exists(target_dir):
            continue
        singleexphdu = fits.open(im)[0]
        print("Single exposure read")
        print("Starting with ", root)
        new_header = singleexphdu.header.copy() 
        to_replace = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PA_APER', 'RA_APER', 'DEC_APER']
        for key in to_replace:
            new_header[key] = refhdu.header[key]
        array, footprint = reproject_exact(singleexphdu, refhdu.header, parallel=True)
        print("Reprojection done")
        fits.writeto(target_dir, array, new_header, overwrite=True)
        print("Write done")
        regrid_n_done +=1
        avg_time = (time.time() - regrid_t_start)/regrid_n_done
        print('\n\n\nAverage time per image ', 
              np.round(avg_time), '\nRemaining time ', (N_todo-regrid_n_done)*np.round(avg_time/60,1), " min\n\n")
        os.system('rm -rf ./working_dir/*regrid*fits')

if DO_REGRID_PAR:
    drizzled_astrom_flist = glob('./DRIZZLED/*/*/*_drz_sci.fits')
    drizzled_astrom_regrid_flist = glob('./DRIZZLED/*/*/*_drz_sci_regrid.fits')
    done_arr = [[w[:-5] not in [q[:-12] for q in drizzled_astrom_regrid_flist] for w in drizzled_astrom_flist]]
    drizzled_astrom_flist = [drizzled_astrom_flist[w] for w in np.where(done_arr)[1]]
    regrid_t_start = time.time()
    regrid_n_done = 0
    N_todo = len(drizzled_astrom_flist)
    print("To do: ", N_todo)
    num_cores = 4
    results = Parallel(n_jobs=num_cores)(delayed(regrid_worker)(i) for i in drizzled_astrom_flist)





### REGRID ALL FILES SUCH THAT (X,Y) COORDINATES IN THE IMAGES ALIGN WITH GUIDOS LIST ###
if DO_REGRID4:
    drizzled_astrom_flist = glob('./MultiDrizzle/*/*/*_drz_sci_astrometry.fits')
    drizzled_astrom_regrid_flist = glob('./MultiDrizzle/*/*/*_drz_sci_astrometry_regrid.fits')
    done_arr = [[w[:-5] not in [q[:-12] for q in drizzled_astrom_regrid_flist] for w in drizzled_astrom_flist]]
    drizzled_astrom_flist = [drizzled_astrom_flist[w] for w in np.where(done_arr)[1]]
    regrid_t_start = time.time()
    regrid_n_done = 0
    N_todo = len(drizzled_astrom_flist) - len(drizzled_astrom_regrid_flist)
    refhdu = fits.open('template.fits')[0]
    regrid_n_done = 0
    for im in drizzled_astrom_flist:
        print(im)
        root = im.split('/')[-1]
        target_dir = '/'.join(im.split('/')[:-1])+'/'
        target_dir = target_dir +root[:-5]+'_regrid.fits'
        if os.path.exists(target_dir):
            continue
        singleexphdu = fits.open(im)[0]
        print("Single exposure read")
        print("Starting with ", root)
        array, interp = reproject_exact(singleexphdu, refhdu.header, parallel=True)
        print("Reprojection done")
        new_header = singleexphdu.header.copy()
        to_replace = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PA_APER', 'RA_APER', 'DEC_APER']
        for key in to_replace:
            new_header[key] = refhdu.header[key]
        fits.writeto(target_dir, array, new_header, overwrite=True)
        print("Write done")
        regrid_n_done +=1
        avg_time = (time.time() - regrid_t_start)/regrid_n_done
        print('\n\n\nAverage time per image ', 
              np.round(avg_time), '\nRemaining time ', (N_todo-regrid_n_done)*np.round(avg_time/60,1), " min\n\n")
        os.system('rm -rf ./working_dir/*regrid*fits')



if DO_REGRID4_PAR:
    drizzled_astrom_flist = glob('./MultiDrizzle/*/*/*_drz_sci.fits')
    drizzled_astrom_regrid_flist = glob('./MultiDrizzle/*/*/*_drz_sci_regrid.fits')
    drizzled_astrom_flist = drizzled_astrom_flist[[w[:-16] not in [q[:-12] for q in drizzled_astrom_regrid_flist] for w in drizzled_astrom_flist]]
    regrid_t_start = time.time()
    regrid_n_done = 0
    N_todo = len(drizzled_astrom_flist) - len(drizzled_astrom_regrid_flist)
    num_cores = 8

    results = Parallel(n_jobs=num_cores)(delayed(regrid_worker)(i) for i in drizzled_astrom_flist)

### DO SEXTRACTOR AND SCAMP CALIBRATION FOR WCS HEADERS ###
drizzled_flist = glob('./DRIZZLED/*/*/*_drz_sci_regrid.fits')
drizzled_astrom_flist = glob('./DRIZZLED/*/*/*_drz_sci_regrid_astrometry.fits')
if DO_ASTROMETRY:
    initialize()
    astrom_t_start = time.time()
    astrom_n_done = 0
    N_todo = len(drizzled_flist) - len(drizzled_astrom_flist)
    for im in drizzled_flist:
        root        = im.split('/')[-1]
        target_dir = '/'.join(im.split('/')[:-1])+'/'
        target_dir = target_dir +root[:-5]+'_astrometry.fits'
        #hdul = fits.open(im, ignore_missing_end=True)
        #science_hdu = hdul[0]
        if os.path.exists(target_dir):
            continue
        to_remove = './working_dir/*newheader* ./working_dir/*sex.cat* ./working_dir/*drz_sci*'
        os.system('rm -rf '+ to_remove)
        copyfile(im, './working_dir/'+root)
        sex_scamp(root)
        update_header(root)
        # Copy the file with fixed astronomical header to the DRIZZLED folder
        fixed_header_fits = root[:-5]+'_newheader.fits'
        copyfile(fixed_header_fits, '.'+target_dir)
        os.chdir('../')
        astrom_n_done += 1 
        avg_time = (time.time() - astrom_t_start)/astrom_n_done
        print('\n\n\nAverage time per image ', 
              np.round(avg_time), '\nRemaining time ', (N_todo-astrom_n_done)*np.round(avg_time/60,1), " min\n\n")



### DO SEXTRACTOR AND SCAMP CALIBRATION FOR WCS HEADERS for the multidrizzles ###
drizzled_flist = glob('./MultiDrizzle/*/*/_drz_sci_regrid*fits')
drizzled_astrom_flist = glob('./MultiDrizzle/*/*/*_drz_sci*_drz_sci_regrid_astrometry.fits')
if DO_ASTROMETRY4:
    astrom_t_start = time.time()
    astrom_n_done = 0
    N_todo = len(drizzled_flist) - len(drizzled_astrom_flist)
    for im in drizzled_flist:
        root        = im.split('/')[-1]
        target_dir = '../'+'/'.join(im.split('/')[:-1])+'/'
        target_dir = target_dir +root[:-5]+'_astrometry.fits'
        #hdul = fits.open(im, ignore_missing_end=True)
        #science_hdu = hdul[0]
        if os.path.exists(target_dir):
            continue
        to_remove = './working_dir/*newheader* ./working_dir/*sex.cat* ./working_dir/*drz_sci*'
        os.system('rm -rf '+ to_remove)
        copyfile(im, './working_dir/'+root)
        sex_scamp(root)
        update_header(root)
        # Copy the file with fixed astronomical header to the DRIZZLED folder
        fixed_header_fits = root[:-5]+'_newheader.fits'
        copyfile(fixed_header_fits, target_dir)
        os.chdir('../')
        astrom_n_done += 1 
        avg_time = (time.time() - astrom_t_start)/astrom_n_done
        print('\n\n\nAverage time per image ', 
              np.round(avg_time), '\nRemaining time ', (N_todo-astrom_n_done)*np.round(avg_time/60,1), " min\n\n")




### CALIBRATE THE REFERENCE IMAGE TO MATCH WCS AND TRUE SKY COORDINATES ###
if DO_REFERENCE:
    # Do SEXtractor and Scamp on the reference frame
    reffile='30dorf555drz.fits'
    reference_scamp(reffile, reffits=True)



### START APERTURE PHOTOMETRY USING IRAF ###




if DO_APPHOT:
    """Correct the FITS files by masking cosmic rays, multiply them by their exposure time 
       and write the CR masks to FITS files"""
    #create_dir_tree_IRAF()
    os.chdir('./working_dir')
    if DO_APPHOT4:
        drizzled_astrom_regrid_flist = glob('../MultiDrizzle/*/*/drz_sci*regrid.fits')
    else: 
        drizzled_astrom_regrid_flist = glob('../DRIZZLED/*/*/*drz_sci_regrid.fits')
    # We want to do aperture photometry on all these files
    # To keep things findable, we will use the same structure as before
    # Since we do not use PyRAF, we need to make a cl script and run that in IRAF using cmd

    # Check which files are already done
    
    if pms_stars:
        coordfile    = 'pms_stars_xy_coords.coo'
    else:
        coordfile    = 'all_stars_xy_coords.coo'
    if not recenter:
        coordfile = coordfile.replace('.coo', '_refluuk.coo')
    if not pms_stars:
        drizzled_apphot_flist        = glob('../IRAF_cat*/*/*/*regrid.phot')
        not_done_arr = ([(w[:-5]+'.phot').replace('DRIZZLED', 'IRAF_cats') not in 
                    drizzled_apphot_flist for w in drizzled_astrom_regrid_flist])
    elif pms_stars:
        drizzled_apphot_flist        = glob('../IRAF_cat*/*/*/*_pmsstars.phot')
        not_done_arr = ([(w[:-5]+'.phot').replace('DRIZZLED', 'IRAF_cats') not in 
                    drizzled_apphot_flist for w in drizzled_astrom_regrid_flist])

    drizzled_astrom_regrid_flist = [drizzled_astrom_regrid_flist[w] for w in np.where(not_done_arr)[0]]

    all_files = drizzled_astrom_regrid_flist.copy()
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
            if not 0>np.inf:#(os.path.exists(im_exptime) and os.path.exists(im_crmask)):
               # CR clean
                flist                    = glob(folder+'*drz_sci_regrid.fits')
                hdu.data, sigma, CRmask  = GetCRMasked_exptime(flist, hdu.data, folder, hdu.header['EXPTIME'])
                hdu.writeto(im_exptime, overwrite=True)
                # Save CRmask to FITS file
                CRmask = CRmask.astype(type(hdu.data[555,555]))
                CR_hdu = hdu.copy()
                CR_hdu.data = CRmask
                CR_hdu.writeto(im_crmask, overwrite=True)
            else:
                # If CRfile and CRcleaned file already exist, read stddevs from file
                stddevs_df = pd.read_csv('../stddevs.txt', delimiter='\t')
                stddevs_df.columns = ['folder', 'stddev']
                stddev = float(stddevs_df.groupby('folder').median().loc[folder])
            # Define magnitude zeropoints...
            zmag  = {'F336W':23.46,'F438W':24.98,'F555W': 25.81, 'F814W': 24.67, 'F656N': 19.92}[hst_filter]-0.1
            if DO_APPHOT4:
                if pms_stars:
                    target_dir = (im[:-5]+'_pmsstars.phot ').replace('DRIZZLED', 'IRAF_cats_drz')
                else:
                    target_dir = (im[:-5]+'.phot ').replace('DRIZZLED', 'IRAF_cats_drz')
            else:
                if pms_stars:
                    target_dir = (im[:-5]+'_pmsstars.phot ').replace('DRIZZLED', 'IRAF_cats')
                else:
                    target_dir = (im[:-5]+'.phot ').replace('DRIZZLED', 'IRAF_cats')
            centroid_alg = ('centroid' if recenter else 'none')

            # Write task per image to do aperture photometry in IRAF
            iraf_script_images.write('digiphot.apphot.phot image='+im_exptime+' ')
            iraf_script_images.write('coords='+coordfile+' output='+target_dir)
            iraf_script_images.write('salgori=mode annulus=6 dannulus=3 apertur=5 zmag='+str(zmag) + ' interac=no verify=no ')
            iraf_script_images.write('calgori='+centroid_alg +' cbox=3 datamin=0 datamax=INDEF ')
            iraf_script_images.write('gain=CCDGAIN readnoi=3.05 sigma='+str(sigma) + ' itime='+str(hdu.header['EXPTIME']))
            iraf_script_images.write(5*'\n')
            del hdu
        iraf_script_images.close()
    os.chdir('../')
        

if DO_GET_NBadPIX:
    """Function to get the number of masked pixels inside the annulus IRAF used to do the photometry"""
    # For each file, read in the IRAF catalogue, use the (x,y) coordinates and write those to a new iraf command file
    os.chdir('./working_dir')
    if DO_APPHOT4:
        drizzled_astrom_regrid_flist = glob('../MultiDrizzle/*/*/drz_sci*regrid.fits')
    else: 
        drizzled_astrom_regrid_flist = glob('../DRIZZLED/*/*/*drz_sci_regrid.fits')
    all_files = drizzled_astrom_regrid_flist.copy()
    if IRAF_parallel:
        nsplits=6
    else:
        nsplits=1
    for which_chunk, drizzled_astrom_regrid_flist in enumerate(np.array_split(all_files,nsplits)):
        iraf_script_nbadpix = open('app_phot_script_nbadpix'+str(which_chunk+1)+'.cl', 'w') 
        for f_count, im in enumerate(drizzled_astrom_regrid_flist): 
            folder = '/'.join(im.split('/')[:-1])+'/'
            print(im)
            splitted_dir = im.split('/')
            hst_filter   = splitted_dir[2]
            im_crmask  = im[:-5]+'_crmask.fits'
            if (hst_filter == 'F110W') or (hst_filter == 'F160W'):
                continue

            if DO_APPHOT4:
                if pms_stars:
                    iraf_cat_dir = (im[:-5]+'_pmsstars.phot').replace('DRIZZLED', 'IRAF_cats_drz')
                    coordlist_dir = iraf_cat_dir.replace('.phot', '_pmscoords.coo')
                else:
                    iraf_cat_dir = (im[:-5]+'.phot').replace('DRIZZLED', 'IRAF_cats_drz')
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
            if DO_APPHOT4:
                if not pms_stars:
                    target_dir = (im[:-5]+'_nbadpix.phot ').replace('DRIZZLED', 'IRAF_cats_drz')
                else: 
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



if write_DS9_reg:
    write_ds9_regions()


if DO_IRAF_DF or DO_IRAF_NCR_DF: 
    """Concatenate all IRAF photometry catalogues into one CSV"""
    """First do this for the photometry catalogues"""
    # Get the filelists for all catalogs
    if not pms_stars:
        photometry_files  = glob('./IRAF_cats/*/*/*regrid.phot')
        nbadpix_files = glob('./IRAF_cats/*/*/*regrid_nbadpix.phot')
        output_fnames = ['APP_phot_all_exps.csv', 'NBadpix_all_exps.csv']
    elif pms_stars:
        photometry_files  = glob('./IRAF_cats/*/*/*regrid_pmsstars.phot')
        nbadpix_files = glob('./IRAF_cats/*/*/*pmsstars_nbadpix.phot')
        output_fnames = ['APP_phot_all_exps_pmsstars.csv', 'NBadpix_all_exps_pmsstars.csv']
    if DO_IRAF_DF and DO_IRAF_NCR_DF:
        todo = [photometry_files,nbadpix_files]
    elif DO_IRAF_DF and not DO_IRAF_NCR_DF:
        todo = [photometry_files]
    elif DO_IRAF_NCR_DF and not DO_IRAF_DF:
        todo = [nbadpix_files]
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
            drizzle_type = ('SingleDrizzle' if splitted_dir[1] == 'IRAF_cats' else 'MultiDrizzle')
            exp_length   = splitted_dir[3]
            hst_filter   = splitted_dir[2]
            if drizzle_type=='SingleDrizzle':
                fname        = splitted_dir[4].split('_')[0]
                # Lookup associated FITS file
                hdul         = fits.open(glob('./DRIZZLED/*/*/*'+fname+'*')[0])
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
        IRAF_df.sort_index().to_csv(output_fnames[which])
        

print("Finished in ", time.time() - tstart)


    

