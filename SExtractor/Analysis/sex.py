##############################################
## Sex on raw FLT and regridded whitelight ###
##############################################
import numpy as np
from astropy.io import fits
from glob import glob
import os
from joblib import Parallel, delayed
from astropy.io import ascii



"""

def get_xy_coords(filepath):
    outputfile = filepath.replace('_pamcorr_exptime.fits', '_all.coordfile')
    if os.path.exists(outputfile):
        print('return')
        return 
    ds9_command = "ds9 "+filepath+" -regions load ../../radec.reg -regions system image "
    ds9_command += '-regions format XY -regions save '+outputfile + ' -exit'
    os.system(ds9_command)

photometry_frames = np.sort(glob('../SingleFrame_DetRegrid/WFC*/ib*exptime.fits'))
Parallel(n_jobs=6)(delayed(get_xy_coords)(i) for i in photometry_frames)

"""



zmag  = {'F336W':23.46,'F438W':24.98,'F555W': 25.81, 'F814W': 24.67, 'F656N': 19.92}

photometry_frames = np.sort(glob('../SingleFrame_DetRegrid/WFC*/ib*exptime.fits'))
force_rerun = True

def run_sex(im):
    detection_file = im.replace('pamcorr_exptime', 'whitelight_regrid')
    catname = im.split('_pamcorr')[0] + '_phot.fits'
    assoc_file = im.replace('_pamcorr_exptime.fits', '_all.coordfile')
    assoc_df = ascii.read(assoc_file).to_pandas()
    if len(assoc_df.columns)==2:
        assoc_df['ID'] = np.arange(1,len(assoc_df)+1)
        assoc_df[['ID', 'col1', 'col2']].to_csv(assoc_file, sep='\t', header=None, index=None)
    hdul = fits.open(im)
    exptime = hdul[0].header['EXPTIME']
    if exptime<31:
        return
    if not os.path.exists(im.replace('exptime', 'rate')):
        hdul[1].data = hdul[1].data / exptime
        hdul.writeto(im.replace('exptime', 'rate'), overwrite=True)
    filter_ = hdul[0].header['Filter']
#    if filter_ !='F555W':
#        return
    weight_map = im.replace('exptime', 'weight')
    rms_map = im.replace('exptime', 'rms')
    #flag_map = im.replace('exptime', 'flag')

    mzp = zmag[filter_]
    # Divide by exptime
    #hdul[1].data = hdul[1].data / exptime
    #if not os.path.exists(im.replace('exptime', 'rate')):
    #    hdul.writeto(im.replace('exptime', 'rate'), overwrite=True)
    if os.path.exists(catname) and not force_rerun:
        return
    sex_command = 'sex ' + detection_file + ','+im.replace('exptime', 'rate')
    sex_command += ' -c ./sexfiles/params.sex'
    sex_command += ' -PARAMETERS_NAME ./sexfiles/default.param'
    sex_command += ' -FILTER_NAME ./sexfiles/gauss_2.0_5x5.conv'
    sex_command += ' -MAG_ZEROPOINT ' + str(mzp)
    sex_command += ' -ASSOC_NAME ' + assoc_file
    sex_command += ' -CATALOG_NAME ' + catname
    sex_command += ' -WEIGHT_IMAGE ' + weight_map+','+weight_map
    #sex_command += ' -CHECKIMAGE_NAME '+rms_map
    os.system(sex_command)

a=Parallel(n_jobs=6)(delayed(run_sex)(i) for i in photometry_frames)









#### REGRID #####

from reproject import reproject_interp, reproject_exact
import sys
from joblib import Parallel, delayed
from astropy.wcs import WCS
from astropy import wcs
import numpy as np
from astropy.io import fits
from glob import glob
import os
import time

def return_reproject(params):
    image, refim = params
    wfchdu = fits.open(image)[1]
    fobj_root = (image.split('_wfc')[0]).split('/')[-1]
    fobj_path = glob('../FLT_exposures/*/*/*'+fobj_root+'.fits')[0]
    fobj   =  fits.open(fobj_path)
    wcs    =  WCS(wfchdu,fobj=fobj)
    
    ref_hdu = fits.open(refim)[1]
    ref_wcs = WCS(ref_hdu, fobj=fobj)

    array, interp = reproject_interp((ref_hdu.data, ref_wcs), wcs, shape_out=wfchdu.data.shape)
    print('Done')
    return array

flist = glob('./SingleFrame_DetRegrid/WFC*/*pamcorr_rate.fits')
#### Regrid images to this frame ####
filters_to_regrid = ['F336W', 'F438W', 'F555W', 'F656N']
for f in flist[:1]:
    wfc = ('1' if 'wfc1' in f else '2')
    rootname = f.split('WFC'+wfc+'/')[1].split('_pamcorr')[0]
    temp_folder = ''.join(f.split('.fits')[:-1])
    outputfile = temp_folder+'/'+rootname+'_whitelight_regrid.fits'
    if os.path.exists(outputfile):
        continue
    # Get random number of frames to regrid to this frame
    reference_images = glob('./SingleFrame_DetRegrid/WFC'+wfc+'/*pamcorr_rate.fits')
    filters = [fits.open(w)[0].header['FILTER'] for w in reference_images]
    for filter_ in filters_to_regrid:
        nframes = 8
        to_use = np.where([w==filter_ for w in filters])[0]
        reference_images_use = np.random.choice([reference_images[w] for w in to_use],nframes)
        
        final_image = np.zeros((*fits.open(f)[1].data.shape,nframes))
        #for i,im in enumerate(reference_images_use[:1]):
        #
        #    regrid_arr = return_reproject((im,f))
        #    final_image[:,:,i] = regrid_arr
        inputs = [(f,w) for w in reference_images_use]
        returns = Parallel(n_jobs=nframes)(delayed(return_reproject)(i) for i in inputs)
        for i, data in enumerate(returns):
            final_image[:,:,i] = data
        medim_regrid = fits.open(f).copy()
        medim_regrid[1].data = np.nanmedian(final_image, axis=2)
        medim_regrid.writeto(temp_folder+'/'+filter_+'median.fits', overwrite=True)

    # Now add the four median images
    medims = glob(temp_folder+'/*.fits')
    whitelight_array = np.zeros(fits.open(medims[1])[1].data.shape)
    for im in medims:
        whitelight_array += fits.open(im)[1].data
    whitelight_hdu = fits.open(medims[0])
    whitelight_hdu[1].data = whitelight_array
    whitelight_hdu.writeto(outputfile, overwrite=True)

        

###########################################
flist = glob('./SingleFrame_DetRegrid/WFC2/*pamcorr_rate.fits')


refim='../FLT_exposures/REGRID/Whitelight_Images/wfc2/whitelight_wfc2.fits'
t0=time.time()

for iter_,image in enumerate(flist):
    if fits.open(image)[0].header['EXPTIME'] < 31:
        continue
    if os.path.exists(image.replace('_pamcorr_rate', '_whitelight_regrid')):
        continue
    inputs = (image, refim)


    result = return_reproject(inputs)#Parallel(n_jobs=num_cores)(delayed(return_reproject)(i) for i in inputs)
    whitelight_regrid = fits.open(image).copy()
    whitelight_regrid[1].data = result
    whitelight_regrid.writeto(image.replace('_pamcorr_rate', '_whitelight_regrid'), overwrite=True)

    #ref_hdu = fits.open(refim)[0]
    #ref_wcs = WCS(ref_hdu.header)

    
    #primary_hdu = fits.PrimaryHDU(data=None, header=fits.open(im_wfc1)[0].header)
    #output_hdul = fits.HDUList([])
    
    #output_hdul.append(ref_hdu.copy())
    #output_hdul[0].data=final_image
    #output_hdul[0].header['FILTER']='WARNING_USE_OTHER_EXTENSION'
    #output_hdul.append(primary_hdu)
    #output_hdul.writeto(imname, overwrite=True)
    print('ETA: ', (len(flist)-iter_-1) * ((time.time()-t0)/(iter_+1))/60)

"""
