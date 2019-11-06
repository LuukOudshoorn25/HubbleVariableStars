import numpy as np
from astropy.io import fits
from glob import glob
import os



##################################
## Sex on regridded FLT images ###
##################################
zmag  = {'F336W':23.46,'F438W':24.98,'F555W': 25.81, 'F814W': 24.67, 'F656N': 19.92}

wfc=2
flist = glob('./wfc'+str(wfc)+'/ib*exptime_regrid*.fits')
force_rerun = True

for im in flist:
    detection_file = ('whitelight_wfc1.fits'if 'wfc1' in im else 'whitelight_wfc2.fits')
    catname = im.split('_flt')[0] + ('_wfc1' if 'wfc1' in im else '_wfc2') + '_phot.fits'
    assoc_file = ('ALL_WFC1.coordfile' if 'wfc1' in im else 'ALL_WFC2.coordfile')
    hdul = fits.open(im)
    exptime = hdul[0].header['EXPTIME']
    filter_ = hdul[0].header['Filter']
    mzp = zmag[filter_]
    # Divide by exptime
    hdul[1].data = hdul[1].data / exptime
    if not os.path.exists(im.replace('exptime', 'rate')):
        hdul.writeto(im.replace('exptime', 'rate'), overwrite=True)
    if os.path.exists(catname)and not force_rerun:
        continue
    sex_command = 'sex ' + detection_file + ','+im.replace('exptime', 'rate')
    sex_command += ' -c ./sexfiles/params.sex'
    sex_command += ' -PARAMETERS_NAME ./sexfiles/default.param'
    sex_command += ' -FILTER_NAME ./sexfiles/gauss_2.0_5x5.conv'
    sex_command += ' -MAG_ZEROPOINT ' + str(mzp)
    sex_command += ' -ASSOC_NAME ' + assoc_file
    sex_command += ' -CATALOG_NAME ' + catname
    os.system(sex_command)









############################################
## Sex on regridded and added FLT images ###
############################################

import numpy as np
from astropy.io import fits
from glob import glob
import os
from joblib import Parallel, delayed

zmag  = {'F336W':23.46,'F438W':24.98,'F555W': 25.81, 'F814W': 24.67, 'F656N': 19.92}


flist = glob('../Add_Regrid/*add_regrid*fits')
force_rerun = True
#os.system('rm -rf ../Add_Regrid/*phot*fits')

def run_sex(im):
    detection_file = 'whitelight_Guido.fits'
    catname = im.split('_add')[0] + '_phot.fits'
    assoc_file = 'XYposF814Wframe.coordfile'
    hdul = fits.open(im)
    #exptime = hdul[0].header['EXPTIME']
    filter_ = hdul[1].header['Filter']
    mzp = zmag[filter_]
    # Divide by exptime
    #hdul[1].data = hdul[1].data / exptime
    #if not os.path.exists(im.replace('exptime', 'rate')):
    #    hdul.writeto(im.replace('exptime', 'rate'), overwrite=True)
    if os.path.exists(catname)and not force_rerun:
        return
    sex_command = 'sex ' + detection_file + ','+im.replace('exptime', 'rate')
    sex_command += ' -c ./sexfiles/params.sex'
    sex_command += ' -PARAMETERS_NAME ./sexfiles/default.param'
    sex_command += ' -FILTER_NAME ./sexfiles/gauss_2.0_5x5.conv'
    sex_command += ' -MAG_ZEROPOINT ' + str(mzp)
    sex_command += ' -ASSOC_NAME ' + assoc_file
    sex_command += ' -CATALOG_NAME ' + catname
    os.system(sex_command)
Parallel(n_jobs=6)(delayed(run_sex)(i) for i in flist)

##############################################
## Sex on raw FLT and regridded whitelight ###
##############################################
import numpy as np
from astropy.io import fits
from glob import glob
import os
from joblib import Parallel, delayed
from astropy.io import ascii

zmag  = {'F336W':23.46,'F438W':24.98,'F555W': 25.81, 'F814W': 24.67, 'F656N': 19.92}

#wfc=2
photometry_frames = np.sort(glob('../SingleFrame_DetRegrid/WFC*/ib*exptime.fits'))
#detection_frames = np.sort(glob('../SingleFrame_DetRegrid/WFC'+str(wfc)+'/ib*pamcorr_median.fits'))
force_rerun = True


def run_sex(im):
    detection_file = im.replace('exptime', 'median')
    catname = im.split('_pamcorr')[0] + '_phot.fits'
    assoc_file = im.replace('_pamcorr_exptime.fits', '_all.coordfile')
    assoc_df = ascii.read(assoc_file).to_pandas()
    if len(assoc_df.columns)==2:
        assoc_df['ID'] = np.arange(1,len(assoc_df)+1)
        assoc_df[['ID', 'col1', 'col2']].to_csv(assoc_file, sep='\t', header=None, index=None)
    hdul = fits.open(im)
    exptime = hdul[0].header['EXPTIME']
    if not os.path.exists(im.replace('exptime', 'rate')):
        hdul[1].data = hdul[1].data / exptime
        hdul.writeto(im.replace('exptime', 'rate'), overwrite=True)
    filter_ = hdul[0].header['Filter']
    weight_map = im.replace('exptime', 'weight')
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
    sex_command += ' -WEIGHT_IMAGE ' + weight_map
    os.system(sex_command)
Parallel(n_jobs=8)(delayed(run_sex)(i) for i in photometry_frames)







"""

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
flist = glob('../F*/*/*wfc1*pamcorr_exptime.fits')
def return_reproject(params):
    image, refim = params
    wfchdu = fits.open(image)[1]
    fobj   =  fits.open(image.split('_wfc')[0]+'.fits')
    wcs    =  WCS(wfchdu,fobj=fobj)
    
    ref_hdu = fits.open(refim)[0]
    ref_wcs = WCS(ref_hdu)

    array, interp = reproject_exact((wfchdu.data, wcs), ref_wcs, shape_out=ref_hdu.data.shape, parallel=False)
    return array

refim='../../HST_Guido/30dorf814drz.fits'
t0=time.time()

for iter_,image in enumerate(flist):

    im_wfc1 = image
    im_wfc2 = image.replace('wfc1', 'wfc2')

    imname = './Add_Regrid/' + im_wfc1.split('_flt')[0].split('/')[-1] + '_add_regrid.fits'

    #if os.path.exists(imname):
    #    print(imname, 'done')
    #    continue
    inputs = [(im_wfc1, refim), (im_wfc2, refim)]

    num_cores = 2
    results = Parallel(n_jobs=num_cores)(delayed(return_reproject)(i) for i in inputs)

    final_image = np.zeros((*results[0].shape,2))
    final_image[:,:,0] = results[0]
    final_image[:,:,1] = results[1]
    final_image = np.nansum(final_image, axis=2) / fits.open(image)[0].header['EXPTIME']

    ref_hdu = fits.open(refim)[0]
    ref_wcs = WCS(ref_hdu.header)

    
    primary_hdu = fits.PrimaryHDU(data=None, header=fits.open(im_wfc1)[0].header)
    output_hdul = fits.HDUList([])
    
    output_hdul.append(ref_hdu.copy())
    output_hdul[0].data=final_image
    output_hdul[0].header['FILTER']='WARNING_USE_OTHER_EXTENSION'
    output_hdul.append(primary_hdu)
    output_hdul.writeto(imname, overwrite=True)
    print('ETA: ', (len(flist)-iter_-1) * ((time.time()-t0)/(iter_+1))/60)

"""
