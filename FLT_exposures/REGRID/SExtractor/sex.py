import numpy as np
from astropy.io import fits
from glob import glob
import os


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




