from glob import glob
from astropy.io import fits
import numpy as np
from joblib import Parallel, delayed

images = glob('./WFC*/*pamcorr_rate.fits')

def worker(im):
    print(im)
    med_im = im.replace('rate', 'median')
    im_hdul = fits.open(im)
    med_hdul = fits.open(med_im)
    offsets = np.abs(im_hdul[1].data-med_hdul[1].data)/med_hdul[1].data
    GoodPixels = (offsets<6).astype(int).astype(type(im_hdul[1].data[10,10]))
    WeightMap = fits.ImageHDU(data=GoodPixels)
    WeightMap.writeto(im.replace('rate', 'weight'), overwrite=True)

Parallel(n_jobs=7)(delayed(worker)(i) for i in images)
