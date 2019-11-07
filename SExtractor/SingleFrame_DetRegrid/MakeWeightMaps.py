from glob import glob
from astropy.io import fits
import numpy as np
from joblib import Parallel, delayed

images = glob('./WFC*/*pamcorr_rate.fits')
import time

time.sleep(6*60)
def worker(im):
    print(im)
    hdul = fits.open(im)
    exptime = hdul[0].header['EXPTIME']
    if exptime<31:
        return
    med_im = im.replace('rate', 'median')
    im_hdul = fits.open(im)
    med_hdul = fits.open(med_im)
    offsets = np.abs(im_hdul[1].data-med_hdul[1].data)
    rms_map = im.replace('rate', 'rms')
    rms_data = np.median(fits.open(rms_map)[0].data)*med_hdul[1].data/np.median(im_hdul[1].data)
    offsets_std = offsets / rms_data

    GoodPixels = (offsets_std<10).astype(int).astype(type(im_hdul[1].data[10,10]))
    WhereGoodPixels = np.where(GoodPixels==0)
    WhereGoodPixelsX = WhereGoodPixels[1]
    WhereGoodPixelsY = WhereGoodPixels[0]
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
#            if not WhereGoodPixelsY+dy<0 or WhereGoodPixelsY+dy<0 > GoodPixels.shape[0]:
#                if not WhereGoodPixelsX+dx<0 or WhereGoodPixelsX+dx<0 > GoodPixels.shape[1]:
            todoY = WhereGoodPixelsY+dy
            todoX = WhereGoodPixelsX+dx
            mask = (todoY<2050)&(todoX<4095)&(todoY>1)&(todoX>1)
            todoY = todoY[mask]
            todoX = todoX[mask]
            GoodPixels[todoY,todoX] = 0
    WeightMap = fits.ImageHDU(data=GoodPixels)
    WeightMap.writeto(im.replace('rate', 'weight'), overwrite=True)

    #FlagMap = fits.ImageHDU(data=np.array((1-GoodPixels)).astype(np.int16))
    #FlagMap.writeto(im.replace('rate', 'flag'), overwrite=True)

Parallel(n_jobs=6)(delayed(worker)(i) for i in images)
