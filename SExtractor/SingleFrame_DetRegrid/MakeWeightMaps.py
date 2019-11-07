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
    offsets = np.abs(im_hdul[1].data-med_hdul[1].data)/np.abs(med_hdul[1].data)
    if im_hdul[0].header['EXPTIME']>31:
        GoodPixels = (offsets<4).astype(int).astype(type(im_hdul[1].data[10,10]))
    elif im_hdul[0].header['EXPTIME']<31:
        GoodPixels = (offsets<9).astype(int).astype(type(im_hdul[1].data[10,10]))
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

    FlagMap = fits.ImageHDU(data=np.array((1-GoodPixels)).astype(np.int16))
    FlagMap.writeto(im.replace('rate', 'flag'), overwrite=True)

Parallel(n_jobs=7)(delayed(worker)(i) for i in images)
