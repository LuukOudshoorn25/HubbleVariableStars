from glob import glob
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table




all_stars = Table.read('../../../HST_Guido/30dor_all_newerr.UBVIHa.rot', format='ascii').to_pandas()
all_stars.columns = 'ID;x;y;RA;Dec;u_1;eu_2;b_1;eb_2;v_1;ev_2;i_1;ei_2;ha_1;eha_2'.split(';')
all_stars = all_stars.set_index('ID')

deltaX  = all_stars.x.values - all_stars.x.values[:,np.newaxis]
deltaY  = all_stars.y.values - all_stars.y.values[:,np.newaxis]
deltaPix = np.array(np.sqrt(deltaX**2+deltaY**2), dtype=np.float32)
# throw away mergers
deltaPix = pd.DataFrame(deltaPix, index=np.arange(1,len(deltaPix)+1,1), columns = np.arange(1,len(deltaPix)+1,1))

deltaPix[deltaPix==0] = 999
mindists = deltaPix.min(axis=1)
to_keep = mindists.index[mindists>=8]



flist = glob('j*phot')
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
        
concatdf = concatdf.sort_index()




concatdf = concatdf.loc[to_keep]
concatdf = concatdf.drop_duplicates()

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
concatdf = concatdf[(concatdf.XCENTER>210)&(concatdf.XCENTER<3950)]
concatdf = concatdf[(concatdf.YCENTER>500)&(concatdf.YCENTER<4400)]
concatdf = concatdf[(concatdf.MAG>14.2)&(concatdf.MAG<26)] 
Nmeas = concatdf.reset_index().groupby('ID')['IMAGE'].nunique()
concatdf = concatdf.loc[Nmeas.index[Nmeas>=2].values].sort_index()


concatdf.to_pickle('ACS_photometry_df.pickle')

variations_df = concatdf.groupby('ID')['MAG'].max() - concatdf.groupby('ID')['MAG'].min()
variations_df = pd.DataFrame({'MaxminMin':variations_df})
variations_df = (pd.merge(variations_df, concatdf.groupby('ID')['MERR'].median(), 
                 left_index=True, right_index=True))
variations_df['MaxminMin_sigma'] = variations_df.MaxminMin / variations_df.MERR


variations_df.to_pickle('ACS_variations.pickle')




