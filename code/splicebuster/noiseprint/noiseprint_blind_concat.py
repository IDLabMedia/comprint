# This file was created by the Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA')
# and adapted by IDLab-MEDIA, Ghent University - imec, in collaboration with GRIP-UNINA

import numpy as np
from .post_em import EMgu_img, getSpamFromNoiseprint
from .utility.utilityRead import resizeMapWithPadding
from .utility.utilityRead import imread2f
from .utility.utilityRead import jpeg_qtableinv
from tensorflow import keras as ks
import tensorflow as tf
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt 
import os
 
from .noiseprint import genNoiseprint
    
MAX_INT32 = np.int64(2147483647)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def noiseprint_blind_file(filename, models, outfilenames=[]):
    try:
        img, mode = imread2f(filename, channel=1)
    except:
        print('Error opening image')
        return -1, -1, -1e10, None, None, None, None, None, None
    
    try:
        QF = jpeg_qtableinv(filename)
        print('QF=', QF)
    except:
        QF = 200
        
    # Tensorflow requires the shape multiplication to fit in int32
    # https://stackoverflow.com/questions/53067722/tensorflow-error-checked-narrowing-failed-values-not-equal-post-conversion-ab
    size = np.int64(img.shape[0]) * np.int64(img.shape[1]) * np.int64(64) # Found 64 experimentally (due to error message with underflowed value)
    if size > MAX_INT32:
        print('Image size is too big: %s' % str(img.shape))
        return -1, -1, -1e10, None, None, None, None, None, None

    mapp, valid, range0, range1, imgsize, other = noiseprint_blind(img, QF, models, outfilenames=outfilenames)
    return QF, mapp, valid, range0, range1, imgsize, other

def noiseprint_blind(img, QF, models, outfilenames=[]):
    # Backwards compatibility if single model and outfilename are given
    if not isinstance(models, list):
        models = [models]
    if not isinstance(outfilenames, list):
        outfilenames = [outfilenames]
      
    # Get fingerprints
    residuals = []
    for m_i, model in enumerate(models):
        if isinstance(model, str):
            # Noiseprint model
            res = genNoiseprint(img, QF, model)
            
            if len(outfilenames) > m_i:
                # Save noiseprint / compprint to file
                outfilename = outfilenames[m_i]
                ensure_dir(outfilename)
                fig = plt.figure()
                
                vmin = np.min(res[34:-34,34:-34])
                vmax = np.max(res[34:-34,34:-34])
                plt.imshow(res.clip(vmin,vmax), clim=[vmin,vmax], cmap='tab20b')
                #plt.imshow(res, cmap='tab20b')
                plt.axis('off')
                plt.savefig(outfilename, dpi=250, pad_inches=0,  bbox_inches='tight')
                plt.close(fig)
        else:
            # Comprint model
            img_in = (np.reshape(img, (1,img.shape[0],img.shape[1],1))*256 - 127.5) * 1./255    
            res = np.reshape(model.predict(img_in), (img.shape[0], img.shape[1]))

            if len(outfilenames) > m_i:
                # Save noiseprint / compprint to file
                outfilename = outfilenames[m_i]
                ensure_dir(outfilename)
                fig = plt.figure()
                plt.imshow(res, cmap='tab20b')
                plt.axis('off')
                plt.savefig(outfilename, dpi=250, pad_inches=0,  bbox_inches='tight')
                plt.close(fig)

            # Rescale
            res = (res-np.mean(res))*(1/np.var(res))
            
        assert(img.shape==res.shape)
        residuals.append(res.astype(np.float32))
        
    return noiseprint_blind_post_concat(models, residuals, img)

def noiseprint_blind_post_concat(models, residuals, img):
    spams = []
    for res in residuals:
        spam, valid, range0, range1, imgsize = getSpamFromNoiseprint(res, img)
        spams.append(spam)
        
        if np.sum(valid) < 50:
            print('error too small %d' % np.sum(weights))
            return None, valid, range0, range1, imgsize, dict()
    
    concatenated_spam = np.concatenate(tuple(spams), axis=2)
    
    mapp, other = EMgu_img(concatenated_spam, valid, extFeat = range(32), seed = 0, maxIter = 100, replicates = 10, outliersNlogl = 42)
    return mapp, valid, range0, range1, imgsize, other

def genMappFloat(mapp, valid, range0, range1, imgsize):
    mapp_s = np.copy(mapp)
    mapp_s[valid==0] = np.min(mapp_s[valid>0])
    
    mapp_s = resizeMapWithPadding(mapp_s, range0, range1, imgsize)
    
    return mapp_s

def genMappUint8(mapp, valid, range0, range1, imgsize, vmax=None, vmin=None):
    mapp_s = np.copy(mapp)
    mapp_s[valid==0] = np.min(mapp_s[valid>0])
    
    if vmax is None:
        vmax = np.nanmax(mapp_s)
    if vmin is None:     
        vmin = np.nanmin(mapp_s)
        
    mapUint8 = (255* (mapp_s.clip(vmin,vmax) - vmin) /(vmax-vmin)).clip(0, 255).astype(np.uint8)
    mapUint8 = 255 - resizeMapWithPadding(mapUint8, range0, range1, imgsize)
    
    return mapUint8
