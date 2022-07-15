# This file was created by the Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA')
# and adapted by IDLab-MEDIA, Ghent University - imec, in collaboration with GRIP-UNINA

import scipy.io as sio
from time import time
import io
import os
import matplotlib.pyplot as plt 
import splicebuster.noiseprint.noiseprint_blind_concat

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def extract_heatmap_concat(imgfilename, outfilename, models, fingerprint_filenames=[], heatmap_filename=""):
    with open(imgfilename,'rb') as f:
        stream = io.BytesIO(f.read())

    timestamp = time()
    QF, mapp, valid, range0, range1, imgsize, other = splicebuster.noiseprint.noiseprint_blind_concat.noiseprint_blind_file(imgfilename, models, outfilenames=fingerprint_filenames)
    timeApproach = time() - timestamp

    if mapp is None:
        print('Image is too small or too uniform')

    out_dict = dict()
    out_dict['QF'     ] = QF
    out_dict['map'    ] = mapp
    out_dict['valid'  ] = valid
    out_dict['range0' ] = range0
    out_dict['range1' ] = range1
    out_dict['imgsize'] = imgsize
    out_dict['other'  ] = other
    out_dict['time'   ] = timeApproach

    ensure_dir(outfilename)
    
    if outfilename[-4:] == '.mat':
        import scipy.io as sio
        sio.savemat(outfilename, out_dict)
    else:
        import numpy as np
        np.savez(outfilename, **out_dict)
        
    if heatmap_filename:
        # Save heatmap to file
        ensure_dir(heatmap_filename)
        plt.figure(5)
        plt.imshow(mapp, clim=[np.nanmin(mapp),np.nanmax(mapp)], cmap='jet')
        plt.axis('off')
        plt.savefig(heatmap_filename, dpi=250, pad_inches=0,  bbox_inches='tight')
        plt.close()
        print("Heatmap saved to %s" % heatmap_filename)