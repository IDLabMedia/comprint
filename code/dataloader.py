import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.io import read_file, decode_and_crop_jpeg, decode_png, decode_jpeg, decode_gif
from tensorflow.image import random_crop, adjust_jpeg_quality, stateless_random_crop
from tensorflow.math import subtract, multiply
import time
import os
import random

from io import BytesIO
from PIL import Image

# Method to read and decode png file
def process_png(path):
    img = read_file(path)
    img = decode_png(img)
    return img

# Method to randomly choose an element out of a tensor-array
def random_choice(a):
    shape = tf.gather(tf.shape(a), 0) 
    choice_index = tf.random.uniform([], minval=0, maxval=shape, dtype=tf.int32)
    sample = tf.gather(a, choice_index)
    return sample

# Process image for pre-trained network (JPEG artifact cleaning)
# Input: Uncompressed decoded image
# Output: JPEG-compressed decoded image and difference between JPEG - uncompressed, both scaled between -1 and +1.
def process_image(png, QF_list=tf.convert_to_tensor([20,25,30,35,40,50,60,70,80,90])):   
    # Get random patch from PNG
    png = random_crop(png, [50,50,1])
    
    # Get JPEG
    QF = random_choice(QF_list) # Full
    jpg = adjust_jpeg_quality(png, QF)
    
    # convert dtype
    png = tf.cast(png, tf.dtypes.float32)
    jpg = tf.cast(jpg, tf.dtypes.float32)
    
    # Get noise
    noise = subtract(jpg, png)
    
    # Scale data
    jpg_scaled = multiply(subtract(jpg, 127.5), 1/255)
    noise_scaled = multiply(noise, 1/255)
    
    return jpg_scaled, noise_scaled 

# Process image-pair for siamese network (Comprint extraction)
# Input: Uncompressed decoded image 1, Uncompressed decoded image 2, list of QFs to choose from
# Output: JPEG-compressed decoded image 1, JPEG-compressed decoded image 2, random label (0 if different QF, 1 if same QF)
# Note: no support for Photoshop quantization tables
def process_image_siamese(img1, img2, QF_list=tf.convert_to_tensor([20,25,30,35,40,50,60,70,80,90])):
    # Get random patches
    img1 = random_crop(img1, [48,48,1])
    img2 = random_crop(img2, [48,48,1])
    
    # Get first JPEG
    QF1 = random_choice(QF_list)
    jpg1 = adjust_jpeg_quality(img1, QF1)
    
    # Get second JPEG
    label = tf.random.uniform([], dtype=tf.dtypes.float32) < 0.5
    
    if label:
        jpg2 = adjust_jpeg_quality(img2, QF1)
    else:
        QF_list_adap = tf.squeeze(tf.gather(QF_list, tf.where(QF_list != QF1)), axis=1)
        QF2 = random_choice(QF_list_adap)
        jpg2 = adjust_jpeg_quality(img2, QF2)
        
    # convert dtype
    jpg1  = tf.cast(jpg1, tf.dtypes.float32)
    jpg2  = tf.cast(jpg2, tf.dtypes.float32)
    label = tf.cast(label, tf.dtypes.int16)
        
    # Scale data
    jpg1_scaled = multiply(subtract(jpg1, 127.5), 1/255)
    jpg2_scaled = multiply(subtract(jpg2, 127.5), 1/255)
    
    return jpg1_scaled, jpg2_scaled, label

# Process image-pair for siamese network (Comprint extraction)
# Input: Filename of uncompressed decoded image 1, Filename of uncompressed decoded image 2, list of QFs to choose from AS STRINGS(!)
# Output: JPEG-compressed decoded image 1, JPEG-compressed decoded image 2, random label (0 if different QF, 1 if same QF)
# Note: the compressed images need to be pre-saved
# Note: this method has support for Photoshop quantization tables (as pre-saved images)
def process_image_siamese_from_filenames(full_filename1, full_filename2, quality_list=tf.convert_to_tensor(["20", "25", "30", "35", "40", "50", "55", "60", "65", "70", "80", "90", "100", "ps4", "ps5", "ps6", "ps7", "ps8", "ps9", "ps10", "ps11", "ps12"])):    
    # Get first quality
    quality1 = random_choice(quality_list)
    
    # Get second quality
    # Random chance: same quality or different quality
    label = tf.random.uniform([], dtype=tf.dtypes.float32) < 0.5
    if label:
        quality2 = quality1
    else:
        quality_list_adap = tf.squeeze(tf.gather(quality_list, tf.where(quality_list != quality1)), axis=1)
        quality2 = random_choice(quality_list_adap)
     
    # Create filename with quality
    # Compressed variants assumed to be available as filename_jpg_q${QF}.png
    filename_quality1 = full_filename1 + "_jpg_q" + quality1 + ".png"
    filename_quality2 = full_filename2 + "_jpg_q" + quality2 + ".png"

    # Read this image
    img1 = process_png(filename_quality1)
    img2 = process_png(filename_quality2)

    # Crop
    img1 = random_crop(img1, [48,48,1])
    img2 = random_crop(img2, [48,48,1])
        
    # convert dtype
    jpg1 = tf.cast(img1, dtype=tf.float32)
    jpg2 = tf.cast(img2, dtype=tf.float32)
    label = tf.cast(label, tf.dtypes.int16)
        
    # Scale data
    jpg1_scaled = multiply(subtract(jpg1, 127.5), 1/255)
    jpg2_scaled = multiply(subtract(jpg2, 127.5), 1/255)
    
    return jpg1_scaled, jpg2_scaled, label 

def configure_for_performance(ds, batch_size, buffer_size):
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size)
    return ds 
