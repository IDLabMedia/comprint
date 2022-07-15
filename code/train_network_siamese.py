# This file was created by IDLab-MEDIA, Ghent University - imec, in collaboration with GRIP-UNINA

print('START TRAINING')

import os
import sys
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras as ks

import network
import dataloader

# param
EPOCHS = 50
#POCHS = 5

# Fixed structure for Train and Validation data
folder_train = 'data/train'
folder_validation = 'data/validation'

# Fixed filenames for Train and Validation data
file_path_train = '????_small.png' # 4 digits
file_path_validation = '????_small.png' # 4 digits

train_size = 1000
#train_size = 100
val_size = 100
#val_size = 10
BATCH = 200
#BATCH = 20

model_name_init = "Comprint_Full"
model_name = "Comprint_Siamese_Full_jpg_ps_full"

if __name__ == '__main__': 
    if len(sys.argv) < 3:
        print("Need 2 arguments: %s folder_with_Data folder_for_logs Exiting..." % sys.argv[0])
        sys.exit(1)
    else:
        main_folder = sys.argv[1]
        log_folder = sys.argv[2]

    # Create generators
    print('\nCreating dataset\n')
    
    # Generate datasets
    ds_train = Dataset.list_files("%s/%s/%s" % (main_folder, folder_train, file_path_train), shuffle=False)
    ds_val   = Dataset.list_files("%s/%s/%s" % (main_folder, folder_validation, file_path_validation), shuffle=False)

    # Shuffle
    ds_train = ds_train.shuffle(train_size, reshuffle_each_iteration=True)
    ds_train = ds_train.cache() # We can cache the filenames lol
    
    # Create image pairs
    ds_train2 = ds_train.shuffle(train_size, reshuffle_each_iteration=True)
    ds_val2   = ds_val.shuffle(val_size, reshuffle_each_iteration=False)
    
    ds_train = tf.data.Dataset.zip((ds_train, ds_train2))
    ds_val   = tf.data.Dataset.zip((ds_val, ds_val2))
    
    # Repeat dataset
    ds_train = ds_train.repeat()
    ds_val   = ds_val.repeat()
    
    # Process
    # prefix ps: Photoshop level
    jpg_quality_list = tf.convert_to_tensor(["20", "25", "30", "35", "40", "50", "55", "60", "65", "70", "80", "90", "100", "ps4", "ps5", "ps6", "ps7", "ps8", "ps9", "ps10", "ps11", "ps12"], dtype=tf.string) 
    lambda_process_image = lambda f1, f2: dataloader.process_image_siamese_from_filenames(f1, f2, quality_list=jpg_quality_list)
    
    ds_train = ds_train.map(lambda_process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(lambda_process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ds_train = dataloader.configure_for_performance(ds_train, batch_size=BATCH, buffer_size=tf.data.experimental.AUTOTUNE)
    ds_val = dataloader.configure_for_performance(ds_val, batch_size=BATCH, buffer_size=tf.data.experimental.AUTOTUNE)   
    
    # Ignore IO errors 
    ds_train = ds_train.apply(tf.data.experimental.ignore_errors())
    ds_val = ds_val.apply(tf.data.experimental.ignore_errors())
    
    # # Create CNN
    print('\nCreating network\n')
    model = network.Siamese_Network()
    model.build((BATCH, 48, 48, 1))
    
    # Load previous model from checkpoint
    model_filepath = '%s/models/%s' % (main_folder, model_name_init)

    # Load model
    model_old = ks.models.load_model(model_filepath)
    weights = model_old.get_weights()
    model.set_weights(weights)
    
    # Specificy training configuration
    print('\nCompiling model\n')
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer='adam',
        loss=loss
    )

    # Define the Keras TensorBoard callback.
    print('\nDefining Callbacks\n')
    logdir = "%s/data/tensorboard/%s-%s" % (log_folder, model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        update_freq='epoch'
    )

    # Define checkpoint callback
    checkpoint_filepath = '%s/data/checkpoint/%s/cp-{epoch:04d}' % (log_folder, model_name)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Fit model 
    print("\nFitting model on training data \n")
    model.fit(
        ds_train, 
        validation_data=ds_val,
        epochs=EPOCHS, 
        callbacks=[tensorboard_callback, model_checkpoint_callback],
        verbose = 2,
        use_multiprocessing = True,
        steps_per_epoch = 6000,
        validation_steps = 500
    )
    
    # Save model
    model.save('%s/models/%s' % (log_folder, model_name))
