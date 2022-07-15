# This file was created by IDLab-MEDIA, Ghent University - imec, in collaboration with GRIP-UNINA

import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers
import time 
from tensorflow.python.keras.engine.training import reduce_per_replica 

def Create_Network(channels=1, depth=20, filters=64):
    model = ks.Sequential()

    # Add first layer
    model.add(layers.Conv2D(filters=filters, kernel_size=3, padding='same'))
    model.add(layers.ReLU())

    # Add all internal layers
    for i in range(depth-2):
        model.add(layers.Conv2D(filters=filters, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-05))
        model.add(layers.ReLU())

    # Add output layer
    model.add(layers.Conv2D(filters=channels, kernel_size=3, padding='same'))

    return model

class Siamese_Network(tf.keras.models.Model):
    def __init__(self, channels=1, filters=64):
        super(Siamese_Network, self).__init__()
        
        # Add layers 
        self.conv1  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.relu1  = layers.ReLU()
        
        self.conv2  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu2  = layers.ReLU()
    
        self.conv3  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu3  = layers.ReLU()
        
        self.conv4  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch4 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu4  = layers.ReLU()
        
        self.conv5  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch5 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu5  = layers.ReLU()
        
        self.conv6  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch6 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu6  = layers.ReLU()
        
        self.conv7  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch7 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu7  = layers.ReLU()
        
        self.conv8  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch8 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu8  = layers.ReLU()
        
        self.conv9  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch9 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu9  = layers.ReLU()
        
        self.conv10  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch10 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu10  = layers.ReLU()
        
        self.conv11  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch11 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu11  = layers.ReLU()
        
        self.conv12  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch12 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu12  = layers.ReLU()
        
        self.conv13  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch13 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu13  = layers.ReLU()
        
        self.conv14  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch14 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu14  = layers.ReLU()
        
        self.conv15  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch15 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu15  = layers.ReLU()
        
        self.conv16  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch16 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu16  = layers.ReLU()
        
        self.conv17  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch17 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu17  = layers.ReLU()
        
        self.conv18  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch18 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu18  = layers.ReLU()
        
        self.conv19  = layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.batch19 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)
        self.relu19  = layers.ReLU()
        
        self.conv20 = layers.Conv2D(filters=channels, kernel_size=3, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        
        x = self.conv7(x)
        x = self.batch7(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        x = self.batch8(x)
        x = self.relu8(x)
        
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.relu9(x)
        
        x = self.conv10(x)
        x = self.batch10(x)
        x = self.relu10(x)
        
        x = self.conv11(x)
        x = self.batch11(x)
        x = self.relu11(x)
        
        x = self.conv12(x)
        x = self.batch12(x)
        x = self.relu12(x)
        
        x = self.conv13(x)
        x = self.batch13(x)
        x = self.relu13(x)
        
        x = self.conv14(x)
        x = self.batch14(x)
        x = self.relu14(x)
        
        x = self.conv15(x)
        x = self.batch15(x)
        x = self.relu15(x)
        
        x = self.conv16(x)
        x = self.batch16(x)
        x = self.relu16(x)
        
        x = self.conv17(x)
        x = self.batch17(x)
        x = self.relu17(x)
        
        x = self.conv18(x)
        x = self.batch18(x)
        x = self.relu18(x)
        
        x = self.conv19(x)
        x = self.batch19(x)
        x = self.relu19(x)
        
        return self.conv20(x)
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        img1, img2, label = data

        with tf.GradientTape() as tape:
            y_pred1 = self(img1, training=True)  # Forward pass
            y_pred2 = self(img2, training=True)  # Forward pass
            
            # Calculate distances
            diff = tf.square(tf.math.reduce_euclidean_norm(tf.math.subtract(y_pred1, y_pred2), axis=[1,2]))
            # Softmax 
            exp = tf.math.exp(tf.math.negative(diff))
            norm = tf.math.reduce_sum(exp, axis=0)
            p = tf.divide(exp, norm + 1e-6)
            
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(label, p, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(label, p)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        img1, img2, label = data
        # Compute predictions
        y_pred1 = self(img1, training=False)  
        y_pred2 = self(img2, training=False)  

        # Calculate distances
        diff = tf.square(tf.math.reduce_euclidean_norm(tf.math.subtract(y_pred1, y_pred2), axis=[1,2]))
        # Softmax 
        exp = tf.math.exp(tf.math.negative(diff))
        norm = tf.math.reduce_sum(exp, axis=0)
        p = tf.divide(exp, norm + 1e-6)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(label, p, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(label, p)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
        