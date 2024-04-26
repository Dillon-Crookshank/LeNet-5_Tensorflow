import os
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

# Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Define the model as a function
def defineModel():
    model = models.Sequential([
        # Add +2 padding
        layers.ZeroPadding2D(padding=((2, 2), (2, 2)), input_shape=(28, 28, 1)),
        # Convolution -> sigmoid
        layers.Conv2D(6, kernel_size=(5, 5), activation='sigmoid'),
        # Average Pooling
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Convolution -> sigmoid
        layers.Conv2D(16, kernel_size=(5, 5), activation='sigmoid'),
        # Average Pooling
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        # Flatten - 400
        layers.Flatten(),
        # Sigmoid - Fully connected - 120
        layers.Dense(120, activation='sigmoid'),
        # Sigmoid - Fully connected - 84
        layers.Dense(84, activation='sigmoid'),
        # Sigmoid - Fully connected - 10
        layers.Dense(10, activation='sigmoid'),
    ])
    
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model


# Load and normalize the dataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


# Dont train the model again if it has been saved
if (not os.path.exists('LeNet-5_model.keras')):
    model = defineModel()
    model.fit(X_train, y_train, epochs=25)
else:
    model = models.load_model('LeNet-5_model.keras')

model.save('LeNet-5_model.keras')

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')
# Calculate the final loss of the training set
loss = loss_fn(y_train, model(X_train).numpy())

print(f'Final Loss of Training Set: {loss}')

print('Testing Set Evaluation:')
evaluation = model.evaluate(X_test, y_test, verbose=2)

