import ArcFace
import argparse
import cv2
import glob
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import pickle
import warnings
from keras import callbacks
import one_cycle_scheduler
# class EarlyStoppingWithConvergence(callbacks.Callback):
#     def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
#         super(EarlyStoppingWithConvergence, self).__init__()
#         self.monitor = monitor
#         self.min_delta = min_delta
#         self.patience = patience
#         self.verbose = verbose
#         self.baseline = baseline
#         self.restore_best_weights = restore_best_weights

#         if mode not in ['auto', 'min', 'max']:
#             warnings.warn('EarlyStopping mode %s is unknown, '
#                           'fallback to auto mode.' % (mode),
#                           RuntimeWarning)
#             mode = 'auto'

#         if mode == 'min':
#             self.monitor_op = np.less
#             self.min_delta *= -1
#         elif mode == 'max':
#             self.monitor_op = np.greater
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = np.greater
#             else:
#                 self.monitor_op = np.less

#         if self.monitor_op == np.greater:
#             self.min_delta *= -1

#         self.best = np.Inf if self.monitor_op == np.less else -np.Inf
#         self.wait = 0
#         self.stopped_epoch = 0

#     def on_epoch_end(self, epoch, logs=None):
#         current = self.get_monitor_value(logs)
#         if current is None:
#             return

#         if (np.abs(current - self.best) < self.min_delta):
#             self.wait += 1
#         else:
#             self.best = current
#             self.wait = 0

#         if self.wait >= self.patience:
#             self.stopped_epoch = epoch
#             self.model.stop_training = True
#             if self.restore_best_weights and self.best_weights is not None:
#                 if self.verbose > 0:
#                     print('Restoring model weights from the end of the best epoch.')
#                 self.model.set_weights(self.best_weights)

#     def on_train_end(self, logs=None):
#         if self.stopped_epoch > 0 and self.verbose > 0:
#             print('Epoch %05d: early stopping with convergence.' % (self.stopped_epoch + 1))

#     def get_monitor_value(self, logs):
#         logs = logs or {}
#         monitor_value = logs.get(self.monitor)
#         if monitor_value is None:
#             warnings.warn(
#                 'Early stopping conditioned on metric `%s` '
#                 'which is not available. Available metrics are: %s' %
#                 (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
#             )
#         return monitor_value


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, default='Norm',
                help="path to Norm/dir")
ap.add_argument("-o", "--save", type=str, default='models/model.h5',
                help="path to save .h5 model")
ap.add_argument("-l", "--le", type=str, default='models/le.pickle',
	            help="path to label encoder")
ap.add_argument("-b", "--batch_size", type=int, default=16,
	            help="batch Size for model training")
ap.add_argument("-e", "--epochs", type=int, default=200,
	            help="Epochs for Model Training")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
checkpoint_path = args['save']

# Load ArcFace Model
model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
print("ArcFace model ", model.layers[0].input_shape[0][1:], " inputs")
print("represents faces as ",
      model.layers[-1].output_shape[1:], " dimensional vectors")
target_size = model.layers[0].input_shape[0][1:3]
print('target_size: ', target_size)

print(model.summary())

# Variable for store img Embedding
x = []
y = []

names = os.listdir(path_to_dir)
names = sorted(names)
class_number = len(names)

for name in names:
    img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
    img_list = sorted(img_list)

    for img_path in img_list:
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, target_size)
        img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_norm = img_pixels/255  # normalize input in [0, 1]
        img_embedding = model.predict(img_norm)[0]

        x.append(img_embedding)
        y.append(name)
        print(f'Embedding {img_path}')
    print(f'Completed {name} Part')
print('Image data embedding completed...')

# DataFrame
df = pd.DataFrame(x, columns=np.arange(512))
x = df.copy()
x = x.astype('float64')

le = LabelEncoder()
labels = le.fit_transform(y)

print(labels)
labels = tf.keras.utils.to_categorical(labels, class_number)

# Train Deep Neural Network
x_train, x_test, y_train, y_test = train_test_split(x, labels,
                                                    test_size=0.2,
                                                    random_state=0)

model = Sequential([
    layers.Dense(1024, activation='relu', input_shape=[512]),
    layers.Dense(512, activation='relu'),
    layers.Dense(class_number, activation="softmax")
])

# Model Summary
print('Model Summary: ', model.summary())

model.compile(
    optimizer= 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience = 3)
num_samples = class_number
max_lr = 0.01
batch_size = 32
num_epochs = 10
pct_start = 0.3
anneal_strategy = 'cos'
one_cycle = one_cycle_scheduler.OneCycleLr(num_samples, num_epochs, max_lr, batch_size, pct_start, anneal_strategy)

print('Model training started ...')
# Start training
history = model.fit(x_train, y_train,
                    epochs=args['epochs'],
                    batch_size=args['batch_size'],
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, earlystopping, one_cycle])

print('Model training completed')
print(f'Model successfully saved in /{checkpoint_path}')

# save label encoder
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
print('Successfully saved models/le.pickle')

# Plot History
metric_loss = history.history['loss']
metric_val_loss = history.history['val_loss']
metric_accuracy = history.history['accuracy']
metric_val_accuracy = history.history['val_accuracy']

# Construct a range object which will be used as x-axis (horizontal plane) of the graph.
epochs = range(len(metric_loss))

# Plot the Graph.
plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
plt.plot(epochs, metric_accuracy, 'blue', label=metric_accuracy)
plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

# Add title to the plot.
plt.title(str('Model metrics'))

# Add legend to the plot.
plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

# If the plot already exist, remove
plot_png = os.path.exists('metrics.png')
if plot_png:
    os.remove('metrics.png')
    plt.savefig('metrics.png', bbox_inches='tight')
else:
    plt.savefig('metrics.png', bbox_inches='tight')
print('Successfully Saved metrics.png')
