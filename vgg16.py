import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Speed up training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load data
train_generator = test_datagen.flow_from_directory(
    r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\real_vs_fake\real-vs-fake\train',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\real_vs_fake\real-vs-fake\valid',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\real_vs_fake\real-vs-fake\test',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# (DEBUG) Print class distributions
train_labels = train_generator.classes
val_labels = validation_generator.classes
test_labels = test_generator.classes
print(f'Training set class distribution: {Counter(train_labels)}')
print(f'Validation set class distribution: {Counter(val_labels)}')
print(f'Test set class distribution: {Counter(test_labels)}')

# Vgg16 model
def build_vgg16(num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dense(128, activation='relu', name='fc2'))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model


# Create the model
num_classes = 2
vgg16 = build_vgg16(num_classes)
vgg16.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
vgg16.summary()

# Train the model
history = vgg16.fit(
    train_generator,
    validation_data=validation_generator,
    class_weight={0: 1., 1: 1.},
    epochs=100,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)]
)

# Evaluate the model
test_loss, test_accuracy, test_auc = vgg16.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test AUC: {test_auc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Print the loss value of the last epoch
last_epoch_index = -1
train_loss_last_epoch = history.history['loss'][last_epoch_index]
val_loss_last_epoch = history.history['val_loss'][last_epoch_index]
print(f"Training Loss: {train_loss_last_epoch}")
print(f"Validation Loss: {val_loss_last_epoch}")

# Predict on test set
Y_pred = vgg16.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Compute ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels == i, Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Architecture visualization
img_file = r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\Assignment\vgg16architecture.png'
plot_model(vgg16, show_shapes=True, show_layer_names=True, to_file=img_file)

# Confusion matrix
print(confusion_matrix(test_generator.classes[:len(y_pred)], y_pred))

# Classification report
print(classification_report(test_generator.classes[:len(y_pred)], y_pred, target_names=['Real', 'Fake']))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()

# Plot ROC curves
plt.figure()
colors = ['blue', 'red']

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
