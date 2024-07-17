import tensorflow as tf
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Data preparation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                 r'Learning\real_vs_fake\real-vs-fake\train',
                                                 target_size=(256, 256),
                                                 batch_size=32,
                                                 class_mode='categorical')

validation_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                  r'Learning\real_vs_fake\real-vs-fake\valid',
                                                  target_size=(256, 256),
                                                  batch_size=32,
                                                  class_mode='categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                            r'Learning\real_vs_fake\real-vs-fake\test',
                                            target_size=(256, 256),
                                            batch_size=32,
                                            class_mode='categorical')


def build_vgg16():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
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

    model.add(Dense(256, activation='relu', name ='fc1'))
    model.add(Dense(128, activation='relu', name ='fc2'))
    model.add(Dense(196, activation='softmax', name ='output'))

    return model


# Create the model
vgg16 = build_vgg16()
opt = SGD(learning_rate=1e-6, momentum=0.9)
vgg16.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
vgg16.summary()

# Train the model
history = vgg16.fit(
    training_set,
    validation_data=validation_set,
    epochs=100
)

# Evaluate the model
test_loss, test_accuracy = vgg16.evaluate(test_set)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict on test set
Y_pred = vgg16.predict(test_set)
y_pred = Y_pred.argmax(axis=1)

# Results
print(confusion_matrix(test_set.classes, y_pred))
print(classification_report(test_set.classes, y_pred, target_names=['Real', 'Fake']))

# Visualizations
img_file = r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\Assignment\CNN\architecture.png'
tf.keras.utils.plot_model(vgg16, show_shapes=True, show_layer_names=True, to_file=img_file)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy (training & valid)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (training & val)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()
