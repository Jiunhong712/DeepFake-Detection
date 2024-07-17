import tensorflow as tf
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix

input_shape = (256, 256, 3)
num_classes = 2

# Data preparation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                 r'Learning\real_vs_fake\real-vs-fake\train',
                                                 target_size=(256, 256),
                                                 batch_size=500,
                                                 class_mode='binary')

validation_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                  r'Learning\real_vs_fake\real-vs-fake\valid',
                                                  target_size=(256, 256),
                                                  batch_size=100,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                            r'Learning\real_vs_fake\real-vs-fake\test',
                                            target_size=(256, 256),
                                            batch_size=100,
                                            class_mode='binary')


def alexnet(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(4, 4), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='relu'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model


# Create the model
alexnet = alexnet(input_shape, num_classes)
alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
alexnet.summary()

# Train the model
history = alexnet.fit(
    training_set,
    validation_data=validation_set,
    epochs=100
)

# Evaluate the model
test_loss, test_accuracy = alexnet.evaluate(test_set)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict on test set
Y_pred = alexnet.predict(test_set)
y_pred = (Y_pred > 0.5).astype(int)

# Results
print(confusion_matrix(test_set.classes, y_pred))
print(classification_report(test_set.classes, y_pred, target_names=['Real', 'Fake']))

# Visualizations
img_file = r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\Assignment\CNN\architecture.png'
tf.keras.utils.plot_model(alexnet, show_shapes=True, show_layer_names=True, to_file=img_file)

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
