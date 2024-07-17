import tensorflow as tf
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix

input_shape = (256, 256, 3)

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


def lenet5(input_shape):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))

    model.add(Dense(84, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    return model


# Create the model
lenet5 = lenet5(input_shape)
lenet5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lenet5.summary()

# Train the model
history = lenet5.fit(
    training_set,
    validation_data=validation_set,
    epochs=100
)

# Evaluate the model
test_loss, test_accuracy = lenet5.evaluate(test_set)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict on test set
Y_pred = lenet5.predict(test_set)
y_pred = Y_pred.argmax(axis=1)

# Results
print(confusion_matrix(test_set.classes, y_pred))
print(classification_report(test_set.classes, y_pred, target_names=['Real', 'Fake']))

# Visualizations
img_file = r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\Assignment\CNN\architecture.png'
tf.keras.utils.plot_model(lenet5, show_shapes=True, show_layer_names=True, to_file=img_file)

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
