import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Data preparation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                 r'Learning\real_vs_fake\real-vs-fake\train',
                                                 target_size=(32, 32),
                                                 batch_size=500,
                                                 class_mode='binary')

validation_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                  r'Learning\real_vs_fake\real-vs-fake\valid',
                                                  target_size=(32, 32),
                                                  batch_size=100,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                            r'Learning\real_vs_fake\real-vs-fake\test',
                                            target_size=(32, 32),
                                            batch_size=100,
                                            class_mode='binary')

# Create model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(training_set,
                    epochs=100,
                    validation_data=validation_set,
                    )

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_set)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict on test set
Y_pred = model.predict(test_set)
y_pred = (Y_pred > 0.5).astype(int)

# Results
print(confusion_matrix(validation_set.classes, y_pred))
print(classification_report(validation_set.classes, y_pred, target_names=['Real', 'Fake']))

# Visualizations
img_file = r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\Assignment\CNN\architecture.png'
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file=img_file)

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
