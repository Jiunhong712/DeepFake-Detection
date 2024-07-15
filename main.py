import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                 r'Learning\real_vs_fake\real-vs-fake\train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

validation_set = test_datagen.flow_from_directory(r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine '
                                                  r'Learning\real_vs_fake\real-vs-fake\valid',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

# Model creation
model = Sequential()

# Adding layers
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate steps per epoch
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = validation_set.samples // validation_set.batch_size

# Train the model
history = model.fit(training_set,
                    steps_per_epoch=steps_per_epoch,
                    epochs=1,
                    validation_data=validation_set,
                    validation_steps=validation_steps)

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_set)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')

# Predict on validation set
Y_pred = model.predict(validation_set)
y_pred = (Y_pred > 0.5).astype(int)

# Confusion matrix
print(confusion_matrix(validation_set.classes, y_pred))

# Classification report
print(classification_report(validation_set.classes, y_pred, target_names=['Real', 'Fake']))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
