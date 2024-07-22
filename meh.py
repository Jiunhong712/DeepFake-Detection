import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def alexnet(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


img_file = r'C:\Users\xavie\OneDrive\Documents\Y3S2\Y3S2 Machine Learning\Assignment\architecture.png'
input_shape = (256, 256, 3)
num_classes = 2
alexnet = alexnet(input_shape, num_classes)
alexnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
alexnet.summary()
plot_model(alexnet, show_shapes=True, show_layer_names=True, to_file=img_file)


# Plot micro-average ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=2,
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()