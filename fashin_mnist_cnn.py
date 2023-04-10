import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import requests
requests.packages.urllib3.disable_warnings()
import ssl
import warnings
warnings.filterwarnings('ignore')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

class defineCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.995:
            print('Accuracy reached 99.5% hence stopping training...')
            self.model.stop_training = True




if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print("Number of training data: ", x_train.shape[0])
    print("Number of test data: ",x_test.shape[0])

    print("The labels are: ", np.unique(y_train))

    #Normalize Datasets
    x_train = x_train/255.0
    x_test = x_test/255.0



    #Define the model architecture
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape=[28,28,1]),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(units=128, activation='relu'),
                                        tf.keras.layers.Dense(units=10, activation='softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = defineCallback()
    history = model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=20, callbacks=[callbacks])

    #Generate the training and validation loss and accuracy plot
    plt.subplot(121)
    plt.plot(history.history['loss'], 'b-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Make the prediction on the validation data
    predictions = model.predict(x_test)
    predicted_label = np.argmax(predictions, axis=1)

    # Print the first train data with label and first test data with predicted label
    plt.subplot(121)
    plt.imshow(x_train[0]*255, cmap='gray')
    plt.title("Label of the 1st training data is: " + str(y_train[0]))

    plt.subplot(122)
    plt.imshow(x_test[0]*255, cmap='gray')
    plt.title("Label of the 1st test data is: " + str(predicted_label[0]))
    plt.show()