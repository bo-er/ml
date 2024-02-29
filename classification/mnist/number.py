from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils    import to_categorical
from keras          import datasets
from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import numpy as np
import requests
import ssl

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def buildAndCompileModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), 
            activation='relu', 
            input_shape=(28, 28, 1))
            )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Add a flatten layer to convert the 3D feature maps to 1D feature vectors
    model.add(layers.Flatten())
    # Add a dropout layer to prevent overfitting
    model.add(layers.Dropout(0.25))
    # Add a dense layer with relu activation
    model.add(layers.Dense(64, activation='relu'))
    # Add a dense layer with softmax activation for the output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    # Print the model summary
    model.summary()
    return model

def trainModel(model):
    (train_data,train_labels), (test_data,test_labels)  = mnist.load_data()
    train_data = train_data.reshape((60000, 28, 28, 1))
    train_data = train_data.astype('float32') / 255
    train_labels = to_categorical(train_labels)

    test_data  = test_data.reshape((10000, 28, 28, 1))
    test_data  = test_data.astype('float32') / 255
    test_labels  = to_categorical(test_labels)
    model.fit(train_data, 
            train_labels, 
            # An epoch is an iteration over the entire train_data and train_labels provided.
            epochs=10, 
            # batch_size is the number of samples per gradient update. If unspecified, batch_size will default to 32.
            batch_size=64)
    return model

def main():
    model = buildAndCompileModel()
    model = trainModel(model)
    
    i = 0
    for no in [1,2,3,4,5,6,7,8,9]:
        path = "./figures/"+str(no)+".png"
        img = load_img(path, target_size=(28, 28),color_mode="grayscale")
        img = 255-img_to_array(img)

        # plt.imshow(img , cmap=plt.cm.binary)
        # plt.show()

        img  = img.astype('float32')/255
        img  = img.reshape((1, 28, 28, 1))

        y_pred = model.predict(img)
        number = np.argmax(y_pred, axis=1)[0]
        print('prediction:',number)
        print('estimated accuracy: ',y_pred[0][number])
   

if __name__ == "__main__":
    main()
    

