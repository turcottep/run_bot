import os
import time
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

image_resolution = 256

# check if keras is using gpu
print("tf config", tf.config.list_physical_devices('GPU'))


def split_data():
    # split data into training and validation folders in training_split folder
    split = 0.2

    print("splitting data")

    for folder in os.listdir("training"):
        if not os.path.isdir("training/" + folder):
            continue

        files = os.listdir("training/" + folder)
        files = [file for file in files if file.endswith(".png")]
        files = np.array(files)

        train, val = train_test_split(files, test_size=split)

        print("folder", folder)

        print("train", len(train))

        print("val", len(val))

        for file in train:

            if not os.path.exists("training_split/train/" + folder):
                os.makedirs("training_split/train/" + folder)

            # move file to training folder
            os.system("cp training/" + folder + "/" + file + " training_split/train/" + folder + "/" + folder + "_" + file)

        for file in val:

            if not os.path.exists("training_split/val/" + folder):
                os.makedirs("training_split/val/" + folder)

            # move file to validation folder
            os.system("cp training/" + folder + "/" + file + " training_split/val/" + folder + "/" + folder + "_" + file)


def train_model():
    print("Starting AI")

    input_shape = (image_resolution, image_resolution, 3)
    output_shape = 6

    image_generator = ImageDataGenerator(
        brightness_range=[0.5, 1.5],
        rescale=1./255,
        # validation_split=0.2
    )
    batch_size = 32
    training_generator = image_generator.flow_from_directory(
        "training_split/train",
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    validation_generator = image_generator.flow_from_directory(
        "training_split/val",
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(image_resolution, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(output_shape, activation="softmax"))

    model.compile(
        optimizer=SGD(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # fit, save model at each epoch

    history = model.fit(
        training_generator,
        epochs=10,
        # steps_per_epoch=1,
        validation_data=validation_generator,
        # validation_steps=100,
        callbacks=[
            K.callbacks.ModelCheckpoint(
                filepath="models/model.h5",
                monitor="val_loss",
                save_best_only=False,
            ),

        ]
    )

    timestamp = str(int(time.time()))
    model_name = "model_" + timestamp + ".h5"
    model.save(model_name)

    print(history.history.keys())

    # display training curves for accuracy and loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    # save plot
    plt.savefig("training_plot" + timestamp + ".png")

    plt.show()


predictions = [
    "left", "left_up", "noop", "right", "right_up", "up"
]


def test_model():
    model = load_model("model.h5")

    tests = []

    for folder in os.listdir("training"):
        if not os.path.isdir("training/" + folder):
            continue

        files = os.listdir("training/" + folder)
        files = [file for file in files if file.endswith(".png")]
        files = np.array(files)
        for file in files:
            tests.append("training/" + folder + "/" + file)

    random.shuffle(tests)

    for test in tests:
        image = cv2.imread(test)
        image = cv2.resize(image, (image_resolution, image_resolution))
        image = np.array(image)
        image = image.reshape(1, image_resolution, image_resolution, 3)
        image = image / 255

        prediction = model.predict(image, verbose=0)
        prediction_text = predictions[np.argmax(prediction)]
        actual_text = test.split("/")[1]

        # print("prediction:", predictions[np.argmax(prediction)], "actual:", test.split("/")[1])

        # show on original image
        image = cv2.imread(test)
        image = cv2.resize(image, (image_resolution, image_resolution))
        image = cv2.resize(image, (512, 512))
        color = prediction_text == actual_text and (0, 255, 0) or (0, 0, 255)

        # draw background for text
        image = cv2.rectangle(image, (0, 30), (150, 80), (0, 0, 0), -1)

        image = cv2.putText(image, "predicted: " + prediction_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        image = cv2.putText(image, "actual: " + actual_text, (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imshow("image", image)
        cv2.waitKey(10000)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    train_model()
