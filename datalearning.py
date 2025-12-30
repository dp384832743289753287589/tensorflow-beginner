import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split = ["train", "test"],
    shuffle_files = True,
    as_supervised = True,
    with_info = True
  )
def normalize_img(image, label):
    return tf.cast(image, tf.float64) / 255.0, label #convert images values from [0, 255] to [0.0, 1.0] for better training
ds_train = ds_train.map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)
ds_train=ds_train.batch(32)#normalize, batch and optimize dataset performance
ds_test = ds_test.map(normalize_img, num_parallel_calls = tf.data.AUTOTUNE)
ds_test=ds_test.batch(32)
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
model.fit(ds_train, epochs=5,validation_data=ds_test)
model.save("mnist_model.keras")
for image, label in ds_test.take(20):
    predictions = model.predict(image)
    predicted_labels = np.argmax(predictions, axis=1)
    for i in range(len(image)):
        plt.imshow(tf.squeeze(image[i]), cmap='gray')
        plt.title(f" Predicted: {predicted_labels[i]}, Actual: {label[i]}")
        if predicted_labels[i] == label[i]:
            plt.text(10, 10, "Correct", color="green")
        if predicted_labels[i] != label[i]:
            plt.text(10, 10, "Incorrect", color="red")
    plt.show()
    

