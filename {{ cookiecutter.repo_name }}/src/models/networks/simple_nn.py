import tensorflow as tf


def simple_nn(
    input_shape=(28, 28, 1),
    batch_size=32,
    num_classes=10,
    deterministic=False,
) -> tf.keras.models.Model:
    """ """
    if deterministic:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
            ]
        )
    else:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
            ]
        )

    # add output layer
    model.add(tf.keras.layers.Dense(num_classes))

    return model
