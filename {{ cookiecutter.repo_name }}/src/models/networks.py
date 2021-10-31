# #!/usr/bin/env python3
#
# import tensorflow as tf
#
#
# def simple_nn(input_shape=(28, 28), n_classes=10, debug=False):
#     """ """
#
#     if debug:
#         model = tf.keras.models.Sequential(
#             [
#                 tf.keras.layers.Flatten(input_shape=input_shape),
#                 tf.keras.layers.Dense(128, activation="relu"),
#             ]
#         )
#     else:
#         model = tf.keras.models.Sequential(
#             [
#                 tf.keras.layers.InputLayer(input_shape=input_shape),
#                 tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#                 tf.keras.layers.MaxPooling2D(2, 2),
#                 tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#                 tf.keras.layers.MaxPooling2D(2, 2),
#                 tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#                 tf.keras.layers.MaxPooling2D(2, 2),
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(512, activation="relu"),
#             ]
#         )
#
#     # add output layer
#     model.add(tf.keras.layers.Dense(n_classes))
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(0.001),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#     )
#     return model
