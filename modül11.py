import tensorflow as tf

model = tf.keras.models.load_model("mpiigaze_fixed.keras")

model.summary()
print("Loss:", model.loss)

