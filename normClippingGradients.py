import numpy as np
import tensorflow as tf

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1)
])

# Compile the model with an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Generate some random data
X = np.random.rand(1000, 32)
y = np.random.rand(1000, 1)

# Set a clipping threshold
clip_norm = 1.0

# Initialize MeanSquaredError instance
mse = tf.keras.losses.MeanSquaredError()

# Training loop with gradient clipping
for epoch in range(10):
    with tf.GradientTape() as tape:
        predictions = model(X)
        # Use the MSE instance to compute the loss
        loss = mse(y, predictions)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Clip gradients
    clipped_gradients = [tf.clip_by_norm(g, clip_norm) for g in gradients]

    # Apply gradients
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")
