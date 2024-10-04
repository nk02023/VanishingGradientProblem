import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Custom callback to track gradients
class GradientLogger(Callback):
    def __init__(self, y_true):
        super().__init__()
        self.gradients = []
        self.y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)  # Convert to TensorFlow tensor

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            # Forward pass to get predictions
            predictions = self.model(self.model.inputs[0], training=True)
            # Calculate loss using the stored true labels
            loss = tf.keras.losses.categorical_crossentropy(self.y_true, predictions)

        # Get the gradients
        gradients = tape.gradient(loss, self.model.trainable_weights)
        # Append the mean of each gradient to the list
        self.gradients.append([tf.reduce_mean(g).numpy() if g is not None else 0 for g in gradients])

# Function to build and train the model
def build_and_train_model(activation):
    inputs = Input(shape=(4,))
    outputs = Dense(3, activation=activation)(inputs)  # 3 output neurons for 3 classes
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Initialize the gradient logger with true labels
    gradient_logger = GradientLogger(y_encoded)
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, callbacks=[gradient_logger], verbose=0)
    return gradient_logger.gradients

# Train the model with sigmoid activation
print("Training model with sigmoid activation...")
sigmoid_gradients = build_and_train_model('sigmoid')

# Train the model with ReLU activation
print("Training model with ReLU activation...")
relu_gradients = build_and_train_model('relu')

# Plotting the gradients
plt.figure(figsize=(14, 6))

# Plot for sigmoid activation
plt.subplot(1, 2, 1)
plt.plot(np.mean(sigmoid_gradients, axis=1))
plt.title('Mean Gradients - Sigmoid Activation')
plt.xlabel('Epochs')
plt.ylabel('Mean Gradient Value')
plt.ylim([-0.01, 0.1])

# Plot for ReLU activation
plt.subplot(1, 2, 2)
plt.plot(np.mean(relu_gradients, axis=1))
plt.title('Mean Gradients - ReLU Activation')
plt.xlabel('Epochs')
plt.ylabel('Mean Gradient Value')
plt.ylim([-0.01, 0.1])

plt.tight_layout()
plt.show()
