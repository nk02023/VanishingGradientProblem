import numpy as np

# Activation function (logistic sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize the weight matrix W with random values
def initialize_weights(n):
    return np.random.uniform(-0.1, 0.1, (n, n))

# Compute the norm of the weight matrix W
def compute_matrix_norm(W):
    return np.max(np.sum(np.abs(W), axis=1))

# Calculate the scaling factor for the backpropagated error
def compute_scaling_factor(W, q, n, fmax_prime=0.25, wmax=4.0):
    W_norm = compute_matrix_norm(W)
    
    # Calculate the xi (ξ) value for exponential decay
    xi = (n * W_norm) / wmax
    xi_factor = min(xi, 1.0)  # ξ should be less than 1 for the decay
    
    # Exponential decay of the scaling factor over q time steps
    scaling_factor = n * (xi_factor ** q)
    
    return scaling_factor

# Perform forward pass through the RNN
def forward_pass(W, Y_prev):
    net_input = np.dot(W, Y_prev)
    return sigmoid(net_input)

# Main function to simulate the RNN and backpropagation scaling factor
def simulate_rnn(T, n, q):
    # Initialize random weight matrix
    W = initialize_weights(n)
    
    # Initial activation vector Y(0)
    Y = np.random.uniform(-1, 1, n)
    
    print(f"Initial activations Y(0): {Y}")
    
    # Forward pass through T time steps
    for t in range(1, T + 1):
        Y = forward_pass(W, Y)
        print(f"Activations at time {t}: {Y}")
    
    # Compute the upper bound for the absolute scaling factor after q time steps
    scaling_factor = compute_scaling_factor(W, q, n)
    
    print(f"\nUpper bound for the absolute scaling factor after {q} time steps: {scaling_factor}")

# Parameters
T = 10  # Number of forward time steps
n = 5   # Number of neurons in the RNN
q = 5   # Number of time steps for backpropagation

# Run the RNN simulation
simulate_rnn(T, n, q)
