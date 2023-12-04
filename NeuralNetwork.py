import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Load MNIST dataset
    data = pd.read_csv('mnist_train.csv', header=0).append(
           pd.read_csv('mnist_test.csv', header=0), ignore_index=True)
    
    # Extract labels and images
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values / 255.0 # Normalize pixel values to [0,1]
    
    # Split data into train, validation and test sets in (60/20/20) fashion
    n = len(labels)
    n_train = int(0.6 * n)
    n_test = int(0.2 * n)
    X_train, y_train = images[:n_train], labels[:n_train]
    X_test, y_test = images[n_train:n_train + n_test], labels[n_train:n_train + n_test]
    X_val, y_val = images[n_train + n_test:], labels[n_train + n_test:]

    return X_train, y_train, X_test, y_test, X_val, y_val

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid activation function."""
    return x * (1 - x)

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss function."""
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

def one_hot_encode(y, num_classes):
    """Converts the class labels into one-hot encoded vectors."""
    encoded = np.zeros((y.shape[0], num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize the neural network.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List of sizes of hidden layers.
            output_size (int): Number of output classes.
        """
        
        # Initialize weights randomly for each layer
        sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def forward(self, X):
        """Performs a forward pass through the network,
        computing the outputs of each layer."""
        
        # Intermediate values, required in backward pass
        self.a = [None] * len(self.weights)
        
        # Forward propagate first layer
        z = np.dot(X, self.weights[0])
        self.a[0] = sigmoid(z)
        
        # Iterate over other layers
        for i in range(1, len(self.weights)):
            z = np.dot(self.a[i-1], self.weights[i])
            self.a[i] = sigmoid(z)
            
        # Return neural network output
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        """Performs a backward pass through the network,
        computing the derivatives of the loss function with respect to each layer's weights
        and updating the weights accordingly."""
        
        # Derivative of the loss function with respect to each layer's weights
        dW = [None] * len(self.weights)
        
        # Mini-batch size
        m = y.shape[0]
        
        if len(self.weights) == 1:
            # Derivative for the output layer's weights
            delta = (self.a[0] - y) / m
            dW[0] = np.dot(X.T, delta)
        else:
            # Derivative for the last layer's weights
            delta = (self.a[-1] - y) / m
            dW[-1] = np.dot(self.a[-2].T, delta)
            
            # Derivative for hidden layer's weights, iterating backwards
            for i in range(len(self.weights) - 1, 1, -1):
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.a[i-1])
                dW[i-1] = np.dot(self.a[i-2].T, delta)
                
            # Derivative for the first layer
            delta = np.dot(delta, self.weights[1].T) * sigmoid_derivative(self.a[0])
            dW[0] = np.dot(X.T, delta)

        # Apply derivatives on weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]

    def train(self, X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size):
        """Trains the neural network using mini-batch gradient descent,
        with a configurable number of epochs, learning rate, and batch size.
        
        Returns:
            A tuple containing two lists: train_losses and test_losses."""
        
        # One-hot encode the labels
        y_train_encoded = one_hot_encode(y_train, num_classes=10)
        y_test_encoded = one_hot_encode(y_test, num_classes=10)
        
        # Define empty lists to store the losses
        train_losses = []
        test_losses = []

        # Train for the specified number of epochs
        for epoch in range(epochs):
            # Iterate over the batches
            for i in range(0, X_train.shape[0], batch_size):
                # Inputs and expected outputs
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train_encoded[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                # Backward pass and weight update
                self.backward(X_batch, y_batch, learning_rate)
                
            # Calculate and append losses for train and test sets
            train_loss = np.mean(cross_entropy_loss(y_train_encoded, self.forward(X_train)))
            train_losses.append(train_loss)
            
            test_loss = np.mean(cross_entropy_loss(y_test_encoded, self.forward(X_test)))
            test_losses.append(test_loss)

            # Print loss and accuracy
            train_accuracy = self.evaluate(X_train, y_train_encoded)
            test_accuracy = self.evaluate(X_test, y_test_encoded)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f},",
                  f"Train Accuracy: {train_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")
        
        # Return train and test losses
        return train_losses, test_losses

    def evaluate(self, X, y):
        """Computes the accuracy of the trained neural network on a given dataset."""
        
        # Considers the highest value in output as the network's prediction
        y_pred = np.argmax(self.forward(X), axis=1)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)

def plot(train_losses_list, test_losses_list):
    # Create the X axis of the plot
    epochs = range(1, len(train_losses_list[0]) + 1)
    
    # Plot each set of training losses as a line on the graph
    for i in range(len(train_losses_list)):
        plt.plot(epochs, train_losses_list[i], label='Train ' + str(i))
        
    # Plot each set of test losses as a line on the graph
    for i in range(len(train_losses_list)):
        plt.plot(epochs, test_losses_list[i], label='Test ' + str(i))
    
    # Add a title and axis labels to the graph
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Add a legend to the graph and display the graph
    plt.legend()
    plt.show()

def main():
    # Load and split the data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data()
    
    # Specify network shape
    input_size = X_train.shape[1]
    hidden_sizes = [] # Initially has no hidden layers
    output_size = 10
    
    # Setup hyper-parameters
    learning_rate = 0.01
    epochs = 100
    batch_size = 64
    
    # To store training and testing accuracy at each epoch, for creating the graph chart
    train_losses_list, test_losses_list = [], []
    
    # Train and test different neural networks with different number of hidden layers
    for i in range(4):
        # Create the neural network
        nn = NeuralNetwork(input_size, hidden_sizes, output_size)
        
        # Train the neural network
        print("Training the neural network...")
        train_losses, test_losses = nn.train(X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size)
        
        # Add to list to plot in the end
        train_losses_list.append(train_losses)
        test_losses_list.append(test_losses)
        
        # Evaluate the neural network
        print("Evaluating the neural network...")
        accuracy = nn.evaluate(X_val, one_hot_encode(y_val, num_classes=10))
        print(f"Validation Set Accuracy: {accuracy:.4f}")
        
        # Increase the number of hidden layers for the next trial
        hidden_sizes.append(128)
    
    # Plot training and test losses for comparison
    plot(train_losses_list, test_losses_list)

if __name__ == '__main__':
    main()