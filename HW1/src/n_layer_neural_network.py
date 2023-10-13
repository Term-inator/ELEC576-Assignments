import numpy as np
import matplotlib.pyplot as plt
import three_layer_neural_network as tlnn
from sklearn import datasets, linear_model


class Layer:
    def __init__(self, input_dim, output_dim, actFun_type, seed=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun_type = actFun_type

        np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))

    def actFun(self, z, type):
        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1/(1+np.exp(-z))
        elif type == 'relu':
            return np.maximum(0,z)

        return None

    def diff_actFun(self, z, type):
        if type == 'tanh':
            return 1 - np.power(np.tanh(z), 2)
        elif type == 'sigmoid':
            return np.exp(-z)/(1+np.exp(-z))**2
        elif type == 'relu':
            return np.where(z >= 0, 1, 0)

        return None

    def feedforward(self, X):
        self.z = X @ self.W + self.b
        self.a = self.actFun(self.z, self.actFun_type)
        self.X = X
        return self.a

    def backprop(self, delta, reg_lambda, epsilon):
        delta = delta * self.diff_actFun(self.z, self.actFun_type)
        dW = self.X.T @ delta + reg_lambda * self.W
        db = np.sum(delta, axis=0, keepdims=True)
        delta = delta @ self.W.T

        self.W += -epsilon * dW
        self.b += -epsilon * db

        return delta


class DeepNeuralNetwork(tlnn.NeuralNetwork):
    def __init__(self, nn_input_dim, nn_hidden_dims, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        super().__init__(nn_input_dim, 2, nn_output_dim, actFun_type, reg_lambda, seed)
        self.layers = []
        self.layers.append(Layer(nn_input_dim, nn_hidden_dims[0], actFun_type))
        for i in range(1, len(nn_hidden_dims)):
            self.layers.append(Layer(nn_hidden_dims[i-1], nn_hidden_dims[i], actFun_type))
        self.layers.append(Layer(nn_hidden_dims[-1], nn_output_dim, actFun_type))

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)

        exp_scores = np.exp(X)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def backprop(self, X, y, epsilon):
        delta = self.probs
        num_examples = len(X)
        delta[range(num_examples), y] -= 1

        for layer in reversed(self.layers):
            delta = layer.backprop(delta, self.reg_lambda, epsilon)

        return delta

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # cross entropy
        data_loss = -np.sum(y * np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * np.sum(np.square(self.layers[0].W))
        for i in range(1, len(self.layers)):
            data_loss += self.reg_lambda / 2 * np.sum(np.square(self.layers[i].W))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            delta = self.backprop(X, y, epsilon)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


def main():
    # generate and visualize Make-Moons dataset
    X, y = tlnn.generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    actFun_type = 'tanh'
    nn_input_dim = 2
    nn_hidden_dims = [5,5,5,5,5,5,5,5,5]
    model = DeepNeuralNetwork(nn_input_dim=nn_input_dim, nn_hidden_dims=nn_hidden_dims, nn_output_dim=2, actFun_type=actFun_type)
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)
    plt.savefig(f'deep_{actFun_type}_{nn_input_dim}_{len(nn_hidden_dims)}.png')


if __name__ == "__main__":
    main()