import numpy as np

class NeuralNetwork(object):
    
    def __init__(self):
        self.input = 2
        self.hidden = 3
        self.output = 1
        self.W1 = np.random.random((self.input, self.hidden))
        self.W2 = np.random.random((self.hidden, self.output))
        self.alpha = 0.01
        
    def forwardPropagation(self, X):
        self.z = np.dot(X, self.W1)
        self.z1 = self.sigmoid(self.z)
        self.z2 = np.dot(self.z1, self.W2)
        
        return self.sigmoid(self.z2)
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def dSigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def backwardPropagation(self, X, y):
        print(".............. Initiating Backpropagation ..............")
        for i in range(60000):
            prediction = self.forwardPropagation(X)
        
            if(i % 2500) == 0:
                m = (np.subtract(prediction, y)).shape[0]
                minus = np.subtract(prediction, y)
                squared = np.square(minus)
                print("Error:", np.sum(squared) / m)
                
            delta2 = np.multiply(-(y - prediction), self.dSigmoid(self.z2))
            dJdW2 = np.dot(self.z1.T, delta2)
            
            delta1 = np.dot(delta2, self.W2.T) * self.dSigmoid(self.z)
            dJdW1 = np.dot(X.T, delta1)
        
            self.W2 -= self.alpha * dJdW2
            self.W1 -= self.alpha * dJdW1
            
        print("---------------- After ----------------\n", prediction) 
        
network = NeuralNetwork()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("---------------- Before ----------------\n", network.forwardPropagation(X))
network.backwardPropagation(X, y)