import numpy as np

class NeuralNetwork(object):
    
    def __init__(self):
        self.input = 2
        self.hidden = 3
        self.output = 1
       
        self.W1 = np.random.randn(self.input, self.hidden)
        self.W2 = np.random.randn(self.hidden, self.output)
        
    def forwardPropagation(self, X):
        self.z = np.dot(X, self.W1)
        self.z1 = self.sigmoid(self.z)
        self.z2 = np.dot(self.z1, self.W2)
        result = self.sigmoid(self.z2)
        
        return result
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
 
network = NeuralNetwork()

X = np.random.randn(3, 2)
Y = np.array([[0.75], [0.82], [0.93]])
 
print(network.forwardPropagation(X))