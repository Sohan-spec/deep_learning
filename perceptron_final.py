import numpy as np

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

training_inputs=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_outputs=np.array([[0,1,1,0]]).T #.T means transpose of a matrix

np.random.seed(1)

synaptic_weights= 2*np.random.random((3,1))-1 #creates a matrix of order 3x1, fills it with values in b/w 0 and 1, then subtracts 1 from it such that the values are in bw -1 and 1

print('Random starting synaptic weights:')
print(synaptic_weights)

for i in range(20000):
    input_layer=training_inputs
    outputs=sigmoid(np.dot(input_layer,synaptic_weights))
    error=training_outputs-outputs
    
    adjustments=error*sigmoid_derivative(outputs)
    
    synaptic_weights+=np.dot(input_layer.T,adjustments)

print('Synaptic weights after training')
print(synaptic_weights)
    
print('Outputs after training')
print(outputs)