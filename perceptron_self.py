import numpy as np

tr_inp=np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
tr_op=np.array([[0,1,1,0]]).T

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

synaptic_weights=2*np.random.random((3,1))-1
print('Synaptic weights generated')
print(synaptic_weights)
for i in range(20000):
    inputs=tr_inp
    outputs=sigmoid(np.dot(inputs,synaptic_weights))
    
    errors=tr_op-outputs
    adjustments=errors*sigmoid_derivative(outputs)
    synaptic_weights+=np.dot(inputs.T,adjustments)


print('Synaptic weights after training')
print(synaptic_weights)
    
print('Outputs after training')
print(outputs)
                                                    