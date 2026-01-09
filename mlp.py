"""
This module implements a simple neural network using a multi-layer perceptron (MLP) architecture.
"""

import random 
from autogrand_fundamentals import Value 


class Neuron:
    def __init__(self, num_inputs): 
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1)) 

    def __call__(self, x):   # x is a list of values from the previous layer
        # w * x + b  Weighted sum of inputs plus bias 
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) 
        # Apply activation function (tanh)
        out = act.tanh() 
        return out  
    
    def parameters(self):
        return self.w + [self.b] 
    

class Layer:
    def __init__(self, no_inputs, no_outputs):
        # no_inputs: number of inputs to this layer
        # no_outputs: number of neurons in this layer
        self.neurons = [Neuron(no_inputs) for _ in range(no_outputs)]

    def __call__(self, x):
        """
                ┌─ neuron 1 ─┐
        x ──────┼─ neuron 2 ─┼─ outputs
                └─ neuron k ─┘

        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    



class MLP:  
    def __init__(self, nin, nout):
        self.sz = [nin] + nout  # e.g., [3,4,4,1] 
        self.layers = [Layer(self.sz[i], self.sz[i+1]) for i in range(len(nout))]

    def __call__(self, x):
        """
        f(x)=Ln​(Ln−1​(...L1​(x)))
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



if __name__ == "__main__":
    import pandas as pd 
    data = pd.read_csv("data/tra")
    mlp = MLP(3, [4,4,1])
    inputs = [Value(1.0), Value(2.0), Value(3.0)]
    output = mlp(inputs)
    print(output)
    print(len(mlp.parameters()))  # 4*3 + 4 + 4*4 + 4 + 4*1 + 1 = 41
  




