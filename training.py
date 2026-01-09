

from mlp import MLP 
import time 
from drwa import draw_dot

# 4 examples of input data
xs = []

# Desired targets
ys = []

import pandas as pd
data = pd.read_csv("data/Iris.csv")

print(data.head())

label_map = {
    "Iris-setosa": -1.0,
    "Iris-versicolor": 1.0,
}

 

with open("data/Iris.csv") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split(',')
        *features, label = parts

        if label not in label_map:
            continue  # skip virginica

        xs.append([float(v) for v in features])
        ys.append(label_map[label])



 

 
n = MLP(4, [4,1])

learning_rate = 0.01
n_epochs = 5

for epoch in range(iterations := n_epochs):
    ypred = [n(x) for x in xs]  # Forward pass: compute predictions
#     print(ypred)
    loss = sum((y_p - y)**2 for y_p, y in zip(ypred, ys)) / len(ys)  # Mean Squared Error

    for p in n.parameters():
        p._grad = 0.0  # Reset gradients to zero

    loss.backward()  # Backward pass: compute gradients
    
    for p in n.parameters():
        p.weight -= learning_rate * p._grad  # Update parameters
    
    print(f"Epoch {epoch+1}, Loss: {loss.weight}")
    # time.sleep(0.1)  # Sleep to simulate time taken for training




 

# print("Final predictions after training:")
# for x in xs:
#     print(n(x).weight)


    
