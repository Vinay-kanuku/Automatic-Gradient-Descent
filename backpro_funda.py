
"""
Backpropagation is the recursive allplication of the chain rule to compute gradients for each node in the computational graph.
"""



def function(x):
    return x**2 + 2*x + 10 

a = 10 # Initial input 
h = 0.1  # small nudge to the input
out1 = function(a) 
out2 = function(a+h)  # how much the output changes with a small change in input 
print((out2-out1)/h)   # normalized change in output with respect to change in input 


a = 10 
b = 5 
h = 1e-4   # this is typically safe value to chose for proper approximation 

print(f"Original value:{a*b + 1}")
d1 = a*b + 1 
b += h  # nudge a by a small amount 
d2 = a*b + 1
print((d2-d1)/h) 
print(f"Numerical gradient with respect to a: {(d2-d1)/h}")
print(a*b + 1)
print(10*h + a*b+1)

"""If I increase a by a tiny amount, how much does the output change?”

        Here:

        Increase a by h

        Output increases by 5h

        So the sensitivity is 5

"""