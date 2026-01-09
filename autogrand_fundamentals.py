# For Deep Learning, we need the computer to remember: "This 6.0 came from multiplying a and
# Requires building a computational graph.
# We need a class that holds the weight and keeps a record of its parents.

from drwa import draw_dot 
import math 


class Value:
    def __init__(self, weight, _children=set(), _op='', lable="None"):
        self.weight = weight 
        self._parents = set(_children)
        self._grad = 0   # grad of cur node with respect to final output 
        self._op = _op
        self._backward = lambda: None
        self.lable = lable 

    def __repr__(self):
        return f"Value(weight={self.weight})" 

    def __add__(self,other):
        # check if other is a Value instance 
        other = other if isinstance(other, Value) else Value(other, lable="int") 
        # calculate the output and make a new Value instance 
        out = Value(self.weight + other.weight, (self, other), '+') 
        def __backward():
            self._grad += 1.0 * out._grad
            other._grad += 1.0 * out._grad 

        out._backward = __backward 

        return out 
    
    def __radd__(self,other):
        # for right add
        return self + other 
    
    def __sub__(self,other):
        # check if other is a Value instance 
        other = other if isinstance(other, Value) else Value(other, lable="int") 
        # calculate the output and make a new Value instance 
        out = Value(self.weight - other.weight, (self, other), '-') 
        
        def __backward():
            self._grad += 1.0 * out._grad
            other._grad += -1.0 * out._grad
        out._backward = __backward

        return out
    
    def __rsub__(self,other):
        # for right sub
        other = other if isinstance(other, Value) else Value(other, lable="int") 
        return other - self
    
    def __neg__(self):
        out = Value(-self.weight, (self,), 'neg')

        def _backward():
            self._grad -= out._grad

        out._backward = _backward
        return out

    

    def __mul__(self,other):
        # check if other is a Value instance 
        other = other if isinstance(other, Value) else Value(other, lable="int") 
        # calculate the output and make a new Value instance 
        out = Value(self.weight * other.weight, (self, other), '*') 
        
        def __backward():
            self._grad += other.weight * out._grad
            other._grad += self.weight * out._grad
        out._backward = __backward

        return out 
    
    def __rmul__(self,other):
        return self * other 
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    
        

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * self**-1

    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports scalar powers"
        out = Value(self.weight ** power, (self,), f'**{power}')

        def _backward():
            self._grad += power * (self.weight ** (power - 1)) * out._grad

        out._backward = _backward
        return out

    
    def __rpow__(self, base):
        # calculate the output and make a new Value instance 
        out = Value(base ** self.weight, (self,), f'**{base}') 
        
        def __backward():
            self._grad += (base ** self.weight) * math.log(base) * out._grad
        out._backward = __backward

        return out
    
    def tanh(self):
        t = math.tanh(self.weight)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self._grad += (1 - t*t) * out._grad

        out._backward = _backward
        return out

    
    def backward(self):
        topological = [] 
        visited = set() 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topological.append(v)
        build_topo(self)
        self._grad = 1.0
        for node in reversed(topological):
            node._backward()
        

# ------------- Backpropagation -----------------#
# We built the graph forward. Now we need to walk it backward to calculate the gradients. This relies on the Chain Rule from calculus.

    """" 1. The Intuition: The Chain Rule

        Imagine our expression is L = d * f. And d was created by c + e.

        So, L = (c + e) * f.

        We want to know: "If I change c, how does L change?" (∂c/∂L​)
    """
def prop():
    h = 1e-4   
    a = Value(22, lable="a")  
    b = Value(3.0, lable="b")
    c = Value(4.0, lable="c") 
    e = a * c;  e.lable = "e"
    d = (e + b); d.lable="d"
    L1 = d * a; L1.lable="L1"  
 
 
    # a = Value(22, lable="a")  
    # b = Value(3.0, lable="b")
    # c = Value(4.0 + h, lable="c") 
    # e = a * c;  e.lable = "e"
    # d = (e + b); d.lable="d"
    # L2 = d * a; L2.lable="L2"  
    
    L1.backward() 

    # print((L2.weight - L1.weight)/h)

    draw_dot(L1)
     

    
if __name__ == "__main__":

    # a = Value(22, lable="a")  
    # b = Value(3.0, lable="b")
    # c = Value(4.0, lable="c") 
    # e = a * c;  e.lable = "e"
    # d = (e + b) * a; d.lable="d"  # By running e = a * b + c, you didn't just do math. You built a graph.
    # L = d + 1 ; L.lable="L"  # L is the final output node.
    # 1. weight is the value of the node (forward pass).
    # 2. grad will be the derivative of the final output with respect to this node (backward pass)
    # 3. Right now, grad is 0 because we haven't calculated it yet. In PyTorch, this is exactly what happens when you create a tensor.

    # prop()
    a = Value(3.0, lable="a")
    b = Value(2.0, lable="b")
    c =  a / b; c.lable="c"  # b = 2^a

    draw_dot(c)

    # grad of L with respect to itsef is 1  
    # If i change L by h, it just changes by 1h  
     


  




