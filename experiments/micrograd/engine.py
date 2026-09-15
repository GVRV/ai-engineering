import math

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}|grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data + other.data,
            _children=(self, other),
            _op='+'
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data * other.data,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        out = Value(
            self.data ** other,
            _children=(self, ),
            _op=f'**{other}'
        )

        def _backward():
            self.grad = other * (self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(
            math.exp(self.data),
            _children=(self,),
            _op='exp'
        )

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        data = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(
            data,
            _children=(self,),
            _op='tanh'
        )

        def _backward():
            self.grad += (1 - (data ** 2)) * out.grad
        out._backward = _backward

        return out

    # def different_backward(self):
    #     if self._op != '':
    #         if self._op == '+':
    #             child1, child2 = self._prev
    #             child1.grad = self.grad
    #             child2.grad = self.grad
    #         if self._op == '*':
    #             child1, child2 = self._prev
    #             child1.grad = self.grad * child2.data
    #             child2.grad = self.grad * child1.data
    #         if self._op == 'tanh':
    #             for child in self._prev:
    #                 child.grad = 1 - (self.data ** 2)

    #     for child in self._prev:
    #         child.different_backward()

    # def start_backward(self):
    #     self.grad = 1.0
    #     self.different_backward()

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# a = Value(2.0)
# print(a)

# b = Value(-3.0)
# print(b)

# c = Value(10.0)
# print(c)

# d = a * b + c
# print(d)
# print(d._prev)
# print(d._op)

# f = Value(-2.0)
# L = d * f
# print(L)

# L.backward()
# print(a.grad)
# print(b.grad)
# print(c.grad)
# print(d.grad)
# print(f.grad)
# print(L.grad)