from macrograd.engine import Value

# build a expression
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label="f")
L = d * f; L.label = 'L'

L.zero_grad()
L.backward()

# can visualize the expression graph using notebook