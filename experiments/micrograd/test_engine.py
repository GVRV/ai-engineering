from engine import Value

def assert_grad(value, expected_value):
    assert abs(value - expected_value) < 0.0001

# def test_neural_network_backpropagation():
#     x1 = Value(2.0)
#     x2 = Value(0.0)

#     w1 = Value(-3.0)
#     w2 = Value(1.0)

#     b = Value(6.8813735870195432)

#     x1w1 = x1 * w1
#     x2w2 = x2 * w2
#     x1w1x2w2 = x1w1 + x2w2

#     n = x1w1x2w2 + b
#     o = n.tanh()

#     o.start_backward()

#     assert_grad(o.grad, 1.0)

#     assert_grad(n.grad, 0.5)

#     assert_grad(x1w1x2w2.grad, 0.5)
#     assert_grad(b.grad, 0.5)

#     assert_grad(x1w1.grad, 0.5)
#     assert_grad(x2w2.grad, 0.5)

#     assert_grad(x1.grad, -1.5)
#     assert_grad(w1.grad, 1.0)

#     assert_grad(x2.grad, 0.5)
#     assert_grad(w2.grad, 0)

def test_neural_network_backpropagation_with_closure():
    x1 = Value(2.0)
    x2 = Value(0.0)

    w1 = Value(-3.0)
    w2 = Value(1.0)

    b = Value(6.8813735870195432)

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2

    n = x1w1x2w2 + b
    o = n.tanh()

    o.backward()

    assert_grad(o.grad, 1.0)
    assert_grad(n.grad, 0.5)
    assert_grad(x1w1x2w2.grad, 0.5)
    assert_grad(b.grad, 0.5)
    assert_grad(x1w1.grad, 0.5)
    assert_grad(x2w2.grad, 0.5)
    assert_grad(x1.grad, -1.5)
    assert_grad(w1.grad, 1.0)
    assert_grad(x2.grad, 0.5)
    assert_grad(w2.grad, 0)


def test_gradient_duplicates():
    a = Value(3.0)
    b = a + a
    b.backward()

    assert_grad(a.grad, 2.0)

    a = Value(2.0)
    b = Value(4.0)
    c = a / b

    assert_grad(c.data, 0.5)

def test_modular_tanh():
    x1 = Value(2.0)
    x2 = Value(0.0)

    w1 = Value(-3.0)
    w2 = Value(1.0)

    b = Value(6.8813735870195432)

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2

    n = x1w1x2w2 + b

    n2 = n * 2
    expn2 = n2.exp()

    numerator = expn2 - 1
    demoninator = expn2 + 1

    o = numerator / demoninator
    o.backward()

    assert_grad(o.grad, 1.0)
    assert_grad(n.grad, 0.5)
    assert_grad(x1w1x2w2.grad, 0.5)
    assert_grad(b.grad, 0.5)
    assert_grad(x1w1.grad, 0.5)
    assert_grad(x2w2.grad, 0.5)
    assert_grad(x1.grad, -1.5)
    assert_grad(w1.grad, 1.0)
    assert_grad(x2.grad, 0.5)
    assert_grad(w2.grad, 0)