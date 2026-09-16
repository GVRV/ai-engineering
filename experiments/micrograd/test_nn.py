from nn import MultiLayerPerceptron

def test_loss_minimises_using_mlp():
    mlp = MultiLayerPerceptron(3, [4, 4, 1])
    assert len(mlp.parameters()) == 41

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # forward pass
    ypred = [mlp(x) for x in xs]
    loss = sum((y_out - y_ground_truth)**2 for y_ground_truth, y_out in zip(ys, ypred))
    loss.backward()

    # Check that parameters have a gradient now
    assert mlp.layers[0].neurons[0].weights[0].grad != 0

    # Nudge parameters to minimise loss
    for p in mlp.parameters():
        p.data += -0.01 * p.grad

    # another forward pass
    ypred = [mlp(x) for x in xs]
    new_loss = sum((y_out - y_ground_truth)**2 for y_ground_truth, y_out in zip(ys, ypred))
    assert new_loss.data < loss.data


def test_loss_minimization_loop():
    mlp = MultiLayerPerceptron(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # initial loss
    ypred = [mlp(x) for x in xs]
    initial_loss = sum((y_out - y_ground_truth)**2 for y_ground_truth, y_out in zip(ys, ypred))

    # loss minimization loop
    for _ in range(100):
        # forward pass
        ypred = [mlp(x) for x in xs]
        loss = sum((y_out - y_ground_truth)**2 for y_ground_truth, y_out in zip(ys, ypred))

        # backward pass (with zero grad!!!)
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()

        # Nudge parameters to minimise loss
        for p in mlp.parameters():
            p.data += -0.05 * p.grad

    ypred = [mlp(x) for x in xs]
    final_loss = sum((y_out - y_ground_truth)**2 for y_ground_truth, y_out in zip(ys, ypred))
    print(ypred)
    print(initial_loss)
    print(final_loss)
    assert final_loss.data < initial_loss.data