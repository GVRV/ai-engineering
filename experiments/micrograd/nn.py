from engine import Value
import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inputs):
        # w * x + b
        activation = sum((weight * input for weight, input in zip(self.weights, inputs)), self.bias)
        output = activation.tanh()
        return output

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MultiLayerPerceptron:
    def __init__(self, num_inputs, list_of_num_outputs):
        size = [num_inputs] + list_of_num_outputs
        # Generate a layer for each pair of input to next output, etc
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(list_of_num_outputs))]

    def __call__(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
