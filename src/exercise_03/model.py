import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        n_neuron1 = 256
        n_neuron2 = 128
        n_neuron3 = 64

        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_neuron1)
        self.activation1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(n_neuron1, n_neuron2)
        self.activation2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(n_neuron2, n_neuron3)
        self.activation3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(n_neuron3, output_dim)
        self.activation4 = nn.Identity()  # Identity activation for output layer

    def forward(self, x, use_activation=True):
        x1 = self.fc1(x)
        x2 = self.activation1(x1)
        x3 = self.fc2(x2)
        x4 = self.activation2(x3)
        x5 = self.fc3(x4)
        x6 = self.activation3(x5)
        x7 = self.fc4(x6)
        x8 = self.activation4(x7)
        return x8


if __name__ == "__main__":
    model = MultiLayerPerceptron(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass
