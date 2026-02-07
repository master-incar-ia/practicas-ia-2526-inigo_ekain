import torch
import torch.nn as nn


class MultiplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, dim1, dim2):
        super().__init__()
        self.fc_input = nn.Linear(input_dim, dim1)
        self.fc=nn.Linear(dim1, dim2)
        self.fc_output = nn.Sigmoid(dim2, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, use_activation=True,layers=10):
        x = self.fc_input(x)
        if use_activation:
                x = self.activation(x)
        for _ in range(layers-1):
            x = self.fc(x)
            if use_activation:
                x = self.activation(x)
        x = self.fc_output(x)
        return x
            
        

if __name__ == "__main__":
    model = MultiplePerceptron(1, 1, 10, 10)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass
