import torch
import torch.nn as nn


class MultiplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, use_activation=True,layers=2):
        for _ in range(layers-1):
            x = self.fc(x)
            if use_activation:
                x = self.activation(x)
        x = self.fc(x)
        return x
            
        

if __name__ == "__main__":
    model = MultiplePerceptron(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass
