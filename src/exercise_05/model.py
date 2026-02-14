import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, output_dim, hidden_layers=5):
        super().__init__()
        self.linear1 = nn.Linear(3 * 32 * 32, 512)
        self.BatchNorm1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.BatchNorm2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.BatchNorm3 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        # Capa 1
        x = self.linear1(x)
        x = self.activation(x)
        x = self.BatchNorm1(x)
        x = self.dropout(x)
        # Capa 2
        x = self.linear2(x)
        x = self.activation(x)
        x = self.BatchNorm2(x)
        x = self.dropout(x)
        # Capa 3
        x = self.linear3(x)
        x = self.activation(x)
        x = self.BatchNorm3(x)
        x = self.dropout(x)
        # Capa de salida
        x = self.output_layer(x)
        return x


# Definimos el modelo VGG_16, siguiendo la página 54 de la diapositiva "Convolutional Neural Networks (CNNs)".
class VGG(nn.Module):
    def __init__(self, output_dim=1000):
        super().__init__()

        # 1º Bloque de VGG:
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2º Bloque de VGG:
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3º Bloque de VGG:
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4º Bloque del VGG:
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5º Blque del VGG:
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu13 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #  Ultima capa del VGG:
        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(7 * 7 * 512, 4096)
        self.relu14 = nn.ReLU()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu15 = nn.ReLU()
        self.fc8 = nn.Linear(4096, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Bloque 1
        x1 = self.conv1_1(x)
        x2 = self.relu1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu2(x3)
        x5 = self.pool1(x4)

        # Bloque 2
        x6 = self.conv2_1(x5)
        x7 = self.relu3(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu4(x8)
        x10 = self.pool2(x9)

        # Bloque 3
        x11 = self.conv3_1(x10)
        x12 = self.relu5(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu6(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu7(x15)
        x17 = self.pool3(x16)

        # Bloque 4
        x18 = self.conv4_1(x17)
        x19 = self.relu8(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu9(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu10(x22)
        x24 = self.pool4(x23)

        # Bloque 5
        x25 = self.conv5_1(x24)
        x26 = self.relu11(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu12(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu13(x29)
        x31 = self.pool5(x30)

        # Clasificador
        x32 = self.flatten(x31)
        x33 = self.fc6(x32)
        x34 = self.relu14(x33)
        x35 = self.fc7(x34)
        x36 = self.relu15(x35)
        x37 = self.fc8(x36)
        x38 = self.softmax(x37)

        return x38


if __name__ == "__main__":
    model = VGG(output_dim=10)
    print(model)
    x = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10
    print(model(x))
    pass
