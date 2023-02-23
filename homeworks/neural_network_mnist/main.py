# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.input_weights = Parameter(torch.randn(d, h))
        self.layer1_weights = Parameter(torch.randn(h, k))
        self.input_bias = Parameter(torch.zeros(h))
        self.layer1_bias = Parameter(torch.zeros(k))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        output = x @ self.input_weights + self.input_bias
        output = relu(output)
        output = output @ self.layer1_weights + self.layer1_bias
        return output


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.input_weights = Parameter(torch.randn(d, h0))
        self.input_bias = Parameter(torch.zeros(h0))
        self.layer1_weights = Parameter(torch.randn(h0, h1))
        self.layer1_bias = Parameter(torch.zeros(h1))
        self.layer2_weights = Parameter(torch.randn(h1, k))
        self.layer2_bias = Parameter(torch.zeros(k))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        output = x @ self.input_weights + self.input_bias
        output = relu(output)
        output = output @ self.layer1_weights + self.layer1_bias
        output = relu(output)
        output = output @ self.layer2_weights + self.layer2_bias
        return output


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    avg_loss = []
    avg_acc = []

    while True:
        total_loss = 0
        total_correct = 0
        num_predictions = 0

        for _, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model.forward(images)
            loss = cross_entropy(input=output, target=labels)
            loss.backward()
            optimizer.step()

            output = torch.argmax(output, dim=1)
            num_predictions += len(labels)
            correct_predictions = torch.eq(output, labels)
            num_correct = torch.sum(correct_predictions).item()

            total_loss += loss.data
            total_correct += num_correct

        avg_loss.append(total_loss / num_predictions)
        avg_acc.append(total_correct / num_predictions)

        if avg_acc[-1] >= .90:
            break

    return avg_loss


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(x, y)
    test_dataset = TensorDataset(x_test, y_test)

    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset)

    f1 = F1(64, 784, 10)
    optimizer1 = Adam(f1.parameters(), lr=1e-5)

    f2 = F2(32, 32, 784, 10)
    optimizer2 = Adam(f2.parameters(), lr=1e-5)

    train_loss1 = train(f1, optimizer1, train_loader)
    train_loss2 = train(f2, optimizer2, train_loader)

    test_loss, test_acc = [], []

    epochs = torch.arange(len(train_loss1))
    plt.plot(epochs, train_loss1)
    plt.show()


if __name__ == "__main__":
    main()
