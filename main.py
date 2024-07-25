import numpy as np
from scipy.fftpack import ifft
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Fractional Fourier Transform (FrFT) function
def frft(f, a):
    N = len(f)
    shft = np.remainder((np.arange(N) + N // 2), N)
    x = np.zeros(N, dtype=complex)
    x[shft] = f
    f = x
    alpha = a * np.pi / 2
    f = np.concatenate((f, np.zeros(N % 2 * N)))
    N = len(f)
    n = np.arange(N)
    k = n
    E = np.exp(-1j * alpha / 2 * n ** 2)
    E2 = np.exp(1j * alpha / 2 * k ** 2)
    y = E2 * ifft(E * f * E)
    y = y[:N//2 + N % 2 * N]
    y[shft[:N//2]] = y[:N//2]
    return y

# Discrete Fractional Fourier Transform function
def dfrft(x, a):
    N, M = x.shape
    F = np.zeros((N, M), dtype=complex)
    for i in range(N):
        F[i, :] = frft(x[i, :], a)
    return F

# Neural Network Model Definition
class FractionalOrderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FractionalOrderModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Bayesian model definition for Pyro
def model(data, labels):
    fc1w_prior = dist.Normal(torch.zeros_like(model.fc1.weight), torch.ones_like(model.fc1.weight))
    fc1b_prior = dist.Normal(torch.zeros_like(model.fc1.bias), torch.ones_like(model.fc1.bias))
    fc2w_prior = dist.Normal(torch.zeros_like(model.fc2.weight), torch.ones_like(model.fc2.weight))
    fc2b_prior = dist.Normal(torch.zeros_like(model.fc2.bias), torch.ones_like(model.fc2.bias))
    fc3w_prior = dist.Normal(torch.zeros_like(model.fc3.weight), torch.ones_like(model.fc3.weight))
    fc3b_prior = dist.Normal(torch.zeros_like(model.fc3.bias), torch.ones_like(model.fc3.bias))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
              'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior}
    
    lifted_module = pyro.random_module("module", model, priors)
    lifted_reg_model = lifted_module()
    
    with pyro.plate("map", len(data)):
        prediction = lifted_reg_model(data)
        pyro.sample("obs", dist.Normal(prediction, 0.1), obs=labels)

# Guide function for Bayesian inference
def guide(data, labels):
    fc1w_mu = torch.randn_like(model.fc1.weight)
    fc1w_sigma = torch.randn_like(model.fc1.weight)
    fc1b_mu = torch.randn_like(model.fc1.bias)
    fc1b_sigma = torch.randn_like(model.fc1.bias)
    
    fc2w_mu = torch.randn_like(model.fc2.weight)
    fc2w_sigma = torch.randn_like(model.fc2.weight)
    fc2b_mu = torch.randn_like(model.fc2.bias)
    fc2b_sigma = torch.randn_like(model.fc2.bias)
    
    fc3w_mu = torch.randn_like(model.fc3.weight)
    fc3w_sigma = torch.randn_like(model.fc3.weight)
    fc3b_mu = torch.randn_like(model.fc3.bias)
    fc3b_sigma = torch.randn_like(model.fc3.bias)
    
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = pyro.param("fc1w_sigma", fc1w_sigma)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = pyro.param("fc1b_sigma", fc1b_sigma)
    
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = pyro.param("fc2w_sigma", fc2w_sigma)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = pyro.param("fc2b_sigma", fc2b_sigma)
    
    fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    fc3w_sigma_param = pyro.param("fc3w_sigma", fc3w_sigma)
    fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    fc3b_sigma_param = pyro.param("fc3b_sigma", fc3b_sigma)
    
    fc1w_prior = dist.Normal(fc1w_mu_param, torch.exp(fc1w_sigma_param))
    fc1b_prior = dist.Normal(fc1b_mu_param, torch.exp(fc1b_sigma_param))
    fc2w_prior = dist.Normal(fc2w_mu_param, torch.exp(fc2w_sigma_param))
    fc2b_prior = dist.Normal(fc2b_mu_param, torch.exp(fc2b_sigma_param))
    fc3w_prior = dist.Normal(fc3w_mu_param, torch.exp(fc3w_sigma_param))
    fc3b_prior = dist.Normal(fc3b_mu_param, torch.exp(fc3b_sigma_param))
    
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
              'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior}
    
    lifted_module = pyro.random_module("module", model, priors)
    return lifted_module()

# Evaluation functions
def accuracy(preds, labels):
    return accuracy_score(labels, preds.round())

def recall(preds, labels):
    return recall_score(labels, preds.round(), average='macro')

def dice_coefficient(preds, labels):
    preds = preds.round()
    intersection = (preds * labels).sum()
    return (2. * intersection) / (preds.sum() + labels.sum())

# Example training function
def train(data, labels, num_iterations=1000):
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for j in range(num_iterations):
        loss = svi.step(data, labels)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j, loss))

# Example data
input_size = 10
hidden_size = 50
output_size = 1

# Initialize the model
model = FractionalOrderModel(input_size, hidden_size, output_size)

# Example data for evaluation
data = torch.randn(100, input_size)  # 100 samples with input_size features
labels = torch.randint(0, 2, (100, output_size)).float()  # 100 labels (binary classification)

# Train the model
train(data, labels)

# Get predictions
model.eval()
with torch.no_grad():
    preds = model(data)

# Evaluate the model
acc = accuracy(preds, labels)
rec = recall(preds, labels)
dice = dice_coefficient(preds, labels)

print(f"Accuracy: {acc:.4f}")
print(f"Recall: {rec:.4f}")
print(f"Dice Coefficient: {dice:.4f}")

# Execute the fractional-order transformation and visualization
def main():
    # Load an image in grayscale
    x = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    # Define the fractional orders to apply
    alpha = [0.8, 1.2]

    # Display the original image
    plt.figure()
    plt.imshow(x, cmap='gray')
    plt.title("Original Image")
    plt.show()

    # Apply the fractional-order transformation and display the results
    for a in alpha:
        temp = np.uint8(np.abs(dfrft(np.double(x), a)))
        plt.figure()
        plt.imshow(temp, cmap='gray')
        plt.title(f"Fractional Order Transform with alpha={a}")
        plt.show()

if __name__ == "__main__":
    main()
