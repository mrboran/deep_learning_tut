# load_data.py

import numpy as np
import matplotlib.pyplot as plt
import torch

training_data = np.load('training_data.npy', allow_pickle=True)
print(len(training_data))

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[0] for i in training_data])

plt.imshow(X[0], cmap='gray')
print(y[0])
