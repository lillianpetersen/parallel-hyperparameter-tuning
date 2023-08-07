import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset 

############################################
# Dataset
############################################

# Dataset object
class RegularDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.Tensor(np.array(X))  # store X as a pytorch Tensor
		self.y = torch.Tensor(np.array(y))  # store y as a pytorch Tensor
		self.len=len(self.X)				# number of samples in the data 

	def __getitem__(self, index):
		return self.X[index], self.y[index] # get the appropriate item

	def __len__(self):
		return self.len

############################################
# Model Classes
############################################

# simple MLP
class Linear(nn.Module):
	def __init__(self, size):
		self.size = size
		super(Linear, self).__init__()
			
		self.model = nn.Sequential(
			nn.Linear(size, 1),
			)

	def forward(self, x):
		x = self.model(x)
		return x

class OneHidden(nn.Module):
	def __init__(self, size, hidden_dim, dropout):
		self.size = size
		self.hidden_dim = hidden_dim
		self.dropout = dropout
		super(OneHidden, self).__init__()
				
		self.model = nn.Sequential(
			nn.Linear(size, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, x):
		x = self.model(x)
		return x

