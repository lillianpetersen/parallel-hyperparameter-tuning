import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

############################################
# train and validate
############################################

def train(model, dataloader, optimizer, device):
	epoch_loss = []
	model.train() # Set model to training mode 
	model = model.to(device)
	
	for batch in dataloader:	
		X, y = batch
		X = X.to(device)
		y = y.to(device)

		y_pred = model(X)

		loss = nn.functional.mse_loss(y_pred, y)
		epoch_loss.append(loss.item())
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return np.array(epoch_loss).mean()


def validate(model, dataloader, device):
	val_loss = []
	model.eval() # Set model to evaluation mode 
	with torch.no_grad():	
		for batch in dataloader:
			X, y = batch
			X = X.to(device)
			y = y.to(device)
			
			y_pred = model(X)

			loss = nn.functional.mse_loss(y_pred, y)
			val_loss.append(loss.item())
			
	return np.array(val_loss).mean()

