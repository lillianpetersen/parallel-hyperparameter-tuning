import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset 
import torch.optim as optim

import base
import classes

############################################
# train model, given hyperparameters
############################################

def train_model(model_class, X_train, X_val, y_train, y_val, device, lr, size, batch_size, modeldir, save_descript, hidden_dim=None, dropout=None):
	'''
	Inputs:
		model_class: either classes.Linear or classes.OneHidden
		X_train: numpy array (N x size)
		X_val: numpy array (N x size)
		y_train: numpy array (N) (this code is made for regression)
		y_val: numpy array (N)
		device: str, eg 'cuda:0'
		lr: learning rate
		size: number of features in X
		batch_size: int, batch size
		model_dir: str, directory to save best model to
		save_descript: str, name to save best model under
		hidden_dim: int, hidden dimension of OneHidden
		dropout: float between 0 and 1. dropout percentage of OneHidden
	No output, just saves best model
	'''

	# data
	train_data = classes.RegularDataset(X_train, y_train)
	val_data = classes.RegularDataset(X_val, y_val)
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
	
	if model_class==classes.Linear:
		model = model_class(size).to(device)
	elif model_class==classes.OneHidden:
		model = model_class(size, hidden_dim, dropout).to(device)

	optimizer = optim.Adam(model.parameters(), lr=lr)

	val_loss_curve = []
	train_loss_curve = []

	best_loss = 3
	best_epoch = 0
	epoch = -1
	while(True):
		epoch += 1
		# Train model on training data
		epoch_loss = base.train(model, train_dataloader, optimizer, device=device)
		
		# Validate on validation data 
		val_loss = base.validate(model, val_dataloader, device=device) 
		
		# Record train and loss performance 
		train_loss_curve.append(epoch_loss)
		val_loss_curve.append(val_loss)

		if val_loss < best_loss:
			best_loss = val_loss
			torch.save(model.state_dict(), modeldir+'/best_'+save_descript+'.pt')
			best_epoch = epoch
			print(best_epoch, best_loss)
		
		if epoch > best_epoch+50: break

############################################
# apply trained model to val set
############################################

def apply_model(model_class, X_test, y_test, device, size, batch_size, modeldir, figdir, save_descript, title, hidden_dim=None, dropout=None):
	'''
	Inputs:
		model_class: either classes.Linear or classes.OneHidden
		X_test: numpy array (N x size)
		y_train: numpy array (N) (this code is made for regression)
		device: str, eg 'cuda:0'
		size: number of features in X
		batch_size: int, batch size
		model_dir: str, directory to save best model to
		save_descript: str, name to save best model under
		title: title of plot
		hidden_dim: int, hidden dimension of OneHidden
		dropout: float between 0 and 1. dropout percentage of OneHidden
	No output, just saves best model
	'''

	# data
	test_data = classes.RegularDataset(X_test, y_test)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
	
	if model_class==classes.Linear:
		model = model_class(size).to(device)
	elif model_class==classes.OneHidden:
		model = model_class(size, hidden_dim, dropout).to(device)
	else:
		raise Exception("Unsupported model type")

	model.load_state_dict(torch.load(modeldir+'/best_'+save_descript+'.pt'))
	model.eval()

	model = model.to(device)
	preds = []
	y_true = []
	with torch.no_grad(): 
		for batch in test_dataloader:
			X_batch, y_batch = batch
			X_batch = X_batch.to(device)
			if GRU:
				h = model.init_hidden(X_batch.shape[0])
				h = h.data
				y_pred, h = model(X_batch, h)
			else:
				y_pred = model(X_batch)
			preds.extend(y_pred.tolist())
			y_true.extend(y_batch.tolist())

	rmse_test, corr_test = benchmark.benchmark(np.array(y_true), np.array(preds), title, save_descript, figdir, test_within_group=test_within_group, groups=groups)

	return rmse_test, corr_test


