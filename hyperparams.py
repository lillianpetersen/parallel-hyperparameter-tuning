import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import classes
import base

############################################
# hyperparameter search
############################################

def get_num_gpus():
	"""Returns the number of GPUs available"""
	from pycuda import driver
	driver.init()
	num_gpus = driver.Device.count()
	return num_gpus

class MultiGPUObjective:
	def __init__(self, gpu_queue, model_class, X_train, X_val, y_train, y_val, batch_size, size):
		# Shared queue to manage GPU IDs.
		self.gpu_queue = gpu_queue
		self.model_class = model_class
		self.X_train = X_train
		self.X_val = X_val
		self.y_train = y_train
		self.y_val = y_val
		self.size = size
		self.batch_size = batch_size

	def __call__(self, trial):
		# Fetch GPU ID for this trial.
		gpu_id = self.gpu_queue.get()

		# objective function
		loss = ml.objective(trial, self.model_class, self.X_train, self.X_val, self.y_train, self.y_val, device='cuda:'+str(gpu_id), size=self.size, batch_size=self.batch_size)

		# Return GPU ID to the queue.
		self.gpu_queue.put(gpu_id)

		# GPU ID is stored as an objective value.
		return loss


def objective(trial, model_class, X_train, X_val, y_train, y_val, device, size, batch_size=None):
	# Generate the optimizers.
	if batch_size is None:
		batch_size = trial.suggest_categorical("batch_size", [16,32,64])
	if model_class==classes.OneHidden:
		lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
		hidden_dim = trial.suggest_int("hidden_dim", 256, 2048)
		dropout = trial.suggest_float("dropout", 0.2, 0.5)
		# define model
		model = model_class(size, hidden_dim, dropout).to(device)
	elif model_class==classes.Linear:
		lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
		model = model_class(size).to(device)
	else: raise Exception("Unsupported model class")

	optimizer = optim.Adam(model.parameters(), lr=lr)

	# data
	train_data = classes.RegularDataset(X_train, y_train)
	val_data = classes.RegularDataset(X_val, y_val)
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

	val_loss_curve = []
	train_loss_curve = []

	for epoch in range(80):
		# Train
		epoch_loss = base.train(model, train_dataloader, optimizer, device=device)
		# Validate
		val_loss = base.validate(model, val_dataloader, device=device) 

		# Record train and loss performance 
		train_loss_curve.append(epoch_loss)
		val_loss_curve.append(val_loss)

		trial.report(np.amin(val_loss), epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			return 2

	return np.amin(val_loss)

