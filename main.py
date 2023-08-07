import pandas as pd
import numpy as np

import optuna
from multiprocessing import Manager
from joblib import parallel_backend

import hyperparams
import train
import classes

## For you to define
X_train = None
X_val = None
X_test = None
y_train = None
y_val = None
y_test = None
save_descript = None
title = None

#### Linear Head
device = 'cuda:0'
size = X_train.shape[1]
batch_size=64

print('Optimizing hyperparameters...')
study = optuna.create_study(direction="minimize")
n_gpu = get_num_gpus()
print('GPUs:', n_gpu)
if n_gpu>1:
	with Manager() as manager:
		gpu_queue = manager.Queue()
		for i in range(n_gpu):
			gpu_queue.put(i)
		with parallel_backend("multiprocessing", n_jobs=n_gpu):
			study.optimize(hyperparams.MultiGPUObjective(gpu_queue, classes.Linear, X_train, X_val, y_train, y_val, size=size, batch_size=batch_size), n_trials=50, n_jobs=n_gpu)
else:
	func = lambda trial: hyperparams.objective(trial, classes.Linear, X_train, X_val, y_train, y_val, device=device, size=size, batch_size=batch_size)
	study.optimize(func, n_trials=50)

## Re-train using best hyperparameters
print('Training using best hyperparameters')
params_df = pd.DataFrame(best_trial.params, index=[1])
params_df.to_csv(modeldir+'/best_linear_'+save_descript+'.csv')

train.train_model(classes.Linear, X_train, X_val, y_train, y_val, 
		device=device, size=size, lr=best_trial.params['lr'], batch_size=batch_size, 
		save_descript=save_descript, modeldir=modeldir)
rmse_linear, corr_linear = train.apply_model(classes.Linear, X_test, y_test, 
		device=device, size=size, batch_size=batch_size,
		save_descript=save_descript, modeldir=modeldir, title=title, figdir=figdir)


### OneHidden Head
print('\nOneHidden on '+model_descript)
print('Optimizing hyperparameters...')
study = optuna.create_study(direction="minimize")
n_gpu = get_num_gpus()
print('GPUs:', n_gpu)
if n_gpu>1:
	with Manager() as manager:
		gpu_queue = manager.Queue()
		for i in range(n_gpu):
			gpu_queue.put(i)
		with parallel_backend("multiprocessing", n_jobs=n_gpu):
			study.optimize(hyperparams.MultiGPUObjective(gpu_queue, classes.OneHidden, X_train, X_val, y_train, y_val, size=size, batch_size=batch_size), n_trials=100, n_jobs=n_gpu)
else:
	func = lambda trial: hyperparams.objective(trial, classes.OneHidden, X_train, X_val, y_train, y_val, device=device, size=size, batch_size=batch_size)
	study.optimize(func, n_trials=100)

## Re-train using best hyperparameters
print('Training using best hyperparameters')
best_trial = study.best_trial
print(best_trial.params)
params_df = pd.DataFrame(best_trial.params, index=[1])
params_df.to_csv(modeldir+'/best_onehidden_'+save_descript+'.csv')

train.train_model(classes.OneHidden, X_train, X_val, y_train, y_val, 
		device=device, size=size, lr=best_trial.params['lr'], batch_size=batch_size, 
		hidden_dim=best_trial.params['hidden_dim'], dropout=best_trial.params['dropout'],
		save_descript=save_descript, modeldir=modeldir)
rmse, corr = train.apply_model(classes.OneHidden, X_test, y_test, 
		device=device, size=size, batch_size=batch_size,
		hidden_dim=best_trial.params['hidden_dim'], 
		dropout=best_trial.params['dropout'], title=title
		save_descript=save_descript, modeldir=modeldir, figdir=figdir)

