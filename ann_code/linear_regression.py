"""THWS/MAI/ANN - Assignment 1 - linear regression

Created: Magda Gregorova, 9/5/2024
"""


import torch


def linear_single_forward(x, w, b):
	"""Linear model for single input - forward pass (naive implementation with for loops).

	Args:
	x: torch.tensor of shape (d) - input instance
	w: torch.tensor of shape (d) - weight vector
	b: float - bias

	Returns:
	out: torch.tensor of shape (1) - output of linear transform
	cache: tuple (x, w, b)
	"""

	# forward pass - compute predictions iteratively
	num_dims = x.shape[0]

	out = torch.zeros(1)
	for i in range(num_dims):
		out += x[i] * w[i]
	out += b

	cache = (x, w, b)
	return out, cache


def squared_error_forward(y_pred, y):
	"""Squared error loss - forward pass.

	Args:
	y_pred: torch tensor of shape (1) - prediction
	y: torch tensor of shape (1) - true label

	Returns:
	loss: torch.tensor of shape (1) - squared error loss
	cache: tuple (y_pred, y)
	"""

	# forward pass
	loss = (y_pred - y)**2

	cache = (y_pred, y)
	return loss, cache


def linear_single_lgrad(cache):
	"""Linear model for single input - local gradient (naive implementation with for loops).

	Args:
	cache: tuple (x, w, b)
		x: torch.tensor of shape (d) - input instance
		w: torch.tensor of shape (d) - weight vector
		b: float containing bias

	Returns:
	xg: torch.tensor of shape (d) - local gradient with respect to input
	wg: torch.tensor of shape (d) - local gradient with respect to input weight vector
	bg: float - local gradient with respect to bias
	"""

	x, w, b = cache
	# xg = torch.zeros_like(x)
	# wg = torch.zeros_like(w)
	# bg = torch.zeros_like(b)

	# compute local gradients iteratively
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the local gradient calculation.                            #	
	################################################################################

	xg = w
	wg = x
	bg = torch.ones_like(b)

	#pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################

	return xg, wg, bg


def squared_error_lgrad(cache):
	"""Squared error loss - local gradient.

	Args:
	cache: tuple (y_pred, y)
		y_pred: torch tensor of shape (1) - prediction
		y: torch tensor of shape (1) - true label

	Returns:
	y_predg: torch tensor of shape (1) - local gradient with respect to y_pred
	yg: torch tensor of shape (1) - local gradient with respect to y
	"""

	y_pred, y = cache

	# compute local gradients
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the local gradient calculation.                            #	
	################################################################################

	yg = 2 * (y - y_pred)
	y_predg = -yg

	#pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################

	return y_predg, yg


def linear_single_ggrad(cache, gout):
	"""Linear model for single input - global gradient.

	Args:
	cache: tuple (xg, wg, bg)
		xg: torch.tensor of shape (d) - local gradient with respect to input
		wg: torch.tensor of shape (d) - local gradient with respect to input weight vector
		bg: float - local gradient with respect to bias
	gout: torch.tensor of shape (1) - upstream global gradient

	Returns:
	xgrad: torch.tensor of shape (d) - global gradient with respect to input
	wgrad: torch.tensor of shape (d) - global gradient with respect to input weight vector
	bgrad: float - global gradient with respect to bias
	"""

	xg, wg, bg = cache

	# compute global gradients
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the global gradient calculation.                           #	
	################################################################################

	xgrad = gout * xg
	wgrad = gout * wg
	bgrad = gout * bg


	#pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################
	
	return xgrad, wgrad, bgrad


def linear_forward(X, w, b):
	"""Linear model - forward pass.

	Args:
	X: torch.tensor of shape (n, d) - input instances
	w: torch.tensor of shape (d, 1) - weight vector
	b: float - bias

	Returns:
	out: torch.tensor of shape (n, 1) - outputs of linear transform
	cache: tuple (X, w, b)
	"""

	# forward pass
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the forward pass.										   #	
	################################################################################
	pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################
	return out, cache


def mse_forward(y_pred, y):
	"""MSE loss - forward pass.

	Args:
	y_pred: torch tensor of shape (n, 1) - prediction
	y: torch tensor of shape (n, 1) - true label

	Returns:
	loss: torch.tensor of shape (1) - squared error loss
	cache: tuple (y_pred, y)
	"""

	# forward pass
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the forward pass.										   #	
	################################################################################
	pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################
	return loss, cache


def linear_backward(cache, gout):
	"""Linear model - backward pass.

	Args:
	cache: tuple (X, w, b)
		X: torch.tensor of shape (n, d) - input instances
		w: torch.tensor of shape (d, 1) - weight vector
		b: float - bias
	gout: torch.tensor of shape (n, 1) - upstream global gradient

	Returns:
	xgrad: torch.tensor of shape (n, d) - global gradient with respect to input
	wgrad: torch.tensor of shape (d, 1) - global gradient with respect to input weight vector
	bgrad: float - global gradient with respect to bias
	"""

	# forward pass
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the forward pass.										   #	
	################################################################################
	pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################
	return Xgrad, wgrad, bgrad


def mse_backward(cache):
	"""MSE loss - backward pass.

	Args:
	cache: tuple (y_pred, y)
		y_pred: torch tensor of shape (n, 1) - prediction
		y: torch tensor of shape (n, 1) - true label

	Returns:
	y_predgrad: torch tensor of shape (n, 1) - global gradient with respect to y_pred
	ygrad: torch tensor of shape (n, 1) - global gradient with respect to y
	"""

	# backward pass
	################################################################################
	### START OF YOUR CODE                                                         #
	### TODO: implement the backward pass.										   #	
	################################################################################
	pass
	################################################################################
	### END OF YOUR CODE                                                           #
	################################################################################
	return y_predgrad, ygrad

