import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
		
	gradient = torch.as_tensor(gradient)
	magnitude = torch.linalg.norm(gradient)
	if magnitude == 0:
		direction = torch.zeros_like(gradient)
	else:
		direction = gradient / magnitude

	return {
		'magnitude': magnitude,
		'direction': direction,
		'descent_direction': -direction,
	}

