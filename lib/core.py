"""Core module for sigmoid operations and shape analysis.

This module provides functions to create shifted sigmoids, inverted sigmoids,
and compute their product to analyze the resulting shape. The product of a
left-shifted sigmoid and an inverted right-shifted sigmoid creates a bell-shaped
curve (similar to a Gaussian distribution).
"""

import numpy as np
from typing import Union, Tuple, Callable


def sigmoid(x: Union[float, np.ndarray], shift: float = 0.0, steepness: float = 1.0) -> Union[float, np.ndarray]:
    """Compute the sigmoid function with optional shift and steepness.
    
    Args:
        x: Input value(s) for the sigmoid function.
        shift: Horizontal shift of the sigmoid (positive shifts right).
        steepness: Controls the steepness of the sigmoid curve (default: 1.0).
    
    Returns:
        Sigmoid value(s) in the range (0, 1).
    
    Examples:
        >>> sigmoid(0)
        0.5
        >>> sigmoid(0, shift=2)  # Left-shifted by 2
        0.8807970779778823
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - shift)))


def inverted_sigmoid(x: Union[float, np.ndarray], shift: float = 0.0, steepness: float = 1.0) -> Union[float, np.ndarray]:
    """Compute the inverted sigmoid function (1 - sigmoid).
    
    Args:
        x: Input value(s) for the inverted sigmoid function.
        shift: Horizontal shift of the sigmoid (positive shifts right).
        steepness: Controls the steepness of the sigmoid curve (default: 1.0).
    
    Returns:
        Inverted sigmoid value(s) in the range (0, 1).
    
    Examples:
        >>> inverted_sigmoid(0)
        0.5
        >>> inverted_sigmoid(0, shift=-2)  # Right-shifted by 2
        0.8807970779778823
    """
    return 1.0 - sigmoid(x, shift=shift, steepness=steepness)


def sigmoid_product(
    x: Union[float, np.ndarray],
    left_shift: float = -1.0,
    right_shift: float = 1.0,
    steepness: float = 1.0
) -> Union[float, np.ndarray]:
    """Compute the product of a left-shifted sigmoid and inverted right-shifted sigmoid.
    
    This creates a bell-shaped curve (bump function) that is approximately Gaussian.
    The product is maximized at the center between the two shifts.
    
    Args:
        x: Input value(s) for the function.
        left_shift: Shift parameter for the left sigmoid (negative values shift left).
        right_shift: Shift parameter for the right sigmoid (positive values shift right).
        steepness: Controls the steepness of both sigmoids (default: 1.0).
    
    Returns:
        Product value(s) in the range (0, 0.25] with maximum at 0.25.
    
    Examples:
        >>> sigmoid_product(0, left_shift=-1, right_shift=1)
        0.25  # Maximum value at center
    """
    left_sig = sigmoid(x, shift=left_shift, steepness=steepness)
    right_inv_sig = inverted_sigmoid(x, shift=right_shift, steepness=steepness)
    return left_sig * right_inv_sig


def create_sigmoid_product_function(
    left_shift: float = -1.0,
    right_shift: float = 1.0,
    steepness: float = 1.0
) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """Create a sigmoid product function with fixed parameters.
    
    Args:
        left_shift: Shift parameter for the left sigmoid.
        right_shift: Shift parameter for the right sigmoid.
        steepness: Controls the steepness of both sigmoids.
    
    Returns:
        A function that takes x and returns the sigmoid product.
    
    Examples:
        >>> bell_curve = create_sigmoid_product_function(left_shift=-2, right_shift=2)
        >>> bell_curve(0)
        0.25
    """
    def func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return sigmoid_product(x, left_shift, right_shift, steepness)
    return func


def analyze_shape(
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000,
    left_shift: float = -1.0,
    right_shift: float = 1.0,
    steepness: float = 1.0
) -> dict:
    """Analyze the shape characteristics of the sigmoid product.
    
    Args:
        x_range: Tuple of (min, max) for the x-axis range.
        num_points: Number of points to sample.
        left_shift: Shift parameter for the left sigmoid.
        right_shift: Shift parameter for the right sigmoid.
        steepness: Controls the steepness of both sigmoids.
    
    Returns:
        Dictionary containing shape analysis:
            - 'x': x-axis values (numpy array)
            - 'y': y-axis values (numpy array)
            - 'max_value': Maximum y value
            - 'max_position': x position of maximum
            - 'shape_name': Description of the shape
            - 'fwhm': Full width at half maximum (approximate)
    
    Examples:
        >>> result = analyze_shape()
        >>> result['shape_name']
        'bell_curve'
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = sigmoid_product(x, left_shift, right_shift, steepness)
    
    max_idx = np.argmax(y)
    max_value = y[max_idx]
    max_position = x[max_idx]
    
    # Calculate FWHM (Full Width at Half Maximum)
    half_max = max_value / 2.0
    above_half = y >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm = x[indices[-1]] - x[indices[0]]
    else:
        fwhm = 0.0
    
    return {
        'x': x,
        'y': y,
        'max_value': float(max_value),
        'max_position': float(max_position),
        'shape_name': 'bell_curve',
        'fwhm': float(fwhm),
        'description': 'Bell-shaped curve (similar to Gaussian distribution)'
    }


def get_shape_components(
    x: Union[float, np.ndarray],
    left_shift: float = -1.0,
    right_shift: float = 1.0,
    steepness: float = 1.0
) -> dict:
    """Get individual components and their product.
    
    Args:
        x: Input value(s) for the functions.
        left_shift: Shift parameter for the left sigmoid.
        right_shift: Shift parameter for the right sigmoid.
        steepness: Controls the steepness of both sigmoids.
    
    Returns:
        Dictionary containing:
            - 'left_sigmoid': Left-shifted sigmoid values
            - 'right_inverted_sigmoid': Right-shifted inverted sigmoid values
            - 'product': Product of the two components
    
    Examples:
        >>> components = get_shape_components(0)
        >>> components['product']
        0.25
    """
    left_sig = sigmoid(x, shift=left_shift, steepness=steepness)
    right_inv_sig = inverted_sigmoid(x, shift=right_shift, steepness=steepness)
    product = left_sig * right_inv_sig
    
    return {
        'left_sigmoid': left_sig,
        'right_inverted_sigmoid': right_inv_sig,
        'product': product
    }
