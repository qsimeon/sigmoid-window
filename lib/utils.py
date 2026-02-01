"""Utility functions for visualization and data processing.

This module provides helper functions for plotting sigmoid shapes,
exporting data, and performing common operations on sigmoid-related data.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import json


def generate_x_values(
    x_range: Tuple[float, float] = (-10, 10),
    num_points: int = 1000
) -> np.ndarray:
    """Generate evenly spaced x values for plotting.
    
    Args:
        x_range: Tuple of (min, max) for the x-axis range.
        num_points: Number of points to generate.
    
    Returns:
        Numpy array of x values.
    
    Examples:
        >>> x = generate_x_values((-5, 5), 11)
        >>> len(x)
        11
    """
    return np.linspace(x_range[0], x_range[1], num_points)


def normalize_values(
    values: np.ndarray,
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Normalize values to a target range.
    
    Args:
        values: Input array to normalize.
        target_range: Tuple of (min, max) for the target range.
    
    Returns:
        Normalized array.
    
    Examples:
        >>> normalize_values(np.array([0, 5, 10]), (0, 1))
        array([0. , 0.5, 1. ])
    """
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val == min_val:
        return np.full_like(values, (target_range[0] + target_range[1]) / 2.0)
    
    normalized = (values - min_val) / (max_val - min_val)
    scaled = normalized * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def find_peaks(
    x: np.ndarray,
    y: np.ndarray,
    threshold: Optional[float] = None
) -> List[Tuple[float, float]]:
    """Find local peaks in the data.
    
    Args:
        x: X-axis values.
        y: Y-axis values.
        threshold: Optional minimum threshold for peak detection.
    
    Returns:
        List of (x, y) tuples representing peak positions.
    
    Examples:
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> y = np.array([0, 1, 0, 2, 0])
        >>> find_peaks(x, y)
        [(1, 1), (3, 2)]
    """
    peaks = []
    
    for i in range(1, len(y) - 1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            if threshold is None or y[i] >= threshold:
                peaks.append((float(x[i]), float(y[i])))
    
    return peaks


def calculate_statistics(values: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for an array of values.
    
    Args:
        values: Input array.
    
    Returns:
        Dictionary containing statistical measures:
            - 'mean': Mean value
            - 'std': Standard deviation
            - 'min': Minimum value
            - 'max': Maximum value
            - 'median': Median value
    
    Examples:
        >>> stats = calculate_statistics(np.array([1, 2, 3, 4, 5]))
        >>> stats['mean']
        3.0
    """
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }


def export_to_dict(
    x: np.ndarray,
    y: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Export x, y data to a dictionary format.
    
    Args:
        x: X-axis values.
        y: Y-axis values.
        metadata: Optional metadata to include.
    
    Returns:
        Dictionary containing the data and metadata.
    
    Examples:
        >>> data = export_to_dict(np.array([0, 1]), np.array([0.5, 0.7]))
        >>> 'x' in data and 'y' in data
        True
    """
    result = {
        'x': x.tolist() if isinstance(x, np.ndarray) else x,
        'y': y.tolist() if isinstance(y, np.ndarray) else y,
        'num_points': len(x)
    }
    
    if metadata:
        result['metadata'] = metadata
    
    return result


def export_to_json(
    x: np.ndarray,
    y: np.ndarray,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    indent: int = 2
) -> None:
    """Export x, y data to a JSON file.
    
    Args:
        x: X-axis values.
        y: Y-axis values.
        filepath: Path to the output JSON file.
        metadata: Optional metadata to include.
        indent: JSON indentation level (default: 2).
    
    Examples:
        >>> export_to_json(np.array([0, 1]), np.array([0.5, 0.7]), 'data.json')
    """
    data = export_to_dict(x, y, metadata)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def export_to_csv(
    x: np.ndarray,
    y: np.ndarray,
    filepath: str,
    header: bool = True,
    delimiter: str = ','
) -> None:
    """Export x, y data to a CSV file.
    
    Args:
        x: X-axis values.
        y: Y-axis values.
        filepath: Path to the output CSV file.
        header: Whether to include a header row (default: True).
        delimiter: CSV delimiter (default: ',').
    
    Examples:
        >>> export_to_csv(np.array([0, 1]), np.array([0.5, 0.7]), 'data.csv')
    """
    data = np.column_stack((x, y))
    
    if header:
        header_str = f'x{delimiter}y'
        np.savetxt(filepath, data, delimiter=delimiter, header=header_str, comments='')
    else:
        np.savetxt(filepath, data, delimiter=delimiter)


def create_plot_config(
    title: str = 'Sigmoid Product Shape',
    xlabel: str = 'x',
    ylabel: str = 'y',
    grid: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, Any]:
    """Create a configuration dictionary for plotting.
    
    Args:
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        grid: Whether to show grid.
        figsize: Figure size as (width, height).
    
    Returns:
        Dictionary containing plot configuration.
    
    Examples:
        >>> config = create_plot_config(title='My Plot')
        >>> config['title']
        'My Plot'
    """
    return {
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'grid': grid,
        'figsize': figsize
    }


def interpolate_values(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    kind: str = 'linear'
) -> np.ndarray:
    """Interpolate y values at new x positions.
    
    Args:
        x: Original x values.
        y: Original y values.
        x_new: New x positions for interpolation.
        kind: Interpolation method ('linear', 'nearest', 'cubic', etc.).
    
    Returns:
        Interpolated y values at x_new positions.
    
    Examples:
        >>> x = np.array([0, 1, 2])
        >>> y = np.array([0, 1, 4])
        >>> x_new = np.array([0.5, 1.5])
        >>> interpolate_values(x, y, x_new, 'linear')
        array([0.5, 2.5])
    """
    from scipy import interpolate
    
    f = interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value='extrapolate')
    return f(x_new)


def calculate_area_under_curve(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'trapezoid'
) -> float:
    """Calculate the area under the curve.
    
    Args:
        x: X-axis values.
        y: Y-axis values.
        method: Integration method ('trapezoid' or 'simpson').
    
    Returns:
        Area under the curve.
    
    Examples:
        >>> x = np.array([0, 1, 2])
        >>> y = np.array([0, 1, 0])
        >>> calculate_area_under_curve(x, y)
        1.0
    """
    if method == 'trapezoid':
        return float(np.trapz(y, x))
    elif method == 'simpson':
        from scipy import integrate
        return float(integrate.simpson(y, x))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trapezoid' or 'simpson'.")


def describe_shape(analysis_result: Dict[str, Any]) -> str:
    """Generate a human-readable description of the shape.
    
    Args:
        analysis_result: Dictionary from analyze_shape() function.
    
    Returns:
        String description of the shape.
    
    Examples:
        >>> result = {'shape_name': 'bell_curve', 'max_value': 0.25, 'max_position': 0.0}
        >>> desc = describe_shape(result)
        >>> 'bell' in desc.lower()
        True
    """
    shape_name = analysis_result.get('shape_name', 'unknown')
    max_val = analysis_result.get('max_value', 0)
    max_pos = analysis_result.get('max_position', 0)
    fwhm = analysis_result.get('fwhm', 0)
    
    description = f"The shape is a {shape_name.replace('_', ' ')} "
    description += f"with a maximum value of {max_val:.4f} at x = {max_pos:.4f}. "
    
    if fwhm > 0:
        description += f"The full width at half maximum (FWHM) is {fwhm:.4f}. "
    
    description += "This shape is created by multiplying a left-shifted sigmoid "
    description += "with an inverted right-shifted sigmoid, resulting in a smooth, "
    description += "bell-shaped curve similar to a Gaussian distribution."
    
    return description
