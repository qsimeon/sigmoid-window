#!/usr/bin/env python3
"""
Sigmoid Product Shape Demo
===========================

This demo script demonstrates the shape created by the product of a left-shifted
sigmoid with an inverted right-shifted sigmoid. Both sigmoids have a range of (0, 1).

The resulting shape is a bell-like curve (similar to a Gaussian) that:
- Starts near 0 on the left
- Rises smoothly to a peak in the middle
- Falls smoothly back to 0 on the right

This shape is useful for creating smooth window functions, activation functions,
and other applications requiring smooth transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import from the available modules
from core import (
    sigmoid,
    inverted_sigmoid,
    sigmoid_product,
    create_sigmoid_product_function,
    analyze_shape,
    get_shape_components
)

from utils import (
    generate_x_values,
    normalize_values,
    find_peaks,
    calculate_statistics,
    export_to_dict,
    create_plot_config,
    calculate_area_under_curve,
    describe_shape
)


def visualize_components(x_range=(-5, 5), num_points=500, left_shift=-1.5, right_shift=1.5, steepness=2.0):
    """
    Visualize the individual sigmoid components and their product.
    
    Args:
        x_range: Tuple of (min, max) x values
        num_points: Number of points to generate
        left_shift: Shift parameter for the left sigmoid
        right_shift: Shift parameter for the right sigmoid
        steepness: Steepness parameter for both sigmoids
    
    Returns:
        Dictionary containing the plot data
    """
    # Generate x values
    x = generate_x_values(x_range, num_points)
    
    # Get individual components
    components = get_shape_components(x, left_shift, right_shift, steepness)
    
    # Create the plot
    plot_config = create_plot_config(
        title='Sigmoid Product Shape: Component Analysis',
        xlabel='x',
        ylabel='y',
        figsize=(12, 8)
    )
    
    fig, axes = plt.subplots(2, 2, figsize=plot_config['figsize'])
    
    # Plot 1: Left-shifted sigmoid
    axes[0, 0].plot(x, components['left_sigmoid'], 'b-', linewidth=2, label='Left Sigmoid')
    axes[0, 0].axvline(left_shift, color='b', linestyle='--', alpha=0.5, label=f'Shift = {left_shift}')
    axes[0, 0].set_title('Left-Shifted Sigmoid')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # Plot 2: Inverted right-shifted sigmoid
    axes[0, 1].plot(x, components['inverted_right_sigmoid'], 'r-', linewidth=2, label='Inverted Right Sigmoid')
    axes[0, 1].axvline(right_shift, color='r', linestyle='--', alpha=0.5, label=f'Shift = {right_shift}')
    axes[0, 1].set_title('Inverted Right-Shifted Sigmoid')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Both components overlaid
    axes[1, 0].plot(x, components['left_sigmoid'], 'b-', linewidth=2, label='Left Sigmoid', alpha=0.7)
    axes[1, 0].plot(x, components['inverted_right_sigmoid'], 'r-', linewidth=2, label='Inverted Right Sigmoid', alpha=0.7)
    axes[1, 0].set_title('Both Components')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-0.1, 1.1)
    
    # Plot 4: Product (the resulting shape)
    axes[1, 1].plot(x, components['product'], 'g-', linewidth=2.5, label='Product (Bell Shape)')
    axes[1, 1].fill_between(x, components['product'], alpha=0.3, color='g')
    axes[1, 1].set_title('Product: Bell-Like Shape')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Find and mark the peak
    peaks = find_peaks(x, components['product'])
    if peaks:
        peak_x, peak_y = peaks[0]
        axes[1, 1].plot(peak_x, peak_y, 'ro', markersize=10, label=f'Peak: ({peak_x:.2f}, {peak_y:.2f})')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('sigmoid_product_components.png', dpi=150, bbox_inches='tight')
    print("✓ Saved component visualization to 'sigmoid_product_components.png'")
    
    return {
        'x': x,
        'components': components,
        'peaks': peaks
    }


def compare_different_parameters():
    """
    Compare sigmoid products with different shift and steepness parameters.
    """
    x = generate_x_values((-6, 6), 800)
    
    # Different configurations to compare
    configs = [
        {'left_shift': -2, 'right_shift': 2, 'steepness': 1.0, 'label': 'Wide (shift=±2, steep=1)', 'color': 'blue'},
        {'left_shift': -1, 'right_shift': 1, 'steepness': 1.0, 'label': 'Medium (shift=±1, steep=1)', 'color': 'green'},
        {'left_shift': -1, 'right_shift': 1, 'steepness': 2.0, 'label': 'Steep (shift=±1, steep=2)', 'color': 'red'},
        {'left_shift': -0.5, 'right_shift': 0.5, 'steepness': 3.0, 'label': 'Narrow (shift=±0.5, steep=3)', 'color': 'purple'},
    ]
    
    plt.figure(figsize=(12, 7))
    
    for config in configs:
        y = sigmoid_product(x, config['left_shift'], config['right_shift'], config['steepness'])
        plt.plot(x, y, linewidth=2.5, label=config['label'], color=config['color'], alpha=0.8)
    
    plt.title('Sigmoid Product Shapes with Different Parameters', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('sigmoid_product_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved parameter comparison to 'sigmoid_product_comparison.png'")


def analyze_and_report(left_shift=-1.5, right_shift=1.5, steepness=2.0):
    """
    Perform detailed analysis and generate a report.
    
    Args:
        left_shift: Shift parameter for the left sigmoid
        right_shift: Shift parameter for the right sigmoid
        steepness: Steepness parameter
    
    Returns:
        Analysis results dictionary
    """
    print("\n" + "="*70)
    print("SIGMOID PRODUCT SHAPE ANALYSIS")
    print("="*70)
    
    # Perform analysis
    analysis = analyze_shape(
        x_range=(-10, 10),
        num_points=2000,
        left_shift=left_shift,
        right_shift=right_shift,
        steepness=steepness
    )
    
    # Generate description
    description = describe_shape(analysis)
    
    print(f"\nParameters:")
    print(f"  Left Shift:    {left_shift}")
    print(f"  Right Shift:   {right_shift}")
    print(f"  Steepness:     {steepness}")
    
    print(f"\nShape Description:")
    print(f"  {description}")
    
    print(f"\nStatistical Properties:")
    stats = analysis.get('statistics', {})
    print(f"  Maximum Value: {stats.get('max', 0):.6f}")
    print(f"  Mean Value:    {stats.get('mean', 0):.6f}")
    print(f"  Std Dev:       {stats.get('std', 0):.6f}")
    
    print(f"\nPeak Information:")
    peaks = analysis.get('peaks', [])
    if peaks:
        for i, (px, py) in enumerate(peaks, 1):
            print(f"  Peak {i}: x={px:.4f}, y={py:.6f}")
    else:
        print("  No peaks found")
    
    print(f"\nArea Under Curve:")
    x = analysis.get('x', np.array([]))
    y = analysis.get('y', np.array([]))
    if len(x) > 0 and len(y) > 0:
        area = calculate_area_under_curve(x, y)
        print(f"  Total Area: {area:.6f}")
    
    print("\n" + "="*70)
    
    return analysis


def demonstrate_use_cases():
    """
    Demonstrate practical use cases for the sigmoid product shape.
    """
    print("\n" + "="*70)
    print("PRACTICAL USE CASES")
    print("="*70)
    
    x = generate_x_values((-5, 5), 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Use Case 1: Smooth Window Function
    window = sigmoid_product(x, -2, 2, 1.5)
    axes[0, 0].plot(x, window, 'b-', linewidth=2.5)
    axes[0, 0].fill_between(x, window, alpha=0.3)
    axes[0, 0].set_title('Use Case 1: Smooth Window Function', fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Window Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0, 0.5, 'Useful for signal processing\nand data smoothing', 
                    ha='center', va='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Use Case 2: Activation Function
    activation = sigmoid_product(x, -1, 1, 3.0)
    axes[0, 1].plot(x, activation, 'r-', linewidth=2.5)
    axes[0, 1].fill_between(x, activation, alpha=0.3, color='red')
    axes[0, 1].set_title('Use Case 2: Neural Network Activation', fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Activation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0, 0.5, 'Bounded activation function\nwith smooth gradients', 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Use Case 3: Probability Distribution
    prob_dist = sigmoid_product(x, -1.5, 1.5, 2.0)
    # Normalize to make it a proper probability distribution
    prob_dist_normalized = prob_dist / calculate_area_under_curve(x, prob_dist)
    axes[1, 0].plot(x, prob_dist_normalized, 'g-', linewidth=2.5)
    axes[1, 0].fill_between(x, prob_dist_normalized, alpha=0.3, color='green')
    axes[1, 0].set_title('Use Case 3: Probability Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0, 0.15, 'Alternative to Gaussian\nwith explicit bounds', 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Use Case 4: Membership Function (Fuzzy Logic)
    membership = sigmoid_product(x, -2.5, 2.5, 1.0)
    axes[1, 1].plot(x, membership, 'm-', linewidth=2.5)
    axes[1, 1].fill_between(x, membership, alpha=0.3, color='magenta')
    axes[1, 1].set_title('Use Case 4: Fuzzy Membership Function', fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Membership Degree')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0, 0.5, 'Fuzzy set membership\nfor control systems', 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('sigmoid_product_use_cases.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved use cases visualization to 'sigmoid_product_use_cases.png'")
    
    print("\n1. Smooth Window Function:")
    print("   - Signal processing and filtering")
    print("   - Data smoothing and interpolation")
    
    print("\n2. Neural Network Activation:")
    print("   - Bounded output range (0, 1)")
    print("   - Smooth gradients for backpropagation")
    
    print("\n3. Probability Distribution:")
    print("   - Alternative to Gaussian with explicit bounds")
    print("   - Modeling uncertainty with controlled tails")
    
    print("\n4. Fuzzy Membership Function:")
    print("   - Fuzzy logic control systems")
    print("   - Decision-making under uncertainty")
    
    print("\n" + "="*70)


def main():
    """
    Main demonstration function.
    """
    print("\n" + "="*70)
    print("SIGMOID PRODUCT SHAPE DEMONSTRATION")
    print("="*70)
    print("\nThis demo explores the shape created by multiplying:")
    print("  • A left-shifted sigmoid (rising from 0 to 1)")
    print("  • An inverted right-shifted sigmoid (falling from 1 to 0)")
    print("\nThe result is a smooth, bell-like curve with range (0, 1).")
    print("="*70)
    
    try:
        # 1. Visualize components
        print("\n[1/4] Visualizing individual components and their product...")
        viz_data = visualize_components(
            x_range=(-5, 5),
            num_points=500,
            left_shift=-1.5,
            right_shift=1.5,
            steepness=2.0
        )
        
        # 2. Compare different parameters
        print("\n[2/4] Comparing different parameter configurations...")
        compare_different_parameters()
        
        # 3. Detailed analysis
        print("\n[3/4] Performing detailed shape analysis...")
        analysis = analyze_and_report(left_shift=-1.5, right_shift=1.5, steepness=2.0)
        
        # 4. Demonstrate use cases
        print("\n[4/4] Demonstrating practical use cases...")
        demonstrate_use_cases()
        
        # Export data for further analysis
        print("\n" + "="*70)
        print("EXPORTING DATA")
        print("="*70)
        
        x = viz_data['x']
        y = viz_data['components']['product']
        
        data_dict = export_to_dict(x, y, metadata={
            'description': 'Sigmoid product shape',
            'left_shift': -1.5,
            'right_shift': 1.5,
            'steepness': 2.0
        })
        
        print("\n✓ Data exported to dictionary format")
        print(f"  - {len(x)} data points")
        print(f"  - x range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  - y range: [{y.min():.6f}, {y.max():.6f}]")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n✓ The sigmoid product creates a BELL-SHAPED curve")
        print("✓ Properties:")
        print("  - Smooth transitions (infinitely differentiable)")
        print("  - Bounded range: (0, 1)")
        print("  - Symmetric when shifts are symmetric")
        print("  - Adjustable width via shift parameters")
        print("  - Adjustable steepness via steepness parameter")
        print("\n✓ Generated 3 visualization files:")
        print("  1. sigmoid_product_components.png")
        print("  2. sigmoid_product_comparison.png")
        print("  3. sigmoid_product_use_cases.png")
        
        print("\n" + "="*70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
        # Show plots
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
