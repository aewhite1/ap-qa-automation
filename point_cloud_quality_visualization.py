"""
Point Cloud Quality Visualization

This module provides visualization tools for point cloud quality check results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import json
from pathlib import Path
import time

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def visualize_point_cloud_quality(preprocessing_results: Optional[Dict] = None,
                                 postprocessing_results: Optional[Dict] = None,
                                 point_cloud: Optional[Any] = None,
                                 output_dir: Optional[str] = None,
                                 show_plots: bool = True,
                                 create_report: bool = True) -> None:
    """
    Create comprehensive visualizations for point cloud quality check results.
    
    Args:
        preprocessing_results: Results from pre-processing quality checks
        postprocessing_results: Results from post-processing quality checks
        point_cloud: Optional point cloud object (open3d.geometry.PointCloud)
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots
        create_report: Whether to create a combined report
    """
    if not preprocessing_results and not postprocessing_results:
        print("No results to visualize")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid for subplots
    if preprocessing_results and postprocessing_results:
        # Both pre and post processing results
        grid = plt.GridSpec(2, 3, figure=fig)
        
        # Create subplots for pre-processing
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])
        
        # Create subplots for post-processing
        ax4 = fig.add_subplot(grid[1, 0])
        ax5 = fig.add_subplot(grid[1, 1])
        ax6 = fig.add_subplot(grid[1, 2])
        
        # Plot pre-processing results
        _plot_preprocessing_summary(preprocessing_results, [ax1, ax2, ax3])
        
        # Plot post-processing results
        _plot_postprocessing_summary(postprocessing_results, [ax4, ax5, ax6])
        
    elif preprocessing_results:
        # Only pre-processing results
        grid = plt.GridSpec(2, 3, figure=fig)
        
        # Create subplots
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])
        ax4 = fig.add_subplot(grid[1, 0])
        ax5 = fig.add_subplot(grid[1, 1])
        ax6 = fig.add_subplot(grid[1, 2])
        
        # Plot pre-processing results
        _plot_preprocessing_detail(preprocessing_results, [ax1, ax2, ax3, ax4, ax5, ax6])
        
    elif postprocessing_results:
        # Only post-processing results
        grid = plt.GridSpec(2, 3, figure=fig)
        
        # Create subplots
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])
        ax4 = fig.add_subplot(grid[1, 0])
        ax5 = fig.add_subplot(grid[1, 1])
        ax6 = fig.add_subplot(grid[1, 2])
        
        # Plot post-processing results
        _plot_postprocessing_detail(postprocessing_results, [ax1, ax2, ax3, ax4, ax5, ax6])
    
    # Add title
    if preprocessing_results and postprocessing_results:
        title = "Point Cloud Quality Check Results"
    elif preprocessing_results:
        title = "Input Data Quality Check Results"
    else:
        title = "Point Cloud Quality Check Results"
    
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, "quality_check_summary.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "quality_check_summary.pdf"))
    
    # Show figure
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create detailed visualizations
    if preprocessing_results:
        _create_preprocessing_visualizations(preprocessing_results, output_dir, show_plots)
    
    if postprocessing_results:
        _create_postprocessing_visualizations(postprocessing_results, output_dir, show_plots)
    
    # Visualize point cloud if provided
    if point_cloud:
        _visualize_point_cloud(point_cloud, output_dir, show_plots)
    
    # Create combined report
    if create_report and output_dir:
        _create_combined_report(preprocessing_results, postprocessing_results, output_dir)

def _plot_preprocessing_summary(results: Dict, axes: List) -> None:
    """
    Plot summary of pre-processing results.
    
    Args:
        results: Pre-processing results
        axes: List of matplotlib axes
    """
    # Extract data
    checks = []
    values = []
    thresholds = []
    passed = []
    
    # Image count
    if "image_count" in results:
        checks.append("Image Count")
        values.append(results["image_count"])
        thresholds.append(results.get("min_images", 20))
        passed.append(results.get("image_count_check", False))
    
    # Resolution
    if "resolution" in results:
        checks.append("Resolution")
        values.append(results["resolution"][0] * results["resolution"][1] / 1000000)  # Megapixels
        min_res = results.get("min_resolution", (1080, 720))
        thresholds.append(min_res[0] * min_res[1] / 1000000)  # Megapixels
        passed.append(results.get("resolution_check", False))
    
    # Texture
    if "texture_score" in results:
        checks.append("Texture")
        values.append(results["texture_score"])
        thresholds.append(results.get("min_texture_score", 0.3))
        passed.append(results.get("texture_check", False))
    
    # Blur
    if "blur_score" in results:
        checks.append("Sharpness")
        values.append(results["blur_score"])
        thresholds.append(results.get("max_blur_threshold", 100.0))
        passed.append(results.get("blur_check", False))
    
    # Overlap
    if "overlap_score" in results:
        checks.append("Overlap")
        values.append(results["overlap_score"])
        thresholds.append(results.get("min_overlap_score", 0.5))
        passed.append(results.get("overlap_check", False))
    
    # Viewpoint variation
    if "viewpoint_variation" in results:
        checks.append("Viewpoint")
        values.append(results["viewpoint_variation"])
        thresholds.append(results.get("min_viewpoint_variation", 15.0))
        passed.append(results.get("viewpoint_check", False))
    
    # Exposure variation
    if "exposure_variation" in results:
        checks.append("Exposure")
        values.append(1.0 - results["exposure_variation"])  # Invert for consistency
        thresholds.append(1.0 - results.get("max_exposure_variation", 0.5))  # Invert for consistency
        passed.append(results.get("exposure_check", False))
    
    # Create data frame
    df = pd.DataFrame({
        "Check": checks,
        "Value": values,
        "Threshold": thresholds,
        "Passed": passed
    })
    
    # Plot overall result
    ax = axes[0]
    overall_passed = results.get("passed", False)
    ax.pie([1], labels=["PASSED" if overall_passed else "FAILED"],
           colors=["green" if overall_passed else "red"],
           autopct="%1.0f%%", startangle=90)
    ax.set_title("Overall Result")
    
    # Plot check results
    ax = axes[1]
    colors = ["green" if p else "red" for p in passed]
    ax.bar(df["Check"], df["Value"], color=colors)
    ax.set_title("Quality Metrics")
    ax.set_ylabel("Value")
    ax.set_xticklabels(df["Check"], rotation=45, ha="right")
    
    # Add threshold lines
    for i, (check, threshold) in enumerate(zip(df["Check"], df["Threshold"])):
        ax.hlines(threshold, i - 0.4, i + 0.4, colors="black", linestyles="dashed")
    
    # Plot pass/fail counts
    ax = axes[2]
    pass_count = sum(passed)
    fail_count = len(passed) - pass_count
    ax.pie([pass_count, fail_count], labels=["Passed", "Failed"],
           colors=["green", "red"], autopct="%1.0f%%", startangle=90)
    ax.set_title("Check Results")

def _plot_postprocessing_summary(results: Dict, axes: List) -> None:
    """
    Plot summary of post-processing results.
    
    Args:
        results: Post-processing results
        axes: List of matplotlib axes
    """
    # Extract data
    checks = []
    values = []
    thresholds = []
    passed = []
    
    # Point count
    if "point_count" in results:
        checks.append("Point Count")
        values.append(results["point_count"] / 1000)  # Thousands of points
        thresholds.append(results.get("min_points", 10000) / 1000)  # Thousands of points
        passed.append(results.get("point_count_check", False))
    
    # Density
    if "density" in results:
        checks.append("Density")
        values.append(results["density"])
        thresholds.append(results.get("min_density", 100.0))
        passed.append(results.get("density_check", False))
    
    # Noise
    if "noise_ratio" in results:
        checks.append("Noise")
        values.append(1.0 - results["noise_ratio"])  # Invert for consistency
        thresholds.append(1.0 - results.get("max_noise_ratio", 0.1))  # Invert for consistency
        passed.append(results.get("noise_check", False))
    
    # Completeness
    if "completeness" in results:
        checks.append("Completeness")
        values.append(results["completeness"])
        thresholds.append(results.get("min_completeness", 0.8))
        passed.append(results.get("completeness_check", False))
    
    # Outliers
    if "outlier_ratio" in results:
        checks.append("Outliers")
        values.append(1.0 - results["outlier_ratio"])  # Invert for consistency
        thresholds.append(1.0 - results.get("max_outlier_ratio", 0.05))  # Invert for consistency
        passed.append(results.get("outlier_check", False))
    
    # Color consistency
    if "color_consistency" in results:
        checks.append("Color")
        values.append(results["color_consistency"])
        thresholds.append(results.get("min_color_consistency", 0.7))
        passed.append(results.get("color_consistency_check", False))
    
    # Create data frame
    df = pd.DataFrame({
        "Check": checks,
        "Value": values,
        "Threshold": thresholds,
        "Passed": passed
    })
    
    # Plot overall result
    ax = axes[0]
    overall_passed = results.get("passed", False)
    ax.pie([1], labels=["PASSED" if overall_passed else "FAILED"],
           colors=["green" if overall_passed else "red"],
           autopct="%1.0f%%", startangle=90)
    ax.set_title("Overall Result")
    
    # Plot check results
    ax = axes[1]
    colors = ["green" if p else "red" for p in passed]
    ax.bar(df["Check"], df["Value"], color=colors)
    ax.set_title("Quality Metrics")
    ax.set_ylabel("Value")
    ax.set_xticklabels(df["Check"], rotation=45, ha="right")
    
    # Add threshold lines
    for i, (check, threshold) in enumerate(zip(df["Check"], df["Threshold"])):
        ax.hlines(threshold, i - 0.4, i + 0.4, colors="black", linestyles="dashed")
    
    # Plot pass/fail counts
    ax = axes[2]
    pass_count = sum(passed)
    fail_count = len(passed) - pass_count
    ax.pie([pass_count, fail_count], labels=["Passed", "Failed"],
           colors=["green", "red"], autopct="%1.0f%%", startangle=90)
    ax.set_title("Check Results")

def _plot_preprocessing_detail(results: Dict, axes: List) -> None:
    """
    Plot detailed pre-processing results.
    
    Args:
        results: Pre-processing results
        axes: List of matplotlib axes
    """
    # Plot image count
    ax = axes[0]
    if "image_count" in results:
        ax.bar(["Image Count"], [results["image_count"]], color="b")
        ax.axhline(y=results.get("min_images", 20), color="r", linestyle="--")
        ax.set_title("Image Count")
        ax.set_ylabel("Number of Images")
        ax.grid(True)
    
    # Plot resolution
    ax = axes[1]
    if "resolution" in results:
        resolution = results["resolution"]
        ax.bar(["Resolution"], [resolution[0] * resolution[1] / 1000000], color="g")  # Megapixels
        min_res = results.get("min_resolution", (1080, 720))
        ax.axhline(y=min_res[0] * min_res[1] / 1000000, color="r", linestyle="--")  # Megapixels
        ax.set_title("Resolution")
        ax.set_ylabel("Megapixels")
        ax.grid(True)
    
    # Plot texture score
    ax = axes[2]
    if "texture_score" in results:
        ax.bar(["Texture Score"], [results["texture_score"]], color="m")
        ax.axhline(y=results.get("min_texture_score", 0.3), color="r", linestyle="--")
        ax.set_title("Texture Score")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot blur score
    ax = axes[3]
    if "blur_score" in results:
        ax.bar(["Blur Score"], [results["blur_score"]], color="c")
        ax.axhline(y=results.get("max_blur_threshold", 100.0), color="r", linestyle="--")
        ax.set_title("Blur Score")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot overlap score
    ax = axes[4]
    if "overlap_score" in results:
        ax.bar(["Overlap Score"], [results["overlap_score"]], color="y")
        ax.axhline(y=results.get("min_overlap_score", 0.5), color="r", linestyle="--")
        ax.set_title("Overlap Score")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot viewpoint variation
    ax = axes[5]
    if "viewpoint_variation" in results:
        ax.bar(["Viewpoint Variation"], [results["viewpoint_variation"]], color="orange")
        ax.axhline(y=results.get("min_viewpoint_variation", 15.0), color="r", linestyle="--")
        ax.axhline(y=results.get("max_viewpoint_variation", 45.0), color="r", linestyle="--")
        ax.set_title("Viewpoint Variation")
        ax.set_ylabel("Degrees")
        ax.grid(True)

def _plot_postprocessing_detail(results: Dict, axes: List) -> None:
    """
    Plot detailed post-processing results.
    
    Args:
        results: Post-processing results
        axes: List of matplotlib axes
    """
    # Plot point count
    ax = axes[0]
    if "point_count" in results:
        ax.bar(["Point Count"], [results["point_count"]], color="b")
        ax.axhline(y=results.get("min_points", 10000), color="r", linestyle="--")
        ax.set_title("Point Count")
        ax.set_ylabel("Number of Points")
        ax.grid(True)
    
    # Plot density
    ax = axes[1]
    if "density" in results:
        ax.bar(["Density"], [results["density"]], color="g")
        ax.axhline(y=results.get("min_density", 100.0), color="r", linestyle="--")
        ax.set_title("Point Density")
        ax.set_ylabel("Points per Cubic Meter")
        ax.grid(True)
    
    # Plot noise ratio
    ax = axes[2]
    if "noise_ratio" in results:
        ax.bar(["Noise Ratio"], [results["noise_ratio"]], color="m")
        ax.axhline(y=results.get("max_noise_ratio", 0.1), color="r", linestyle="--")
        ax.set_title("Noise Ratio")
        ax.set_ylabel("Ratio")
        ax.grid(True)
    
    # Plot completeness
    ax = axes[3]
    if "completeness" in results:
        ax.bar(["Completeness"], [results["completeness"]], color="c")
        ax.axhline(y=results.get("min_completeness", 0.8), color="r", linestyle="--")
        ax.set_title("Completeness")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot outlier ratio
    ax = axes[4]
    if "outlier_ratio" in results:
        ax.bar(["Outlier Ratio"], [results["outlier_ratio"]], color="y")
        ax.axhline(y=results.get("max_outlier_ratio", 0.05), color="r", linestyle="--")
        ax.set_title("Outlier Ratio")
        ax.set_ylabel("Ratio")
        ax.grid(True)
    
    # Plot color consistency
    ax = axes[5]
    if "color_consistency" in results:
        ax.bar(["Color Consistency"], [results["color_consistency"]], color="orange")
        ax.axhline(y=results.get("min_color_consistency", 0.7), color="r", linestyle="--")
        ax.set_title("Color Consistency")
        ax.set_ylabel("Score")
        ax.grid(True)

def _create_preprocessing_visualizations(results: Dict, output_dir: Optional[str], show_plots: bool) -> None:
    """
    Create detailed visualizations for pre-processing results.
    
    Args:
        results: Pre-processing results
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots
    """
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flatten axes for easier indexing
    axs = axs.flatten()
    
    # Plot detailed results
    _plot_preprocessing_detail(results, axs)
    
    # Add title
    fig.suptitle("Input Data Quality Check Results", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, "preprocessing_results.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "preprocessing_results.pdf"))
    
    # Show figure
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create additional visualizations if data is available
    if "image_scores" in results:
        _create_image_score_visualizations(results, output_dir, show_plots)

def _create_postprocessing_visualizations(results: Dict, output_dir: Optional[str], show_plots: bool) -> None:
    """
    Create detailed visualizations for post-processing results.
    
    Args:
        results: Post-processing results
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots
    """
    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flatten axes for easier indexing
    axs = axs.flatten()
    
    # Plot detailed results
    _plot_postprocessing_detail(results, axs)
    
    # Add title
    fig.suptitle("Point Cloud Quality Check Results", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, "postprocessing_results.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "postprocessing_results.pdf"))
    
    # Show figure
    if show_plots:
        plt.show()
    else:
        plt.close()

def _create_image_score_visualizations(results: Dict, output_dir: Optional[str], show_plots: bool) -> None:
    """
    Create visualizations for individual image scores.
    
    Args:
        results: Pre-processing results
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots
    """
    # Check if image scores are available
    if "image_scores" not in results:
        return
    
    image_scores = results["image_scores"]
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Flatten axes for easier indexing
    axs = axs.flatten()
    
    # Plot texture scores
    ax = axs[0]
    if "texture_scores" in image_scores:
        texture_scores = image_scores["texture_scores"]
        ax.plot(texture_scores, marker="o")
        ax.axhline(y=results.get("min_texture_score", 0.3), color="r", linestyle="--")
        ax.set_title("Texture Scores")
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot blur scores
    ax = axs[1]
    if "blur_scores" in image_scores:
        blur_scores = image_scores["blur_scores"]
        ax.plot(blur_scores, marker="o")
        ax.axhline(y=results.get("max_blur_threshold", 100.0), color="r", linestyle="--")
        ax.set_title("Blur Scores")
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot overlap scores
    ax = axs[2]
    if "overlap_scores" in image_scores:
        overlap_scores = image_scores["overlap_scores"]
        ax.plot(overlap_scores, marker="o")
        ax.axhline(y=results.get("min_overlap_score", 0.5), color="r", linestyle="--")
        ax.set_title("Overlap Scores")
        ax.set_xlabel("Image Pair Index")
        ax.set_ylabel("Score")
        ax.grid(True)
    
    # Plot viewpoint variations
    ax = axs[3]
    if "viewpoint_variations" in image_scores:
        viewpoint_variations = image_scores["viewpoint_variations"]
        ax.plot(viewpoint_variations, marker="o")
        ax.axhline(y=results.get("min_viewpoint_variation", 15.0), color="r", linestyle="--")
        ax.axhline(y=results.get("max_viewpoint_variation", 45.0), color="r", linestyle="--")
        ax.set_title("Viewpoint Variations")
        ax.set_xlabel("Image Pair Index")
        ax.set_ylabel("Degrees")
        ax.grid(True)
    
    # Add title
    fig.suptitle("Individual Image Scores", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if output_dir:
        plt.savefig(os.path.join(output_dir, "image_scores.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "image_scores.pdf"))
    
    # Show figure
    if show_plots:
        plt.show()
    else:
        plt.close()

def _visualize_point_cloud(point_cloud: Any, output_dir: Optional[str], show_plots: bool) -> None:
    """
    Visualize point cloud.
    
    Args:
        point_cloud: Point cloud object (open3d.geometry.PointCloud)
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots
    """
    try:
        import open3d as o3d
        
        if output_dir:
            # Create visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(point_cloud)
            
            # Set view
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([1, 1, 1])
            
            # Capture images
            vis.poll_events()
            vis.update_renderer()
            
            # Save images
            vis.capture_screen_image(os.path.join(output_dir, "point_cloud.png"))
            
            # Close visualization
            vis.destroy_window()
        
        # Show point cloud
        if show_plots:
            o3d.visualization.draw_geometries([point_cloud])
    
    except Exception as e:
        print(f"Failed to visualize point cloud: {str(e)}")

def _create_combined_report(preprocessing_results: Optional[Dict],
                          postprocessing_results: Optional[Dict],
                          output_dir: str) -> None:
    """
    Create a combined report of quality check results.
    
    Args:
        preprocessing_results: Pre-processing results
        postprocessing_results: Post-processing results
        output_dir: Directory to save report
    """
    report = "Point Cloud Quality Check Report\n"
    report += "=" * 40 + "\n\n"
    
    # Add timestamp
    report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add pre-processing results
    if preprocessing_results:
        report += "Input Data Quality Check Results\n"
        report += "-" * 40 + "\n\n"
        
        # Overall result
        if preprocessing_results.get("passed", False):
            report += "OVERALL RESULT: PASSED ✓\n"
            report += "The input data is suitable for point cloud generation.\n\n"
        else:
            report += "OVERALL RESULT: FAILED ✗\n"
            report += "The input data may not be suitable for point cloud generation.\n\n"
        
        # Image count
        if "image_count" in preprocessing_results:
            report += f"Image Count: {preprocessing_results['image_count']} "
            if preprocessing_results.get("image_count_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {preprocessing_results.get('min_images', 20)})\n"
        
        # Resolution
        if "resolution" in preprocessing_results:
            resolution = preprocessing_results["resolution"]
            report += f"Resolution: {resolution[0]}x{resolution[1]} "
            if preprocessing_results.get("resolution_check", False):
                report += "✓\n"
            else:
                min_res = preprocessing_results.get("min_resolution", (1080, 720))
                report += f"✗ (Minimum: {min_res[0]}x{min_res[1]})\n"
        
        # Texture
        if "texture_score" in preprocessing_results:
            report += f"Texture Score: {preprocessing_results['texture_score']:.2f} "
            if preprocessing_results.get("texture_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {preprocessing_results.get('min_texture_score', 0.3):.2f})\n"
        
        # Blur
        if "blur_score" in preprocessing_results:
            report += f"Blur Score: {preprocessing_results['blur_score']:.2f} "
            if preprocessing_results.get("blur_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Maximum: {preprocessing_results.get('max_blur_threshold', 100.0):.2f})\n"
        
        # Overlap
        if "overlap_score" in preprocessing_results:
            report += f"Overlap Score: {preprocessing_results['overlap_score']:.2f} "
            if preprocessing_results.get("overlap_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {preprocessing_results.get('min_overlap_score', 0.5):.2f})\n"
        
        # Viewpoint variation
        if "viewpoint_variation" in preprocessing_results:
            report += f"Viewpoint Variation: {preprocessing_results['viewpoint_variation']:.2f} degrees "
            if preprocessing_results.get("viewpoint_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Range: {preprocessing_results.get('min_viewpoint_variation', 15.0):.2f}-{preprocessing_results.get('max_viewpoint_variation', 45.0):.2f})\n"
        
        # Exposure variation
        if "exposure_variation" in preprocessing_results:
            report += f"Exposure Variation: {preprocessing_results['exposure_variation']:.2f} "
            if preprocessing_results.get("exposure_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Maximum: {preprocessing_results.get('max_exposure_variation', 0.5):.2f})\n"
        
        # Add warnings
        if "warnings" in preprocessing_results and preprocessing_results["warnings"]:
            report += "\nWarnings:\n"
            for i, warning in enumerate(preprocessing_results["warnings"], 1):
                report += f"{i}. {warning}\n"
        
        # Add recommendations
        if "recommendations" in preprocessing_results and preprocessing_results["recommendations"]:
            report += "\nRecommendations:\n"
            for i, rec in enumerate(preprocessing_results["recommendations"], 1):
                report += f"{i}. {rec}\n"
        
        report += "\n"
    
    # Add post-processing results
    if postprocessing_results:
        report += "Point Cloud Quality Check Results\n"
        report += "-" * 40 + "\n\n"
        
        # Overall result
        if postprocessing_results.get("passed", False):
            report += "OVERALL RESULT: PASSED ✓\n"
            report += "The point cloud has good quality.\n\n"
        else:
            report += "OVERALL RESULT: FAILED ✗\n"
            report += "The point cloud may have quality issues.\n\n"
        
        # Point count
        if "point_count" in postprocessing_results:
            report += f"Point Count: {postprocessing_results['point_count']:,} "
            if postprocessing_results.get("point_count_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {postprocessing_results.get('min_points', 10000):,})\n"
        
        # Density
        if "density" in postprocessing_results:
            report += f"Density: {postprocessing_results['density']:.2f} points/m³ "
            if postprocessing_results.get("density_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {postprocessing_results.get('min_density', 100.0):.2f})\n"
        
        # Noise
        if "noise_ratio" in postprocessing_results:
            report += f"Noise Ratio: {postprocessing_results['noise_ratio']:.2%} "
            if postprocessing_results.get("noise_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Maximum: {postprocessing_results.get('max_noise_ratio', 0.1):.2%})\n"
        
        # Completeness
        if "completeness" in postprocessing_results:
            report += f"Completeness: {postprocessing_results['completeness']:.2%} "
            if postprocessing_results.get("completeness_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {postprocessing_results.get('min_completeness', 0.8):.2%})\n"
        
        # Outliers
        if "outlier_ratio" in postprocessing_results:
            report += f"Outlier Ratio: {postprocessing_results['outlier_ratio']:.2%} "
            if postprocessing_results.get("outlier_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Maximum: {postprocessing_results.get('max_outlier_ratio', 0.05):.2%})\n"
        
        # Color consistency
        if "color_consistency" in postprocessing_results:
            report += f"Color Consistency: {postprocessing_results['color_consistency']:.2%} "
            if postprocessing_results.get("color_consistency_check", False):
                report += "✓\n"
            else:
                report += f"✗ (Minimum: {postprocessing_results.get('min_color_consistency', 0.7):.2%})\n"
        
        # Add warnings
        if "warnings" in postprocessing_results and postprocessing_results["warnings"]:
            report += "\nWarnings:\n"
            for i, warning in enumerate(postprocessing_results["warnings"], 1):
                report += f"{i}. {warning}\n"
        
        # Add recommendations
        if "recommendations" in postprocessing_results and postprocessing_results["recommendations"]:
            report += "\nRecommendations:\n"
            for i, rec in enumerate(postprocessing_results["recommendations"], 1):
                report += f"{i}. {rec}\n"
    
    # Save report
    with open(os.path.join(output_dir, "quality_check_report.txt"), 'w') as f:
        f.write(report)
    
    print(f"Report saved to {os.path.join(output_dir, 'quality_check_report.txt')}")
