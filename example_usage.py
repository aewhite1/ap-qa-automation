"""
Example Usage of Point Cloud Quality Checks

This script demonstrates how to use the point cloud quality check modules
for both pre-processing and post-processing stages.
"""

import os
import sys
from pathlib import Path

# Import quality check modules
from point_cloud_quality_checks import run_quality_pipeline, PointCloudQualityPipeline
from point_cloud_quality_checks_preprocessing import check_images_for_point_cloud, check_video_for_point_cloud
from point_cloud_quality_checks_postprocessing import check_point_cloud_quality
from point_cloud_quality_visualization import visualize_point_cloud_quality

def example_1_check_images():
    """
    Example 1: Check a folder of images for point cloud generation.
    """
    print("\n=== Example 1: Check Images ===\n")
    
    # Replace with your image folder path
    image_folder = "path/to/your/images"
    
    # Check if the folder exists
    if not os.path.exists(image_folder):
        print(f"Image folder not found: {image_folder}")
        print("Please update the path to a valid image folder.")
        return
    
    # Create output directory
    output_dir = "results/example_1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run quality checks
    results = check_images_for_point_cloud(
        folder_path=image_folder,
        output_dir=output_dir,
        min_images=20,
        min_resolution=(1080, 720),
        visualize=True
    )
    
    # Print overall result
    print(f"\nOverall result: {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"Check the detailed report at: {output_dir}/image_quality_report.txt")

def example_2_check_video():
    """
    Example 2: Check a video for point cloud generation.
    """
    print("\n=== Example 2: Check Video ===\n")
    
    # Replace with your video file path
    video_path = "path/to/your/video.mp4"
    
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please update the path to a valid video file.")
        return
    
    # Create output directory
    output_dir = "results/example_2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run quality checks
    results = check_video_for_point_cloud(
        video_path=video_path,
        output_dir=output_dir,
        min_images=20,
        min_resolution=(1080, 720),
        sample_rate=5,  # Sample every 5th frame
        visualize=True
    )
    
    # Print overall result
    print(f"\nOverall result: {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"Check the detailed report at: {output_dir}/video_quality_report.txt")

def example_3_check_point_cloud():
    """
    Example 3: Check a point cloud file.
    """
    print("\n=== Example 3: Check Point Cloud ===\n")
    
    # Replace with your point cloud file path
    point_cloud_path = "path/to/your/pointcloud.ply"
    
    # Check if the file exists
    if not os.path.exists(point_cloud_path):
        print(f"Point cloud file not found: {point_cloud_path}")
        print("Please update the path to a valid point cloud file.")
        return
    
    # Create output directory
    output_dir = "results/example_3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run quality checks
    results = check_point_cloud_quality(
        point_cloud_path=point_cloud_path,
        output_dir=output_dir,
        min_points=10000,
        min_density=100.0,
        visualize=True
    )
    
    # Print overall result
    print(f"\nOverall result: {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"Check the detailed report at: {output_dir}/point_cloud_quality_report.txt")

def example_4_complete_pipeline():
    """
    Example 4: Run the complete quality check pipeline.
    """
    print("\n=== Example 4: Complete Pipeline ===\n")
    
    # Replace with your input data path (image folder or video file)
    input_path = "path/to/your/input"
    
    # Replace with your point cloud file path
    point_cloud_path = "path/to/your/pointcloud.ply"
    
    # Check if the paths exist
    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        print("Please update the path to a valid input path.")
        return
    
    if not os.path.exists(point_cloud_path):
        print(f"Point cloud file not found: {point_cloud_path}")
        print("Please update the path to a valid point cloud file.")
        return
    
    # Create output directory
    output_dir = "results/example_4"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run quality pipeline
    results = run_quality_pipeline(
        input_path=input_path,
        point_cloud_path=point_cloud_path,
        output_dir=output_dir,
        input_type="auto",  # Automatically detect input type
        visualize=True
    )
    
    # Print overall results
    if results["preprocessing"]:
        print(f"\nPre-processing result: {'PASSED' if results['preprocessing']['passed'] else 'FAILED'}")
    
    if results["postprocessing"]:
        print(f"Post-processing result: {'PASSED' if results['postprocessing']['passed'] else 'FAILED'}")
    
    print(f"Check the detailed reports and visualizations at: {output_dir}")

def example_5_custom_pipeline():
    """
    Example 5: Create a custom quality check pipeline.
    """
    print("\n=== Example 5: Custom Pipeline ===\n")
    
    # Replace with your input data path (image folder or video file)
    input_path = "path/to/your/input"
    
    # Replace with your point cloud file path
    point_cloud_path = "path/to/your/pointcloud.ply"
    
    # Check if the paths exist
    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        print("Please update the path to a valid input path.")
        return
    
    if not os.path.exists(point_cloud_path):
        print(f"Point cloud file not found: {point_cloud_path}")
        print("Please update the path to a valid point cloud file.")
        return
    
    # Create output directory
    output_dir = "results/example_5"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = PointCloudQualityPipeline(output_dir=output_dir)
    
    # Check input data with custom parameters
    preprocessing_results = pipeline.check_input_data(
        input_path=input_path,
        input_type="auto",
        min_images=30,  # Require more images
        min_resolution=(1920, 1080),  # Require higher resolution
        min_texture_score=0.4,  # Require more texture
        max_blur_threshold=150.0,  # Allow more blur
        min_overlap_score=0.6,  # Require more overlap
        min_viewpoint_variation=10.0,  # Allow smaller camera movements
        max_viewpoint_variation=40.0,  # Limit maximum camera movement
        max_exposure_variation=0.4,  # Allow more exposure variation
        sample_rate=3,  # Sample every 3rd frame for video
        visualize=True
    )
    
    # Check point cloud with custom parameters
    postprocessing_results = pipeline.check_point_cloud(
        point_cloud_path=point_cloud_path,
        min_points=20000,  # Require more points
        min_density=200.0,  # Require higher density
        max_noise_ratio=0.05,  # Allow less noise
        min_completeness=0.9,  # Require higher completeness
        max_outlier_ratio=0.03,  # Allow fewer outliers
        min_color_consistency=0.8,  # Require higher color consistency
        visualize=True
    )
    
    # Create combined visualization
    pipeline.create_combined_visualization(show_plots=True)
    
    # Print overall results
    if preprocessing_results:
        print(f"\nPre-processing result: {'PASSED' if preprocessing_results['passed'] else 'FAILED'}")
    
    if postprocessing_results:
        print(f"Post-processing result: {'PASSED' if postprocessing_results['passed'] else 'FAILED'}")
    
    print(f"Check the detailed reports and visualizations at: {output_dir}")

if __name__ == "__main__":
    # Run all examples
    example_1_check_images()
    example_2_check_video()
    example_3_check_point_cloud()
    example_4_complete_pipeline()
    example_5_custom_pipeline()
    
    print("\nAll examples completed. Check the 'results' directory for outputs.")
