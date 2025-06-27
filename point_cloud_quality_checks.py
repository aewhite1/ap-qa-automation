"""
Point Cloud Quality Checks

This module provides a unified interface for running quality checks on data
for point cloud generation, both before and after the reconstruction process.
"""

import os
import logging
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import time
from pathlib import Path

from point_cloud_quality_checks_preprocessing import PointCloudQualityPreCheck, check_images_for_point_cloud, check_video_for_point_cloud
from point_cloud_quality_checks_postprocessing import PointCloudQualityPostCheck, check_point_cloud_quality
from point_cloud_quality_visualization import visualize_point_cloud_quality

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PointCloudQualityPipeline:
    """
    Pipeline for running quality checks on data for point cloud generation.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the quality check pipeline.
        
        Args:
            output_dir: Directory to save results and visualizations
        """
        self.output_dir = output_dir
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Results storage
        self.preprocessing_results = None
        self.postprocessing_results = None
        self.point_cloud = None
    
    def check_input_data(self, 
                        input_path: str, 
                        input_type: str = "auto",
                        min_images: int = 20,
                        min_resolution: Tuple[int, int] = (1080, 720),
                        min_texture_score: float = 0.3,
                        max_blur_threshold: float = 100.0,
                        min_overlap_score: float = 0.5,
                        min_viewpoint_variation: float = 15.0,
                        max_viewpoint_variation: float = 45.0,
                        max_exposure_variation: float = 0.5,
                        sample_rate: int = 1,
                        visualize: bool = True) -> Dict:
        """
        Run quality checks on input data (images or video).
        
        Args:
            input_path: Path to image folder or video file
            input_type: Type of input data ("images", "video", or "auto" to detect)
            min_images: Minimum number of images required
            min_resolution: Minimum image resolution (width, height)
            min_texture_score: Minimum texture score (0-1)
            max_blur_threshold: Maximum blur threshold (lower values are more blurry)
            min_overlap_score: Minimum overlap score between consecutive images (0-1)
            min_viewpoint_variation: Minimum camera movement between images (degrees)
            max_viewpoint_variation: Maximum camera movement between images (degrees)
            max_exposure_variation: Maximum variation in exposure/brightness
            sample_rate: Sample every nth frame for video
            visualize: Whether to visualize results
            
        Returns:
            Dictionary with check results
        """
        logger.info(f"Checking input data: {input_path}")
        
        # Determine input type if auto
        if input_type == "auto":
            if os.path.isdir(input_path):
                input_type = "images"
            elif os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                input_type = "video"
            else:
                logger.error(f"Could not automatically determine input type for {input_path}")
                return {"error": f"Could not automatically determine input type for {input_path}"}
        
        # Create pre-processing checker
        pre_checker = PointCloudQualityPreCheck(
            min_images=min_images,
            min_resolution=min_resolution,
            min_texture_score=min_texture_score,
            max_blur_threshold=max_blur_threshold,
            min_overlap_score=min_overlap_score,
            min_viewpoint_variation=min_viewpoint_variation,
            max_viewpoint_variation=max_viewpoint_variation,
            max_exposure_variation=max_exposure_variation
        )
        
        # Run checks based on input type
        if input_type == "images":
            self.preprocessing_results = pre_checker.check_image_folder(input_path)
        else:  # video
            self.preprocessing_results = pre_checker.check_video(input_path, sample_rate=sample_rate)
        
        # Save results
        if self.output_dir:
            # Save results
            pre_checker.save_results(os.path.join(self.output_dir, f"preprocessing_results.json"))
            
            # Save report
            with open(os.path.join(self.output_dir, f"preprocessing_report.txt"), 'w') as f:
                f.write(pre_checker.get_detailed_report())
        
        # Print report
        print(pre_checker.get_detailed_report())
        
        # Visualize results
        if visualize and self.output_dir:
            pre_checker.visualize_results(self.output_dir)
        
        return self.preprocessing_results
    
    def check_point_cloud(self, 
                         point_cloud_path: str,
                         min_points: int = 10000,
                         min_density: float = 100.0,
                         max_noise_ratio: float = 0.1,
                         min_completeness: float = 0.8,
                         max_outlier_ratio: float = 0.05,
                         min_color_consistency: float = 0.7,
                         visualize: bool = True) -> Dict:
        """
        Run quality checks on a point cloud.
        
        Args:
            point_cloud_path: Path to point cloud file
            min_points: Minimum number of points required
            min_density: Minimum point density (points per cubic meter)
            max_noise_ratio: Maximum ratio of noise points
            min_completeness: Minimum completeness score (0-1)
            max_outlier_ratio: Maximum ratio of outlier points
            min_color_consistency: Minimum color consistency score (0-1)
            visualize: Whether to visualize results
            
        Returns:
            Dictionary with check results
        """
        logger.info(f"Checking point cloud: {point_cloud_path}")
        
        # Create post-processing checker
        post_checker = PointCloudQualityPostCheck(
            min_points=min_points,
            min_density=min_density,
            max_noise_ratio=max_noise_ratio,
            min_completeness=min_completeness,
            max_outlier_ratio=max_outlier_ratio,
            min_color_consistency=min_color_consistency
        )
        
        # Run checks
        self.postprocessing_results = post_checker.check_point_cloud(point_cloud_path)
        
        # Save point cloud reference
        try:
            import open3d as o3d
            self.point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        except Exception as e:
            logger.warning(f"Failed to load point cloud for visualization: {str(e)}")
        
        # Save results
        if self.output_dir:
            # Save results
            post_checker.save_results(os.path.join(self.output_dir, "postprocessing_results.json"))
            
            # Save report
            with open(os.path.join(self.output_dir, "postprocessing_report.txt"), 'w') as f:
                f.write(post_checker.get_detailed_report())
        
        # Print report
        print(post_checker.get_detailed_report())
        
        # Visualize results
        if visualize and self.output_dir:
            post_checker.visualize_results(self.output_dir)
        
        return self.postprocessing_results
    
    def create_combined_visualization(self, show_plots: bool = True) -> None:
        """
        Create combined visualizations for both pre-processing and post-processing results.
        
        Args:
            show_plots: Whether to display plots
        """
        if not self.preprocessing_results and not self.postprocessing_results:
            logger.warning("No results to visualize")
            return
        
        if not self.output_dir:
            logger.warning("No output directory specified for visualizations")
            return
        
        # Create visualizations
        visualize_point_cloud_quality(
            preprocessing_results=self.preprocessing_results,
            postprocessing_results=self.postprocessing_results,
            point_cloud=self.point_cloud,
            output_dir=self.output_dir,
            show_plots=show_plots,
            create_report=True
        )
        
        logger.info(f"Combined visualizations saved to {self.output_dir}")


def run_quality_pipeline(input_path: str,
                        point_cloud_path: Optional[str] = None,
                        output_dir: Optional[str] = None,
                        input_type: str = "auto",
                        visualize: bool = True) -> Dict:
    """
    Convenience function to run the complete quality check pipeline.
    
    Args:
        input_path: Path to image folder or video file
        point_cloud_path: Optional path to point cloud file
        output_dir: Directory to save results and visualizations
        input_type: Type of input data ("images", "video", or "auto" to detect)
        visualize: Whether to visualize results
        
    Returns:
        Dictionary with check results
    """
    # Create pipeline
    pipeline = PointCloudQualityPipeline(output_dir=output_dir)
    
    # Check input data
    preprocessing_results = pipeline.check_input_data(
        input_path=input_path,
        input_type=input_type,
        visualize=visualize
    )
    
    # Check point cloud if provided
    postprocessing_results = None
    if point_cloud_path:
        postprocessing_results = pipeline.check_point_cloud(
            point_cloud_path=point_cloud_path,
            visualize=visualize
        )
    
    # Create combined visualization
    if preprocessing_results and postprocessing_results and visualize:
        pipeline.create_combined_visualization(show_plots=visualize)
    
    # Return results
    return {
        "preprocessing": preprocessing_results,
        "postprocessing": postprocessing_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quality checks for point cloud generation")
    parser.add_argument("--input", required=True, help="Path to image folder or video file")
    parser.add_argument("--point-cloud", help="Path to point cloud file")
    parser.add_argument("--output-dir", help="Directory to save results and visualizations")
    parser.add_argument("--type", choices=["images", "video", "auto"], default="auto", 
                       help="Input type (images, video, or auto to detect)")
    parser.add_argument("--no-visualize", action="store_true", help="Don't visualize results")
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_quality_pipeline(
        input_path=args.input,
        point_cloud_path=args.point_cloud,
        output_dir=args.output_dir,
        input_type=args.type,
        visualize=not args.no_visualize
    )
