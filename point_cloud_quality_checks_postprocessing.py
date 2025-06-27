"""
Point Cloud Quality Checks for Post-Processing Stage

This module implements quality checks to evaluate the quality of generated point clouds
after the reconstruction process. These checks help determine if the point cloud
meets quality standards for downstream applications.
"""

import os
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PointCloudQualityPostCheck:
    """
    Class for performing quality checks on generated point clouds.
    """
    
    def __init__(self, 
                min_points: int = 10000,
                min_density: float = 100.0,
                max_noise_ratio: float = 0.1,
                min_completeness: float = 0.8,
                max_outlier_ratio: float = 0.05,
                min_color_consistency: float = 0.7):
        """
        Initialize the quality check parameters.
        
        Args:
            min_points: Minimum number of points required
            min_density: Minimum point density (points per cubic meter)
            max_noise_ratio: Maximum ratio of noise points
            min_completeness: Minimum completeness score (0-1)
            max_outlier_ratio: Maximum ratio of outlier points
            min_color_consistency: Minimum color consistency score (0-1)
        """
        self.min_points = min_points
        self.min_density = min_density
        self.max_noise_ratio = max_noise_ratio
        self.min_completeness = min_completeness
        self.max_outlier_ratio = max_outlier_ratio
        self.min_color_consistency = min_color_consistency
        
        # Results storage
        self.results = {
            "passed": False,
            "point_count": 0,
            "point_count_check": False,
            "density": 0.0,
            "density_check": False,
            "noise_ratio": 0.0,
            "noise_check": False,
            "completeness": 0.0,
            "completeness_check": False,
            "outlier_ratio": 0.0,
            "outlier_check": False,
            "color_consistency": 0.0,
            "color_consistency_check": False,
            "warnings": [],
            "recommendations": []
        }
        
        # Point cloud data storage
        self.point_cloud = None
        self.points = None
        self.colors = None
        self.normals = None
    
    def check_point_cloud(self, point_cloud_path: str) -> Dict:
        """
        Run all quality checks on a point cloud file.
        
        Args:
            point_cloud_path: Path to point cloud file (PLY, PCD, etc.)
            
        Returns:
            Dictionary with check results
        """
        logger.info(f"Checking point cloud: {point_cloud_path}")
        
        # Load point cloud
        try:
            import open3d as o3d
            self.point_cloud = o3d.io.read_point_cloud(point_cloud_path)
            
            # Extract data
            self.points = np.asarray(self.point_cloud.points)
            if self.point_cloud.has_colors():
                self.colors = np.asarray(self.point_cloud.colors)
            if self.point_cloud.has_normals():
                self.normals = np.asarray(self.point_cloud.normals)
            
            logger.info(f"Loaded point cloud with {len(self.points)} points")
        except Exception as e:
            logger.error(f"Failed to load point cloud: {str(e)}")
            self.results["warnings"].append(f"Failed to load point cloud: {str(e)}")
            self.results["recommendations"].append("Ensure the point cloud file is in a supported format (PLY, PCD, etc.)")
            return self.results
        
        # Run all checks
        self._check_point_count()
        self._check_density()
        self._check_noise()
        self._check_completeness()
        self._check_outliers()
        self._check_color_consistency()
        
        # Determine overall pass/fail
        checks = [
            self.results["point_count_check"],
            self.results["density_check"],
            self.results["noise_check"],
            self.results["completeness_check"],
            self.results["outlier_check"],
            self.results["color_consistency_check"]
        ]
        
        # Pass if at least 4 out of 6 checks pass
        self.results["passed"] = sum(checks) >= 4
        
        return self.results
    
    def _check_point_count(self) -> None:
        """
        Check if the point cloud has enough points.
        """
        if self.points is None:
            self.results["point_count_check"] = False
            return
        
        # Count points
        point_count = len(self.points)
        self.results["point_count"] = point_count
        
        # Check if there are enough points
        if point_count >= self.min_points:
            self.results["point_count_check"] = True
        else:
            self.results["point_count_check"] = False
            self.results["warnings"].append(
                f"Not enough points: {point_count} (minimum required: {self.min_points})"
            )
            self.results["recommendations"].append(
                "Capture more images or video frames to increase point density."
            )
    
    def _check_density(self) -> None:
        """
        Check if the point cloud has sufficient density.
        """
        if self.points is None:
            self.results["density_check"] = False
            return
        
        # Calculate bounding box volume
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        volume = np.prod(max_bounds - min_bounds)
        
        # Calculate density (points per cubic meter)
        if volume > 0:
            density = len(self.points) / volume
        else:
            density = 0.0
        
        self.results["density"] = density
        
        # Check if density is sufficient
        if density >= self.min_density:
            self.results["density_check"] = True
        else:
            self.results["density_check"] = False
            self.results["warnings"].append(
                f"Insufficient point density: {density:.2f} points/m³ (minimum required: {self.min_density:.2f})"
            )
            self.results["recommendations"].append(
                "Capture images with higher resolution or reduce the distance to the subject."
            )
    
    def _check_noise(self) -> None:
        """
        Check if the point cloud has acceptable noise levels.
        """
        if self.points is None or len(self.points) < 3:
            self.results["noise_check"] = False
            return
        
        # Estimate noise using statistical outlier removal
        try:
            import open3d as o3d
            
            # Create statistical outlier removal filter
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            
            # Apply filter
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Calculate noise ratio
            noise_ratio = 1.0 - (len(ind) / len(self.points))
            self.results["noise_ratio"] = noise_ratio
            
            # Check if noise ratio is acceptable
            if noise_ratio <= self.max_noise_ratio:
                self.results["noise_check"] = True
            else:
                self.results["noise_check"] = False
                self.results["warnings"].append(
                    f"High noise ratio: {noise_ratio:.2%} (maximum allowed: {self.max_noise_ratio:.2%})"
                )
                self.results["recommendations"].append(
                    "Improve lighting conditions and use a camera with better image quality."
                )
        except Exception as e:
            logger.warning(f"Failed to estimate noise: {str(e)}")
            self.results["noise_check"] = False
            self.results["warnings"].append(f"Failed to estimate noise: {str(e)}")
    
    def _check_completeness(self) -> None:
        """
        Check if the point cloud is complete (covers the entire subject).
        """
        if self.points is None or len(self.points) < 10:
            self.results["completeness_check"] = False
            return
        
        # Estimate completeness using voxel grid occupancy
        try:
            import open3d as o3d
            
            # Create voxel grid
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            
            # Calculate bounding box
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
            
            # Determine voxel size (aim for ~100 voxels along longest dimension)
            max_dim = np.max(bbox_size)
            voxel_size = max_dim / 100.0
            
            # Create voxel grid
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
            
            # Estimate completeness based on voxel occupancy pattern
            # This is a simplified approach - in a real implementation, you might
            # use more sophisticated methods like comparing to a reference model
            
            # Get voxels
            voxels = voxel_grid.get_voxels()
            
            # Calculate expected number of voxels for a complete model
            # (assuming a solid object would fill most of its bounding box)
            expected_voxels = np.prod(np.ceil(bbox_size / voxel_size)) * 0.5
            
            # Calculate completeness score
            completeness = min(1.0, len(voxels) / expected_voxels)
            self.results["completeness"] = completeness
            
            # Check if completeness is sufficient
            if completeness >= self.min_completeness:
                self.results["completeness_check"] = True
            else:
                self.results["completeness_check"] = False
                self.results["warnings"].append(
                    f"Incomplete point cloud: {completeness:.2%} (minimum required: {self.min_completeness:.2%})"
                )
                self.results["recommendations"].append(
                    "Capture images from all sides of the subject to ensure complete coverage."
                )
        except Exception as e:
            logger.warning(f"Failed to estimate completeness: {str(e)}")
            self.results["completeness_check"] = False
            self.results["warnings"].append(f"Failed to estimate completeness: {str(e)}")
    
    def _check_outliers(self) -> None:
        """
        Check if the point cloud has acceptable outlier levels.
        """
        if self.points is None or len(self.points) < 10:
            self.results["outlier_check"] = False
            return
        
        # Estimate outliers using radius outlier removal
        try:
            import open3d as o3d
            
            # Create radius outlier removal filter
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            
            # Calculate average distance to nearest neighbors
            distances = np.asarray(pcd.compute_nearest_neighbor_distance())
            avg_distance = np.mean(distances)
            
            # Apply filter
            cl, ind = pcd.remove_radius_outlier(nb_points=4, radius=avg_distance*3)
            
            # Calculate outlier ratio
            outlier_ratio = 1.0 - (len(ind) / len(self.points))
            self.results["outlier_ratio"] = outlier_ratio
            
            # Check if outlier ratio is acceptable
            if outlier_ratio <= self.max_outlier_ratio:
                self.results["outlier_check"] = True
            else:
                self.results["outlier_check"] = False
                self.results["warnings"].append(
                    f"High outlier ratio: {outlier_ratio:.2%} (maximum allowed: {self.max_outlier_ratio:.2%})"
                )
                self.results["recommendations"].append(
                    "Remove reflective or transparent surfaces from the scene."
                )
        except Exception as e:
            logger.warning(f"Failed to estimate outliers: {str(e)}")
            self.results["outlier_check"] = False
            self.results["warnings"].append(f"Failed to estimate outliers: {str(e)}")
    
    def _check_color_consistency(self) -> None:
        """
        Check if the point cloud has consistent colors.
        """
        if self.points is None or self.colors is None:
            self.results["color_consistency_check"] = False
            return
        
        # Calculate color consistency
        try:
            # Convert colors to HSV for better analysis
            import colorsys
            
            hsv_colors = []
            for rgb in self.colors:
                hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
                hsv_colors.append(hsv)
            
            hsv_colors = np.array(hsv_colors)
            
            # Calculate standard deviation of hue and value
            hue_std = np.std(hsv_colors[:, 0])
            value_std = np.std(hsv_colors[:, 2])
            
            # Calculate color consistency score
            # Lower standard deviation means more consistent colors
            hue_consistency = 1.0 - min(1.0, hue_std * 3.0)
            value_consistency = 1.0 - min(1.0, value_std * 3.0)
            
            # Overall color consistency (weighted average)
            color_consistency = 0.7 * value_consistency + 0.3 * hue_consistency
            self.results["color_consistency"] = color_consistency
            
            # Check if color consistency is sufficient
            if color_consistency >= self.min_color_consistency:
                self.results["color_consistency_check"] = True
            else:
                self.results["color_consistency_check"] = False
                self.results["warnings"].append(
                    f"Poor color consistency: {color_consistency:.2%} (minimum required: {self.min_color_consistency:.2%})"
                )
                self.results["recommendations"].append(
                    "Use consistent lighting conditions and avoid changes in exposure between images."
                )
        except Exception as e:
            logger.warning(f"Failed to estimate color consistency: {str(e)}")
            self.results["color_consistency_check"] = False
            self.results["warnings"].append(f"Failed to estimate color consistency: {str(e)}")
    
    def get_detailed_report(self) -> str:
        """
        Generate a detailed report of the quality check results.
        
        Returns:
            Formatted string with detailed results
        """
        report = "Point Cloud Quality Post-Check Report\n"
        report += "=" * 40 + "\n\n"
        
        # Overall result
        if self.results["passed"]:
            report += "OVERALL RESULT: PASSED ✓\n"
            report += "The point cloud has good quality.\n\n"
        else:
            report += "OVERALL RESULT: FAILED ✗\n"
            report += "The point cloud may have quality issues.\n\n"
        
        # Point count
        report += f"Point Count: {self.results['point_count']:,} "
        if self.results["point_count_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum: {self.min_points:,})\n"
        
        # Density
        report += f"Density: {self.results['density']:.2f} points/m³ "
        if self.results["density_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum: {self.min_density:.2f})\n"
        
        # Noise
        report += f"Noise Ratio: {self.results['noise_ratio']:.2%} "
        if self.results["noise_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Maximum: {self.max_noise_ratio:.2%})\n"
        
        # Completeness
        report += f"Completeness: {self.results['completeness']:.2%} "
        if self.results["completeness_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum: {self.min_completeness:.2%})\n"
        
        # Outliers
        report += f"Outlier Ratio: {self.results['outlier_ratio']:.2%} "
        if self.results["outlier_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Maximum: {self.max_outlier_ratio:.2%})\n"
        
        # Color consistency
        report += f"Color Consistency: {self.results['color_consistency']:.2%} "
        if self.results["color_consistency_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum: {self.min_color_consistency:.2%})\n"
        
        # Warnings
        if self.results["warnings"]:
            report += "\nWarnings:\n"
            for i, warning in enumerate(self.results["warnings"], 1):
                report += f"{i}. {warning}\n"
        
        # Recommendations
        if self.results["recommendations"]:
            report += "\nRecommendations:\n"
            for i, rec in enumerate(self.results["recommendations"], 1):
                report += f"{i}. {rec}\n"
        
        return report
    
    def save_results(self, output_path: str) -> None:
        """
        Save results to a JSON file.
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def visualize_results(self, output_dir: Optional[str] = None) -> None:
        """
        Visualize quality check results.
        
        Args:
            output_dir: Optional directory to save visualization files
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Flatten axes for easier indexing
        axs = axs.flatten()
        
        # Plot point count
        axs[0].bar(["Point Count"], [self.results["point_count"]], color='b')
        axs[0].axhline(y=self.min_points, color='r', linestyle='--')
        axs[0].set_title('Point Count')
        axs[0].set_ylabel('Number of Points')
        axs[0].grid(True)
        
        # Plot density
        axs[1].bar(["Density"], [self.results["density"]], color='g')
        axs[1].axhline(y=self.min_density, color='r', linestyle='--')
        axs[1].set_title('Point Density')
        axs[1].set_ylabel('Points per Cubic Meter')
        axs[1].grid(True)
        
        # Plot noise ratio
        axs[2].bar(["Noise Ratio"], [self.results["noise_ratio"]], color='m')
        axs[2].axhline(y=self.max_noise_ratio, color='r', linestyle='--')
        axs[2].set_title('Noise Ratio')
        axs[2].set_ylabel('Ratio')
        axs[2].grid(True)
        
        # Plot completeness
        axs[3].bar(["Completeness"], [self.results["completeness"]], color='c')
        axs[3].axhline(y=self.min_completeness, color='r', linestyle='--')
        axs[3].set_title('Completeness')
        axs[3].set_ylabel('Score')
        axs[3].grid(True)
        
        # Plot outlier ratio
        axs[4].bar(["Outlier Ratio"], [self.results["outlier_ratio"]], color='y')
        axs[4].axhline(y=self.max_outlier_ratio, color='r', linestyle='--')
        axs[4].set_title('Outlier Ratio')
        axs[4].set_ylabel('Ratio')
        axs[4].grid(True)
        
        # Plot color consistency
        axs[5].bar(["Color Consistency"], [self.results["color_consistency"]], color='orange')
        axs[5].axhline(y=self.min_color_consistency, color='r', linestyle='--')
        axs[5].set_title('Color Consistency')
        axs[5].set_ylabel('Score')
        axs[5].grid(True)
        
        plt.tight_layout()
        
        # Save figure if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, "postprocessing_results.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, "postprocessing_results.pdf"))
        
        plt.show()


def check_point_cloud_quality(point_cloud_path: str, 
                             output_dir: Optional[str] = None,
                             min_points: int = 10000,
                             min_density: float = 100.0,
                             visualize: bool = True) -> Dict:
    """
    Convenience function to check the quality of a point cloud.
    
    Args:
        point_cloud_path: Path to point cloud file
        output_dir: Optional directory to save results and visualizations
        min_points: Minimum number of points required
        min_density: Minimum point density (points per cubic meter)
        visualize: Whether to visualize results
        
    Returns:
        Dictionary with check results
    """
    checker = PointCloudQualityPostCheck(
        min_points=min_points,
        min_density=min_density
    )
    
    results = checker.check_point_cloud(point_cloud_path)
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results
        checker.save_results(os.path.join(output_dir, "point_cloud_quality_results.json"))
        
        # Save report
        with open(os.path.join(output_dir, "point_cloud_quality_report.txt"), 'w') as f:
            f.write(checker.get_detailed_report())
    
    # Print report
    print(checker.get_detailed_report())
    
    # Visualize results
    if visualize:
        checker.visualize_results(output_dir)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check the quality of a point cloud")
    parser.add_argument("--input", required=True, help="Path to point cloud file")
    parser.add_argument("--output-dir", help="Directory to save results and visualizations")
    parser.add_argument("--min-points", type=int, default=10000, help="Minimum number of points required")
    parser.add_argument("--min-density", type=float, default=100.0, help="Minimum point density (points per cubic meter)")
    parser.add_argument("--max-noise", type=float, default=0.1, help="Maximum noise ratio")
    parser.add_argument("--min-completeness", type=float, default=0.8, help="Minimum completeness score (0-1)")
    parser.add_argument("--max-outliers", type=float, default=0.05, help="Maximum outlier ratio")
    parser.add_argument("--min-color-consistency", type=float, default=0.7, help="Minimum color consistency score (0-1)")
    parser.add_argument("--no-visualize", action="store_true", help="Don't visualize results")
    
    args = parser.parse_args()
    
    # Create quality checker
    checker = PointCloudQualityPostCheck(
        min_points=args.min_points,
        min_density=args.min_density,
        max_noise_ratio=args.max_noise,
        min_completeness=args.min_completeness,
        max_outlier_ratio=args.max_outliers,
        min_color_consistency=args.min_color_consistency
    )
    
    # Run checks
    results = checker.check_point_cloud(args.input)
    
    # Save results
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Save results
        checker.save_results(os.path.join(args.output_dir, "point_cloud_quality_results.json"))
        
        # Save report
        with open(os.path.join(args.output_dir, "point_cloud_quality_report.txt"), 'w') as f:
            f.write(checker.get_detailed_report())
    
    # Print report
    print(checker.get_detailed_report())
    
    # Visualize results
    if not args.no_visualize:
        checker.visualize_results(args.output_dir)
