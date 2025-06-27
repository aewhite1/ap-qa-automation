"""
Point Cloud Quality Checks for Pre-Processing Stage

This module implements quality checks to evaluate whether images or videos
will create successful point clouds before the reconstruction process.
These checks help determine if the input data meets quality standards
for 3D reconstruction using COLMAP, NeRF studio, or other tools.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PointCloudQualityPreCheck:
    """
    Class for performing quality checks on images or videos before point cloud generation.
    """
    
    def __init__(self, 
                 min_images: int = 20,
                 min_resolution: Tuple[int, int] = (1080, 720),
                 min_texture_score: float = 0.3,
                 max_blur_threshold: float = 100.0,
                 min_overlap_score: float = 0.5,
                 min_viewpoint_variation: float = 15.0,
                 max_viewpoint_variation: float = 45.0,
                 max_exposure_variation: float = 0.5):
        """
        Initialize the quality check parameters.
        
        Args:
            min_images: Minimum number of images required
            min_resolution: Minimum image resolution (width, height)
            min_texture_score: Minimum texture score (0-1)
            max_blur_threshold: Maximum blur threshold (lower values are more blurry)
            min_overlap_score: Minimum overlap score between consecutive images (0-1)
            min_viewpoint_variation: Minimum camera movement between images (degrees)
            max_viewpoint_variation: Maximum camera movement between images (degrees)
            max_exposure_variation: Maximum variation in exposure/brightness
        """
        self.min_images = min_images
        self.min_resolution = min_resolution
        self.min_texture_score = min_texture_score
        self.max_blur_threshold = max_blur_threshold
        self.min_overlap_score = min_overlap_score
        self.min_viewpoint_variation = min_viewpoint_variation
        self.max_viewpoint_variation = max_viewpoint_variation
        self.max_exposure_variation = max_exposure_variation
        
        # Results storage
        self.results = {
            "passed": False,
            "image_count": 0,
            "image_count_check": False,
            "resolution_check": False,
            "texture_check": False,
            "blur_check": False,
            "overlap_check": False,
            "viewpoint_check": False,
            "exposure_check": False,
            "image_scores": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Image data storage
        self.images = []
        self.image_paths = []
    
    def check_image_folder(self, folder_path: str) -> Dict:
        """
        Run all quality checks on a folder of images.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Dictionary with check results
        """
        logger.info(f"Checking image folder: {folder_path}")
        
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(folder_path).glob(f"*{ext}")))
            image_files.extend(list(Path(folder_path).glob(f"*{ext.upper()}")))
        
        # Sort image files to ensure consistent order
        image_files.sort()
        
        # Check if there are enough images
        if len(image_files) < self.min_images:
            self.results["image_count"] = len(image_files)
            self.results["image_count_check"] = False
            self.results["passed"] = False
            self.results["warnings"].append(
                f"Not enough images: {len(image_files)} (minimum required: {self.min_images})"
            )
            self.results["recommendations"].append(
                f"Capture at least {self.min_images} images of the subject from different angles."
            )
            return self.results
        
        # Load images
        self.images = []
        self.image_paths = []
        
        logger.info(f"Loading {len(image_files)} images...")
        for img_path in tqdm(image_files, desc="Loading images"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.images.append(img)
                    self.image_paths.append(str(img_path))
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {str(e)}")
        
        # Update image count
        self.results["image_count"] = len(self.images)
        
        # Check if there are enough valid images
        if len(self.images) < self.min_images:
            self.results["image_count_check"] = False
            self.results["passed"] = False
            self.results["warnings"].append(
                f"Not enough valid images: {len(self.images)} (minimum required: {self.min_images})"
            )
            self.results["recommendations"].append(
                f"Capture at least {self.min_images} images of the subject from different angles."
            )
            return self.results
        else:
            self.results["image_count_check"] = True
        
        # Run all checks
        self._check_resolution()
        self._check_texture()
        self._check_blur()
        self._check_overlap()
        self._check_viewpoint_variation()
        self._check_exposure_consistency()
        
        # Determine overall pass/fail
        checks = [
            self.results["image_count_check"],
            self.results["resolution_check"],
            self.results["texture_check"],
            self.results["blur_check"],
            self.results["overlap_check"],
            self.results["viewpoint_check"],
            self.results["exposure_check"]
        ]
        
        # Pass if at least 5 out of 7 checks pass
        self.results["passed"] = sum(checks) >= 5
        
        return self.results
    
    def check_video(self, video_path: str, sample_rate: int = 1) -> Dict:
        """
        Run all quality checks on a video file by extracting frames.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every nth frame
            
        Returns:
            Dictionary with check results
        """
        logger.info(f"Checking video: {video_path}")
        
        # Open video file
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.results["warnings"].append(f"Failed to open video file: {video_path}")
                return self.results
        except Exception as e:
            self.results["warnings"].append(f"Failed to open video file: {str(e)}")
            return self.results
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video has {frame_count} frames at {fps} fps")
        
        # Extract frames
        self.images = []
        self.image_paths = []
        
        frame_idx = 0
        
        logger.info(f"Extracting frames (sampling every {sample_rate} frames)...")
        with tqdm(total=frame_count // sample_rate) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    self.images.append(frame)
                    self.image_paths.append(f"frame_{frame_idx:06d}")
                    pbar.update(1)
                
                frame_idx += 1
        
        # Release video capture
        cap.release()
        
        # Update image count
        self.results["image_count"] = len(self.images)
        
        # Check if there are enough frames
        if len(self.images) < self.min_images:
            self.results["image_count_check"] = False
            self.results["passed"] = False
            self.results["warnings"].append(
                f"Not enough frames: {len(self.images)} (minimum required: {self.min_images})"
            )
            
            # Calculate required video length
            required_frames = self.min_images * sample_rate
            required_seconds = required_frames / fps
            
            self.results["recommendations"].append(
                f"Capture a longer video (at least {required_seconds:.1f} seconds at {fps} fps) "
                f"or reduce the sampling rate (currently every {sample_rate} frames)."
            )
            return self.results
        else:
            self.results["image_count_check"] = True
        
        # Run all checks
        self._check_resolution()
        self._check_texture()
        self._check_blur()
        self._check_overlap()
        self._check_viewpoint_variation()
        self._check_exposure_consistency()
        
        # Determine overall pass/fail
        checks = [
            self.results["image_count_check"],
            self.results["resolution_check"],
            self.results["texture_check"],
            self.results["blur_check"],
            self.results["overlap_check"],
            self.results["viewpoint_check"],
            self.results["exposure_check"]
        ]
        
        # Pass if at least 5 out of 7 checks pass
        self.results["passed"] = sum(checks) >= 5
        
        return self.results
    
    def _check_resolution(self) -> None:
        """
        Check if images meet minimum resolution requirements.
        """
        if not self.images:
            self.results["resolution_check"] = False
            return
        
        # Check resolution of each image
        resolutions = []
        low_res_images = []
        
        for i, img in enumerate(self.images):
            h, w = img.shape[:2]
            resolutions.append((w, h))
            
            if w < self.min_resolution[0] or h < self.min_resolution[1]:
                low_res_images.append((i, w, h))
        
        # Calculate percentage of images that meet resolution requirements
        resolution_pass_rate = 1.0 - (len(low_res_images) / len(self.images))
        
        # Pass if at least 90% of images meet resolution requirements
        if resolution_pass_rate >= 0.9:
            self.results["resolution_check"] = True
        else:
            self.results["resolution_check"] = False
            self.results["warnings"].append(
                f"{len(low_res_images)} out of {len(self.images)} images have resolution below "
                f"the minimum requirement of {self.min_resolution[0]}x{self.min_resolution[1]}"
            )
            
            # Add recommendation
            min_w, min_h = self.min_resolution
            self.results["recommendations"].append(
                f"Use a camera with higher resolution (at least {min_w}x{min_h})."
            )
    
    def _check_texture(self) -> None:
        """
        Check if images have enough texture for feature matching.
        """
        if not self.images:
            self.results["texture_check"] = False
            return
        
        # Calculate texture scores for each image
        texture_scores = []
        
        for img in tqdm(self.images, desc="Analyzing texture"):
            score = self._calculate_texture_score(img)
            texture_scores.append(score)
        
        # Store texture scores
        for i, score in enumerate(texture_scores):
            if i < len(self.results["image_scores"]):
                self.results["image_scores"][i]["texture_score"] = score
            else:
                self.results["image_scores"].append({"texture_score": score})
        
        # Calculate average texture score
        avg_texture_score = np.mean(texture_scores)
        
        # Count images with low texture
        low_texture_count = sum(1 for score in texture_scores if score < self.min_texture_score)
        
        # Pass if average score is above threshold and at most 20% of images have low texture
        if avg_texture_score >= self.min_texture_score and low_texture_count <= 0.2 * len(self.images):
            self.results["texture_check"] = True
        else:
            self.results["texture_check"] = False
            self.results["warnings"].append(
                f"Images have insufficient texture. Average score: {avg_texture_score:.2f} "
                f"(minimum required: {self.min_texture_score:.2f}). "
                f"{low_texture_count} images have low texture."
            )
            
            # Add recommendation
            self.results["recommendations"].append(
                "Capture images of subjects with more texture or surface detail. "
                "Avoid plain, untextured surfaces."
            )
    
    def _calculate_texture_score(self, img: np.ndarray) -> float:
        """
        Calculate a texture score for an image.
        
        Args:
            img: Image as numpy array
            
        Returns:
            Texture score (0-1, higher is better)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate gradients using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # Normalize to 0-1
        grad_mag_norm = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate texture score (mean of gradient magnitude)
        texture_score = np.mean(grad_mag_norm)
        
        return texture_score
    
    def _check_blur(self) -> None:
        """
        Check if images are not too blurry.
        """
        if not self.images:
            self.results["blur_check"] = False
            return
        
        # Calculate blur scores for each image
        blur_scores = []
        
        for img in tqdm(self.images, desc="Analyzing blur"):
            score = self._calculate_blur_score(img)
            blur_scores.append(score)
        
        # Store blur scores
        for i, score in enumerate(blur_scores):
            if i < len(self.results["image_scores"]):
                self.results["image_scores"][i]["blur_score"] = score
            else:
                self.results["image_scores"].append({"blur_score": score})
        
        # Count blurry images
        blurry_count = sum(1 for score in blur_scores if score < self.max_blur_threshold)
        
        # Pass if at most 20% of images are blurry
        if blurry_count <= 0.2 * len(self.images):
            self.results["blur_check"] = True
        else:
            self.results["blur_check"] = False
            self.results["warnings"].append(
                f"{blurry_count} out of {len(self.images)} images are too blurry."
            )
            
            # Add recommendation
            self.results["recommendations"].append(
                "Use a tripod or faster shutter speed to reduce blur. "
                "Ensure proper lighting conditions."
            )
    
    def _calculate_blur_score(self, img: np.ndarray) -> float:
        """
        Calculate a blur score for an image using Laplacian variance.
        
        Args:
            img: Image as numpy array
            
        Returns:
            Blur score (higher is sharper)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance of Laplacian
        blur_score = np.var(laplacian)
        
        return blur_score
    
    def _check_overlap(self) -> None:
        """
        Check if consecutive images have sufficient overlap.
        """
        if not self.images or len(self.images) < 2:
            self.results["overlap_check"] = False
            return
        
        # Calculate overlap scores for consecutive image pairs
        overlap_scores = []
        
        for i in tqdm(range(len(self.images) - 1), desc="Analyzing overlap"):
            score = self._calculate_overlap_score(self.images[i], self.images[i + 1])
            overlap_scores.append(score)
        
        # Store overlap scores
        for i, score in enumerate(overlap_scores):
            if i < len(self.results["image_scores"]):
                self.results["image_scores"][i]["overlap_score"] = score
            else:
                self.results["image_scores"].append({"overlap_score": score})
        
        # Calculate average overlap score
        avg_overlap_score = np.mean(overlap_scores)
        
        # Count pairs with low overlap
        low_overlap_count = sum(1 for score in overlap_scores if score < self.min_overlap_score)
        
        # Pass if average score is above threshold and at most 20% of pairs have low overlap
        if avg_overlap_score >= self.min_overlap_score and low_overlap_count <= 0.2 * len(overlap_scores):
            self.results["overlap_check"] = True
        else:
            self.results["overlap_check"] = False
            self.results["warnings"].append(
                f"Insufficient overlap between consecutive images. Average score: {avg_overlap_score:.2f} "
                f"(minimum required: {self.min_overlap_score:.2f}). "
                f"{low_overlap_count} image pairs have low overlap."
            )
            
            # Add recommendation
            self.results["recommendations"].append(
                "Capture images with more overlap between consecutive shots. "
                "Each part of the subject should appear in at least 3 images."
            )
    
    def _calculate_overlap_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate an overlap score between two images using feature matching.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Overlap score (0-1, higher is better)
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # Check if enough keypoints were found
        if len(kp1) < 10 or len(kp2) < 10 or des1 is None or des2 is None:
            return 0.0
        
        # Match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Calculate overlap score
        overlap_score = len(good_matches) / min(len(kp1), len(kp2))
        
        # Clip to 0-1 range
        overlap_score = min(1.0, overlap_score)
        
        return overlap_score
    
    def _check_viewpoint_variation(self) -> None:
        """
        Check if camera movement between consecutive images is appropriate.
        """
        if not self.images or len(self.images) < 2:
            self.results["viewpoint_check"] = False
            return
        
        # Calculate viewpoint variation for consecutive image pairs
        viewpoint_variations = []
        
        for i in tqdm(range(len(self.images) - 1), desc="Analyzing viewpoint variation"):
            variation = self._calculate_viewpoint_variation(self.images[i], self.images[i + 1])
            viewpoint_variations.append(variation)
        
        # Store viewpoint variations
        for i, variation in enumerate(viewpoint_variations):
            if i < len(self.results["image_scores"]):
                self.results["image_scores"][i]["viewpoint_variation"] = variation
            else:
                self.results["image_scores"].append({"viewpoint_variation": variation})
        
        # Count pairs with inappropriate viewpoint variation
        low_variation_count = sum(1 for var in viewpoint_variations if var < self.min_viewpoint_variation)
        high_variation_count = sum(1 for var in viewpoint_variations if var > self.max_viewpoint_variation)
        
        # Pass if at most 20% of pairs have inappropriate viewpoint variation
        inappropriate_count = low_variation_count + high_variation_count
        if inappropriate_count <= 0.2 * len(viewpoint_variations):
            self.results["viewpoint_check"] = True
        else:
            self.results["viewpoint_check"] = False
            self.results["warnings"].append(
                f"{inappropriate_count} out of {len(viewpoint_variations)} image pairs have inappropriate "
                f"viewpoint variation. {low_variation_count} pairs have too little variation, "
                f"{high_variation_count} pairs have too much variation."
            )
            
            # Add recommendation
            self.results["recommendations"].append(
                f"Move the camera {self.min_viewpoint_variation:.1f}° to {self.max_viewpoint_variation:.1f}° "
                f"between shots for optimal reconstruction."
            )
    
    def _calculate_viewpoint_variation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate viewpoint variation between two images.
        
        Args:
            img1: First image as numpy array
            img2: Second image as numpy array
            
        Returns:
            Viewpoint variation in degrees
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # Check if enough keypoints were found
        if len(kp1) < 10 or len(kp2) < 10 or des1 is None or des2 is None:
            return 0.0
        
        # Match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Check if enough good matches were found
        if len(good_matches) < 10:
            return 0.0
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return 0.0
        
        # Decompose homography to get rotation
        try:
            _, Rs, _, _, _ = cv2.decomposeHomographyMat(H, np.eye(3))
            
            # Extract rotation angles
            angles = []
            for R in Rs:
                # Convert rotation matrix to Euler angles
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(R[2, 1], R[2, 2])
                    y = np.arctan2(-R[2, 0], sy)
                    z = np.arctan2(R[1, 0], R[0, 0])
                else:
                    x = np.arctan2(-R[1, 2], R[1, 1])
                    y = np.arctan2(-R[2, 0], sy)
                    z = 0
                
                # Convert to degrees
                angles.append(np.sqrt(x*x + y*y + z*z) * 180.0 / np.pi)
            
            # Use the smallest angle
            viewpoint_variation = min(angles)
            
        except Exception as e:
            logger.warning(f"Failed to decompose homography: {str(e)}")
            viewpoint_variation = 0.0
        
        return viewpoint_variation
    
    def _check_exposure_consistency(self) -> None:
        """
        Check if images have consistent exposure/brightness.
        """
        if not self.images:
            self.results["exposure_check"] = False
            return
        
        # Calculate brightness for each image
        brightness_values = []
        
        for img in tqdm(self.images, desc="Analyzing exposure"):
            brightness = self._calculate_brightness(img)
            brightness_values.append(brightness)
        
        # Store brightness values
        for i, brightness in enumerate(brightness_values):
            if i < len(self.results["image_scores"]):
                self.results["image_scores"][i]["brightness"] = brightness
            else:
                self.results["image_scores"].append({"brightness": brightness})
        
        # Calculate mean and standard deviation of brightness
        mean_brightness = np.mean(brightness_values)
        std_brightness = np.std(brightness_values)
        
        # Calculate normalized standard deviation (coefficient of variation)
        if mean_brightness > 0:
            brightness_variation = std_brightness / mean_brightness
        else:
            brightness_variation = 0.0
        
        # Pass if brightness variation is below threshold
        if brightness_variation <= self.max_exposure_variation:
            self.results["exposure_check"] = True
        else:
            self.results["exposure_check"] = False
            self.results["warnings"].append(
                f"Inconsistent exposure/brightness across images. Variation: {brightness_variation:.3f} "
                f"(maximum allowed: {self.max_exposure_variation:.3f})"
            )
            
            # Add recommendation
            self.results["recommendations"].append(
                "Use consistent lighting conditions and camera settings (ISO, aperture, shutter speed) "
                "when capturing all images."
            )
    
    def _calculate_brightness(self, img: np.ndarray) -> float:
        """
        Calculate the brightness of an image.
        
        Args:
            img: Image as numpy array
            
        Returns:
            Brightness value (0-1)
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate mean brightness
        brightness = np.mean(gray) / 255.0
        
        return brightness
    
    def get_detailed_report(self) -> str:
        """
        Generate a detailed report of the quality check results.
        
        Returns:
            Formatted string with detailed results
        """
        report = "Point Cloud Quality Pre-Check Report\n"
        report += "=" * 40 + "\n\n"
        
        # Overall result
        if self.results["passed"]:
            report += "OVERALL RESULT: PASSED ✓\n"
            report += "The input should produce a good quality point cloud.\n\n"
        else:
            report += "OVERALL RESULT: FAILED ✗\n"
            report += "The input may not produce a good quality point cloud.\n\n"
        
        # Image count
        report += f"Image Count: {self.results['image_count']} "
        if self.results["image_count_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum: {self.min_images})\n"
        
        # Resolution
        report += f"Resolution: "
        if self.results["resolution_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum: {self.min_resolution[0]}x{self.min_resolution[1]})\n"
        
        # Texture
        report += f"Texture: "
        if self.results["texture_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum score: {self.min_texture_score:.2f})\n"
        
        # Blur
        report += f"Blur: "
        if self.results["blur_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Maximum threshold: {self.max_blur_threshold:.2f})\n"
        
        # Overlap
        report += f"Overlap: "
        if self.results["overlap_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Minimum score: {self.min_overlap_score:.2f})\n"
        
        # Viewpoint variation
        report += f"Viewpoint Variation: "
        if self.results["viewpoint_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Range: {self.min_viewpoint_variation:.1f}° - {self.max_viewpoint_variation:.1f}°)\n"
        
        # Exposure consistency
        report += f"Exposure Consistency: "
        if self.results["exposure_check"]:
            report += "✓\n"
        else:
            report += f"✗ (Maximum variation: {self.max_exposure_variation:.3f})\n"
        
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
        if not self.results["image_scores"]:
            logger.warning("No image scores to visualize")
            return
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Flatten axes for easier indexing
        axs = axs.flatten()
        
        # Extract scores
        texture_scores = [img.get("texture_score", 0) for img in self.results["image_scores"]]
        blur_scores = [img.get("blur_score", 0) for img in self.results["image_scores"]]
        overlap_scores = [img.get("overlap_score", 0) for img in self.results["image_scores"]]
        viewpoint_variations = [img.get("viewpoint_variation", 0) for img in self.results["image_scores"]]
        brightness_values = [img.get("brightness", 0) for img in self.results["image_scores"]]
        
        # Plot texture scores
        axs[0].plot(texture_scores, 'b-')
        axs[0].axhline(y=self.min_texture_score, color='r', linestyle='--')
        axs[0].set_title('Texture Scores')
        axs[0].set_xlabel('Image Index')
        axs[0].set_ylabel('Texture Score')
        axs[0].grid(True)
        
        # Plot blur scores
        axs[1].plot(blur_scores, 'g-')
        axs[1].axhline(y=self.max_blur_threshold, color='r', linestyle='--')
        axs[1].set_title('Blur Scores (higher is sharper)')
        axs[1].set_xlabel('Image Index')
        axs[1].set_ylabel('Blur Score')
        axs[1].grid(True)
        
        # Plot overlap scores
        if overlap_scores:
            axs[2].plot(overlap_scores, 'm-')
            axs[2].axhline(y=self.min_overlap_score, color='r', linestyle='--')
            axs[2].set_title('Overlap Scores')
            axs[2].set_xlabel('Image Pair Index')
            axs[2].set_ylabel('Overlap Score')
            axs[2].grid(True)
        
        # Plot viewpoint variations
        if viewpoint_variations:
            axs[3].plot(viewpoint_variations, 'c-')
            axs[3].axhline(y=self.min_viewpoint_variation, color='r', linestyle='--')
            axs[3].axhline(y=self.max_viewpoint_variation, color='r', linestyle='--')
            axs[3].set_title('Viewpoint Variations')
            axs[3].set_xlabel('Image Pair Index')
            axs[3].set_ylabel('Viewpoint Variation (degrees)')
            axs[3].grid(True)
        
        # Plot brightness values
        axs[4].plot(brightness_values, 'y-')
        axs[4].set_title('Brightness Values')
        axs[4].set_xlabel('Image Index')
        axs[4].set_ylabel('Brightness')
        axs[4].grid(True)
        
        # Plot overall check results
        checks = [
            ("Image Count", self.results["image_count_check"]),
            ("Resolution", self.results["resolution_check"]),
            ("Texture", self.results["texture_check"]),
            ("Blur", self.results["blur_check"]),
            ("Overlap", self.results["overlap_check"]),
            ("Viewpoint", self.results["viewpoint_check"]),
            ("Exposure", self.results["exposure_check"])
        ]
        
        check_names = [name for name, _ in checks]
        check_results = [1 if passed else 0 for _, passed in checks]
        
        axs[5].bar(check_names, check_results, color=['g' if r else 'r' for r in check_results])
        axs[5].set_title('Quality Check Results')
        axs[5].set_ylabel('Pass (1) / Fail (0)')
        axs[5].set_ylim(0, 1.2)
        for i, v in enumerate(check_results):
            axs[5].text(i, v + 0.1, '✓' if v else '✗', ha='center')
        
        plt.tight_layout()
        
        # Save figure if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, "preprocessing_results.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, "preprocessing_results.pdf"))
        
        plt.show()


def check_images_for_point_cloud(folder_path: str, 
                                output_dir: Optional[str] = None,
                                min_images: int = 20,
                                min_resolution: Tuple[int, int] = (1080, 720),
                                visualize: bool = True) -> Dict:
    """
    Convenience function to check if images are suitable for point cloud generation.
    
    Args:
        folder_path: Path to folder containing images
        output_dir: Optional directory to save results and visualizations
        min_images: Minimum number of images required
        min_resolution: Minimum image resolution (width, height)
        visualize: Whether to visualize results
        
    Returns:
        Dictionary with check results
    """
    checker = PointCloudQualityPreCheck(
        min_images=min_images,
        min_resolution=min_resolution
    )
    
    results = checker.check_image_folder(folder_path)
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results
        checker.save_results(os.path.join(output_dir, "image_quality_results.json"))
        
        # Save report
        with open(os.path.join(output_dir, "image_quality_report.txt"), 'w') as f:
            f.write(checker.get_detailed_report())
    
    # Print report
    print(checker.get_detailed_report())
    
    # Visualize results
    if visualize:
        checker.visualize_results(output_dir)
    
    return results


def check_video_for_point_cloud(video_path: str, 
                               output_dir: Optional[str] = None,
                               min_images: int = 20,
                               min_resolution: Tuple[int, int] = (1080, 720),
                               sample_rate: int = 1,
                               visualize: bool = True) -> Dict:
    """
    Convenience function to check if a video is suitable for point cloud generation.
    
    Args:
        video_path: Path to video file
        output_dir: Optional directory to save results and visualizations
        min_images: Minimum number of frames required
        min_resolution: Minimum frame resolution (width, height)
        sample_rate: Sample every nth frame
        visualize: Whether to visualize results
        
    Returns:
        Dictionary with check results
    """
    checker = PointCloudQualityPreCheck(
        min_images=min_images,
        min_resolution=min_resolution
    )
    
    results = checker.check_video(video_path, sample_rate=sample_rate)
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results
        checker.save_results(os.path.join(output_dir, "video_quality_results.json"))
        
        # Save report
        with open(os.path.join(output_dir, "video_quality_report.txt"), 'w') as f:
            f.write(checker.get_detailed_report())
    
    # Print report
    print(checker.get_detailed_report())
    
    # Visualize results
    if visualize:
        checker.visualize_results(output_dir)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check if images or video are suitable for point cloud generation")
    parser.add_argument("--input", required=True, help="Path to image folder or video file")
    parser.add_argument("--type", choices=["images", "video", "auto"], default="auto", 
                       help="Input type (images, video, or auto to detect)")
    parser.add_argument("--output-dir", help="Directory to save results and visualizations")
    parser.add_argument("--min-images", type=int, default=20, help="Minimum number of images required")
    parser.add_argument("--min-width", type=int, default=1080, help="Minimum image width")
    parser.add_argument("--min-height", type=int, default=720, help="Minimum image height")
    parser.add_argument("--min-texture", type=float, default=0.3, help="Minimum texture score (0-1)")
    parser.add_argument("--max-blur", type=float, default=100.0, help="Maximum blur threshold")
    parser.add_argument("--min-overlap", type=float, default=0.5, help="Minimum overlap score (0-1)")
    parser.add_argument("--min-viewpoint", type=float, default=15.0, help="Minimum viewpoint variation (degrees)")
    parser.add_argument("--max-viewpoint", type=float, default=45.0, help="Maximum viewpoint variation (degrees)")
    parser.add_argument("--max-exposure", type=float, default=0.5, help="Maximum exposure variation")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sample every nth frame for video")
    parser.add_argument("--no-visualize", action="store_true", help="Don't visualize results")
    
    args = parser.parse_args()
    
    # Create quality checker
    checker = PointCloudQualityPreCheck(
        min_images=args.min_images,
        min_resolution=(args.min_width, args.min_height),
        min_texture_score=args.min_texture,
        max_blur_threshold=args.max_blur,
        min_overlap_score=args.min_overlap,
        min_viewpoint_variation=args.min_viewpoint,
        max_viewpoint_variation=args.max_viewpoint,
        max_exposure_variation=args.max_exposure
    )
    
    # Determine input type if auto
    if args.type == "auto":
        if os.path.isdir(args.input):
            args.type = "images"
        elif os.path.isfile(args.input) and args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            args.type = "video"
        else:
            print(f"Error: Could not automatically determine input type for {args.input}")
            sys.exit(1)
    
    # Run checks based on input type
    if args.type == "images":
        results = checker.check_image_folder(args.input)
    else:  # video
        results = checker.check_video(args.input, sample_rate=args.sample_rate)
    
    # Save results
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Save results
        checker.save_results(os.path.join(args.output_dir, f"{args.type}_quality_results.json"))
        
        # Save report
        with open(os.path.join(args.output_dir, f"{args.type}_quality_report.txt"), 'w') as f:
            f.write(checker.get_detailed_report())
    
    # Print report
    print(checker.get_detailed_report())
    
    # Visualize results
    if not args.no_visualize:
        checker.visualize_results(args.output_dir)
