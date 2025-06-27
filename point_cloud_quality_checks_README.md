# Point Cloud Quality Checks

A comprehensive suite of quality check tools for point cloud generation from images and videos. This package provides quality checks for both pre-processing (input data) and post-processing (generated point clouds) stages of the point cloud generation pipeline.

## Overview

Point cloud generation from images or videos using techniques like COLMAP, Structure from Motion (SfM), and Neural Radiance Fields (NeRF) requires high-quality input data to produce accurate and complete 3D reconstructions. This package provides tools to assess the quality of input data and generated point clouds, helping to identify potential issues before they lead to failed or poor-quality reconstructions.

### Pre-processing Quality Checks

The pre-processing quality checks analyze input images or video frames to determine if they are suitable for point cloud generation. These checks include:

1. **Image Count**: Ensures there are enough images for reconstruction
2. **Resolution**: Verifies images have sufficient resolution
3. **Texture**: Measures the amount of texture/detail in images
4. **Blur Detection**: Identifies and quantifies image blur
5. **Overlap**: Estimates the overlap between consecutive images
6. **Viewpoint Variation**: Analyzes camera movement between images
7. **Exposure Consistency**: Checks for consistent lighting/exposure

### Post-processing Quality Checks

The post-processing quality checks analyze generated point clouds to evaluate their quality. These checks include:

1. **Point Count**: Ensures the point cloud has enough points
2. **Density**: Measures the density of points in 3D space
3. **Noise**: Quantifies the amount of noise in the point cloud
4. **Completeness**: Evaluates how completely the point cloud covers the subject
5. **Outliers**: Identifies and quantifies outlier points
6. **Color Consistency**: Checks for consistent coloring across the point cloud

## Installation

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from point_cloud_quality_checks import run_quality_pipeline

# Run the complete quality check pipeline
results = run_quality_pipeline(
    input_path="path/to/images",  # Folder of images or video file
    point_cloud_path="path/to/pointcloud.ply",  # Optional point cloud file
    output_dir="results",  # Directory to save results
    input_type="auto",  # Automatically detect input type (images or video)
    visualize=True  # Generate visualizations
)

# Check results
if results["preprocessing"]["passed"]:
    print("Input data is suitable for point cloud generation")
else:
    print("Input data may not be suitable for point cloud generation")

if results["postprocessing"] and results["postprocessing"]["passed"]:
    print("Generated point cloud has good quality")
else:
    print("Generated point cloud may have quality issues")
```

### Pre-processing Checks Only

```python
from point_cloud_quality_checks_preprocessing import check_images_for_point_cloud

# Check a folder of images
results = check_images_for_point_cloud(
    folder_path="path/to/images",
    output_dir="results",
    min_images=20,
    min_resolution=(1080, 720),
    visualize=True
)

# Check results
if results["passed"]:
    print("Images are suitable for point cloud generation")
else:
    print("Images may not be suitable for point cloud generation")
    
    # Check specific issues
    if not results["image_count_check"]:
        print(f"Not enough images: {results['image_count']} (minimum required: {results['min_images']})")
    
    if not results["resolution_check"]:
        print(f"Resolution too low: {results['resolution']} (minimum required: {results['min_resolution']})")
    
    # ... check other specific issues
```

### Post-processing Checks Only

```python
from point_cloud_quality_checks_postprocessing import check_point_cloud_quality

# Check a point cloud file
results = check_point_cloud_quality(
    point_cloud_path="path/to/pointcloud.ply",
    output_dir="results",
    min_points=10000,
    min_density=100.0,
    visualize=True
)

# Check results
if results["passed"]:
    print("Point cloud has good quality")
else:
    print("Point cloud may have quality issues")
    
    # Check specific issues
    if not results["point_count_check"]:
        print(f"Not enough points: {results['point_count']} (minimum required: {results['min_points']})")
    
    if not results["density_check"]:
        print(f"Density too low: {results['density']} (minimum required: {results['min_density']})")
    
    # ... check other specific issues
```

### Custom Pipeline

```python
from point_cloud_quality_checks import PointCloudQualityPipeline

# Create pipeline
pipeline = PointCloudQualityPipeline(output_dir="results")

# Check input data with custom parameters
preprocessing_results = pipeline.check_input_data(
    input_path="path/to/images",
    input_type="images",
    min_images=30,  # Require more images
    min_resolution=(1920, 1080),  # Require higher resolution
    min_texture_score=0.4,  # Require more texture
    max_blur_threshold=150.0,  # Allow more blur
    min_overlap_score=0.6,  # Require more overlap
    min_viewpoint_variation=10.0,  # Allow smaller camera movements
    max_viewpoint_variation=40.0,  # Limit maximum camera movement
    max_exposure_variation=0.4,  # Allow more exposure variation
    visualize=True
)

# Check point cloud with custom parameters
postprocessing_results = pipeline.check_point_cloud(
    point_cloud_path="path/to/pointcloud.ply",
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
```

### Command Line Usage

```bash
# Check input data
python point_cloud_quality_checks.py --input path/to/images --output-dir results

# Check input data and point cloud
python point_cloud_quality_checks.py --input path/to/images --point-cloud path/to/pointcloud.ply --output-dir results

# Check video input
python point_cloud_quality_checks.py --input path/to/video.mp4 --type video --output-dir results
```

## Best Practices for Point Cloud Generation

Based on the quality checks implemented in this package, here are some best practices for capturing images or videos for point cloud generation:

### Image Capture

1. **Capture enough images**: Aim for at least 20-30 images for small objects, and 50+ for larger scenes.
2. **Use high resolution**: Higher resolution images capture more detail. Aim for at least 1080p resolution.
3. **Ensure good lighting**: Use consistent, diffuse lighting to avoid shadows and highlights.
4. **Avoid motion blur**: Use a tripod or fast shutter speed to minimize blur.
5. **Capture with overlap**: Each part of the subject should appear in at least 3 different images.
6. **Vary viewpoint gradually**: Move the camera in small increments (15-30 degrees) between shots.
7. **Maintain consistent exposure**: Use manual exposure settings to keep lighting consistent.
8. **Capture all sides**: Ensure complete coverage of the subject from all angles.
9. **Include texture**: Avoid plain, textureless surfaces. Add temporary texture if needed.
10. **Avoid reflective surfaces**: Reflections can confuse reconstruction algorithms.

### Video Capture

1. **Move slowly**: Move the camera slowly to minimize motion blur.
2. **Orbit the subject**: Move around the subject in a complete orbit.
3. **Vary elevation**: Capture the subject from different heights.
4. **Use high framerate**: Higher framerates provide more images to work with.
5. **Use high resolution**: Aim for at least 1080p resolution, preferably 4K.
6. **Ensure good lighting**: Use consistent, diffuse lighting.
7. **Avoid quick exposure changes**: Maintain consistent exposure throughout the video.

## Technical Details

### Pre-processing Quality Metrics

- **Image Count**: The number of images or video frames available for reconstruction.
- **Resolution**: The width and height of the images in pixels.
- **Texture Score**: A measure of the amount of texture/detail in the images, calculated using gradient magnitude.
- **Blur Score**: A measure of image sharpness, calculated using Laplacian variance.
- **Overlap Score**: An estimate of the overlap between consecutive images, calculated using feature matching.
- **Viewpoint Variation**: The estimated camera movement between consecutive images, calculated using homography decomposition.
- **Exposure Variation**: The variation in brightness/exposure between images, calculated using histogram comparison.

### Post-processing Quality Metrics

- **Point Count**: The total number of points in the point cloud.
- **Density**: The number of points per cubic meter, calculated using the point cloud's bounding box.
- **Noise Ratio**: The ratio of noise points to total points, estimated using statistical outlier removal.
- **Completeness**: A measure of how completely the point cloud covers the subject, estimated using voxel grid occupancy.
- **Outlier Ratio**: The ratio of outlier points to total points, estimated using radius outlier removal.
- **Color Consistency**: A measure of the consistency of point colors, calculated using color space analysis.

## References

- COLMAP: https://colmap.github.io/
- NeRF Studio: https://docs.nerf.studio/
- Structure from Motion: https://en.wikipedia.org/wiki/Structure_from_motion
- Neural Radiance Fields: https://www.matthewtancik.com/nerf

## License

This project is licensed under the MIT License - see the LICENSE file for details.
