# Point Cloud Quality Checks

A comprehensive suite of quality check tools for point cloud generation from images and videos. This package provides quality checks for both pre-processing (input data) and post-processing (generated point clouds) stages of the point cloud generation pipeline.

## Overview

This project implements quality checks for point cloud generation using techniques like COLMAP and NeRF Studio. The quality checks help determine whether images or videos collected will create a successful point cloud before the actual reconstruction process begins, saving time and computational resources.

The quality checks are divided into two main categories:

1. **Pre-processing Quality Checks**: Analyze input images or video frames to determine if they are suitable for point cloud generation.
2. **Post-processing Quality Checks**: Analyze generated point clouds to evaluate their quality.

## Features

### Pre-processing Quality Checks

- **Image Count**: Ensures there are enough images for reconstruction
- **Resolution**: Verifies images have sufficient resolution
- **Texture**: Measures the amount of texture/detail in images
- **Blur Detection**: Identifies and quantifies image blur
- **Overlap**: Estimates the overlap between consecutive images
- **Viewpoint Variation**: Analyzes camera movement between images
- **Exposure Consistency**: Checks for consistent lighting/exposure

### Post-processing Quality Checks

- **Point Count**: Ensures the point cloud has enough points
- **Density**: Measures the density of points in 3D space
- **Noise**: Quantifies the amount of noise in the point cloud
- **Completeness**: Evaluates how completely the point cloud covers the subject
- **Outliers**: Identifies and quantifies outlier points
- **Color Consistency**: Checks for consistent coloring across the point cloud

### Visualization

- Comprehensive visualizations of quality check results
- Individual and combined reports
- Detailed metrics and pass/fail indicators

## Installation

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

See the [detailed documentation](point_cloud_quality_checks_README.md) for comprehensive usage examples and API reference.

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
```

### Command Line Usage

```bash
# Check input data
python point_cloud_quality_checks.py --input path/to/images --output-dir results

# Check input data and point cloud
python point_cloud_quality_checks.py --input path/to/images --point-cloud path/to/pointcloud.ply --output-dir results
```

## Project Structure

- `point_cloud_quality_checks.py`: Main module providing a unified interface
- `point_cloud_quality_checks_preprocessing.py`: Pre-processing quality checks
- `point_cloud_quality_checks_postprocessing.py`: Post-processing quality checks
- `point_cloud_quality_visualization.py`: Visualization tools
- `example_usage.py`: Example usage of the quality check modules
- `requirements.txt`: Required dependencies
- `point_cloud_quality_checks_README.md`: Detailed documentation

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

## References

- COLMAP: https://colmap.github.io/
- NeRF Studio: https://docs.nerf.studio/
- Structure from Motion: https://en.wikipedia.org/wiki/Structure_from_motion
- Neural Radiance Fields: https://www.matthewtancik.com/nerf

## License

This project is licensed under the MIT License - see the LICENSE file for details.
