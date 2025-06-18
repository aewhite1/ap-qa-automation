# Point Cloud Quality Checks for AgTech 3D Models
# This module implements quality checks for 3D point clouds in Databricks pipelines

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Correlation
import numpy as np

# Register UDFs for point cloud analysis
@F.udf(returnType=BooleanType())
def check_ground_plane_orientation(points_json):
    """
    Check if the ground plane is properly oriented (not flipped or severely tilted).
    
    Args:
        points_json: JSON string containing point cloud data with x, y, z coordinates
        
    Returns:
        Boolean indicating if the ground plane orientation is valid
    """
    try:
        import json
        from sklearn.linear_model import RANSACRegressor
        
        # Parse points from JSON
        points = json.loads(points_json)
        
        # Extract x, y, z coordinates
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]
        
        # Create feature matrix (x, y) and target (z)
        X = np.column_stack((xs, ys))
        y = np.array(zs)
        
        # Fit RANSAC regressor to find ground plane
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        
        # Get plane normal vector (coefficients and intercept)
        coef = ransac.estimator_.coef_
        
        # Calculate angle between normal vector and vertical (0, 0, 1)
        normal_vector = np.append(coef, 1.0)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        vertical = np.array([0, 0, 1])
        
        # Calculate angle in degrees
        angle = np.arccos(np.dot(normal_vector, vertical)) * 180 / np.pi
        
        # Check if angle is within threshold (e.g., 30 degrees)
        return angle < 30
    except:
        return False

@F.udf(returnType=BooleanType())
def check_point_density_distribution(points_json):
    """
    Check if the point cloud has a relatively uniform density distribution.
    
    Args:
        points_json: JSON string containing point cloud data with x, y, z coordinates
        
    Returns:
        Boolean indicating if the point density distribution is valid
    """
    try:
        import json
        import numpy as np
        from scipy.spatial import ConvexHull
        
        # Parse points from JSON
        points = json.loads(points_json)
        
        # Extract x, y, z coordinates
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]
        
        # Create point array
        point_array = np.column_stack((xs, ys, zs))
        
        # Compute convex hull
        hull = ConvexHull(point_array)
        
        # Compute volume of convex hull
        volume = hull.volume
        
        # Compute point density (points per unit volume)
        density = len(points) / volume
        
        # Divide space into octants and check density in each
        octant_counts = [0] * 8
        center = np.mean(point_array, axis=0)
        
        for point in point_array:
            # Determine octant (0-7)
            octant = 0
            if point[0] > center[0]: octant |= 1
            if point[1] > center[1]: octant |= 2
            if point[2] > center[2]: octant |= 4
            octant_counts[octant] += 1
        
        # Calculate density variation across octants
        octant_densities = [count / len(points) for count in octant_counts]
        density_variation = np.std(octant_densities)
        
        # Check if density variation is within threshold
        # Lower variation means more uniform distribution
        return density_variation < 0.2
    except:
        return False

@F.udf(returnType=BooleanType())
def check_noise_level(points_json):
    """
    Check if the point cloud has acceptable noise levels.
    
    Args:
        points_json: JSON string containing point cloud data with x, y, z coordinates
        
    Returns:
        Boolean indicating if the noise level is acceptable
    """
    try:
        import json
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        
        # Parse points from JSON
        points = json.loads(points_json)
        
        # Extract x, y, z coordinates
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]
        
        # Create point array
        point_array = np.column_stack((xs, ys, zs))
        
        # Find k nearest neighbors for each point
        k = 10  # Number of neighbors to consider
        nbrs = NearestNeighbors(n_neighbors=k).fit(point_array)
        distances, indices = nbrs.kneighbors(point_array)
        
        # Calculate average distance to neighbors for each point
        avg_distances = np.mean(distances, axis=1)
        
        # Calculate global average and standard deviation
        global_avg = np.mean(avg_distances)
        global_std = np.std(avg_distances)
        
        # Count outliers (points with avg distance > global_avg + 2*global_std)
        outlier_threshold = global_avg + 2 * global_std
        outlier_count = np.sum(avg_distances > outlier_threshold)
        outlier_ratio = outlier_count / len(points)
        
        # Check if outlier ratio is within threshold
        return outlier_ratio < 0.1
    except:
        return False

@F.udf(returnType=BooleanType())
def check_arc_distortion(points_json):
    """
    Check if the point cloud has arc distortion (one side dropping off).
    
    Args:
        points_json: JSON string containing point cloud data with x, y, z coordinates
        
    Returns:
        Boolean indicating if there is no significant arc distortion
    """
    try:
        import json
        import numpy as np
        from sklearn.decomposition import PCA
        
        # Parse points from JSON
        points = json.loads(points_json)
        
        # Extract x, y, z coordinates
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]
        
        # Create point array
        point_array = np.column_stack((xs, ys, zs))
        
        # Perform PCA to find principal components
        pca = PCA(n_components=3)
        pca.fit(point_array)
        
        # Transform points to principal component space
        transformed = pca.transform(point_array)
        
        # Divide points into bins along the first principal component
        num_bins = 10
        bin_indices = np.floor(num_bins * (transformed[:, 0] - np.min(transformed[:, 0])) / 
                              (np.max(transformed[:, 0]) - np.min(transformed[:, 0]))).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Calculate average height (z in original space) for each bin
        bin_heights = []
        for i in range(num_bins):
            bin_points = point_array[bin_indices == i]
            if len(bin_points) > 0:
                bin_heights.append(np.mean(bin_points[:, 2]))
            else:
                bin_heights.append(np.nan)
        
        # Remove NaN values
        bin_heights = [h for h in bin_heights if not np.isnan(h)]
        
        # Check for monotonic decrease or increase (arc distortion)
        if len(bin_heights) < 3:
            return True  # Not enough bins to detect arc
        
        # Calculate differences between consecutive bins
        diffs = np.diff(bin_heights)
        
        # Check if all differences have the same sign (monotonic)
        all_increasing = np.all(diffs > 0)
        all_decreasing = np.all(diffs < 0)
        
        # If monotonically increasing or decreasing, it might indicate arc distortion
        monotonic = all_increasing or all_decreasing
        
        # Calculate the maximum height difference
        max_height_diff = np.max(bin_heights) - np.min(bin_heights)
        avg_height = np.mean(bin_heights)
        relative_diff = max_height_diff / avg_height
        
        # Check if the height difference is significant and monotonic
        return not (monotonic and relative_diff > 0.3)
    except:
        return False

@F.udf(returnType=BooleanType())
def check_completeness(points_json):
    """
    Check if the point cloud is complete (covers the entire object).
    
    Args:
        points_json: JSON string containing point cloud data with x, y, z coordinates
        
    Returns:
        Boolean indicating if the point cloud is complete
    """
    try:
        import json
        import numpy as np
        
        # Parse points from JSON
        points = json.loads(points_json)
        
        # Check if there are enough points
        if len(points) < 1000:
            return False
        
        # Extract x, y, z coordinates
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]
        
        # Create point array
        point_array = np.column_stack((xs, ys, zs))
        
        # Calculate bounding box
        min_x, min_y, min_z = np.min(point_array, axis=0)
        max_x, max_y, max_z = np.max(point_array, axis=0)
        
        # Calculate dimensions
        width = max_x - min_x
        depth = max_y - min_y
        height = max_z - min_z
        
        # Calculate aspect ratios
        aspect_ratio_1 = width / height
        aspect_ratio_2 = depth / height
        
        # Check if aspect ratios are reasonable
        # For trees/plants, we expect height > width and height > depth
        return aspect_ratio_1 < 2.0 and aspect_ratio_2 < 2.0
    except:
        return False

# Define DLT pipeline for point cloud quality checks
@dlt.table(
    name="point_cloud_quality_metrics",
    comment="Quality metrics for 3D point clouds"
)
@dlt.expect_all_or_drop({
    "valid_ground_plane": "ground_plane_valid = true",
    "valid_density_distribution": "density_distribution_valid = true",
    "acceptable_noise_level": "noise_level_valid = true",
    "no_arc_distortion": "no_arc_distortion = true",
    "complete_point_cloud": "completeness_valid = true"
})
def point_cloud_quality_metrics():
    """
    Compute quality metrics for 3D point clouds and apply quality checks.
    """
    # Read point cloud data from source
    # This assumes point cloud data is stored in a table with a 'points' column containing JSON
    return (
        spark.readStream.table("point_clouds")
        .withColumn("ground_plane_valid", check_ground_plane_orientation(F.col("points")))
        .withColumn("density_distribution_valid", check_point_density_distribution(F.col("points")))
        .withColumn("noise_level_valid", check_noise_level(F.col("points")))
        .withColumn("no_arc_distortion", check_arc_distortion(F.col("points")))
        .withColumn("completeness_valid", check_completeness(F.col("points")))
        .withColumn("quality_score", 
                   (F.col("ground_plane_valid").cast("int") + 
                    F.col("density_distribution_valid").cast("int") + 
                    F.col("noise_level_valid").cast("int") + 
                    F.col("no_arc_distortion").cast("int") + 
                    F.col("completeness_valid").cast("int")) / 5.0)
    )

# Define a view for point clouds that pass all quality checks
@dlt.view(
    name="valid_point_clouds",
    comment="Point clouds that pass all quality checks"
)
def valid_point_clouds():
    """
    Filter point clouds that pass all quality checks.
    """
    return (
        dlt.read("point_cloud_quality_metrics")
        .filter(
            (F.col("ground_plane_valid") == True) &
            (F.col("density_distribution_valid") == True) &
            (F.col("noise_level_valid") == True) &
            (F.col("no_arc_distortion") == True) &
            (F.col("completeness_valid") == True)
        )
    )

# Define a view for point clouds that fail quality checks
@dlt.view(
    name="invalid_point_clouds",
    comment="Point clouds that fail one or more quality checks"
)
def invalid_point_clouds():
    """
    Filter point clouds that fail one or more quality checks.
    """
    return (
        dlt.read("point_cloud_quality_metrics")
        .filter(
            (F.col("ground_plane_valid") == False) |
            (F.col("density_distribution_valid") == False) |
            (F.col("noise_level_valid") == False) |
            (F.col("no_arc_distortion") == False) |
            (F.col("completeness_valid") == False)
        )
        .withColumn("failure_reasons", 
                   F.concat_ws(", ",
                              F.when(F.col("ground_plane_valid") == False, F.lit("Invalid ground plane orientation")).otherwise(F.lit("")),
                              F.when(F.col("density_distribution_valid") == False, F.lit("Non-uniform density distribution")).otherwise(F.lit("")),
                              F.when(F.col("noise_level_valid") == False, F.lit("Excessive noise")).otherwise(F.lit("")),
                              F.when(F.col("no_arc_distortion") == False, F.lit("Arc distortion detected")).otherwise(F.lit("")),
                              F.when(F.col("completeness_valid") == False, F.lit("Incomplete point cloud")).otherwise(F.lit(""))
                             ))
    )
