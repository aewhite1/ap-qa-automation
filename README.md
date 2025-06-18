# Point Cloud Quality Checks for AgTech 3D Models

This repository contains implementations of quality checks for 3D point clouds in Databricks pipelines. These checks are designed to identify common issues in point clouds generated from smartphone cameras for AgTech applications, such as:

- Flipped or tilted point clouds
- Excessive noise
- Arc distortion (one side dropping off)
- Incomplete point clouds
- Non-uniform density distribution

## Overview

The quality checks are implemented in both Python and SQL for use in Databricks Delta Live Tables (DLT) pipelines. The checks are designed to be integrated into your existing data processing pipeline to catch problematic point clouds before they proceed to downstream processing.

## Implementation Options

### Python Implementation (`point_cloud_quality_checks.py`)

The Python implementation uses PySpark and Delta Live Tables to define a pipeline that:

1. Reads point cloud data from a source table
2. Applies multiple quality checks using User-Defined Functions (UDFs)
3. Computes quality metrics and scores
4. Filters point clouds into "valid" and "invalid" categories
5. Provides detailed failure reasons for invalid point clouds

### SQL Implementation (`point_cloud_quality_checks.sql`)

The SQL implementation provides the same functionality but uses SQL syntax for those who prefer working with SQL in Databricks:

1. Creates a streaming table with quality constraints
2. Applies UDFs to perform quality checks
3. Creates views for valid and invalid point clouds
4. Provides historical quality metrics and failure pattern analysis

## Quality Checks

The following quality checks are implemented:

1. **Ground Plane Orientation Check**: Detects if the point cloud is flipped upside down or severely tilted off the ground axis.
2. **Point Density Distribution Check**: Ensures the point cloud has a relatively uniform density distribution.
3. **Noise Level Check**: Identifies point clouds with excessive noise that might distort the plant structure.
4. **Arc Distortion Check**: Detects if the point cloud has an arc distortion where one side drops off.
5. **Completeness Check**: Verifies if the point cloud covers the entire object with sufficient points.

## Integration with Databricks

### Prerequisites

- Databricks workspace with Delta Live Tables enabled (Premium plan)
- PySpark and necessary libraries (numpy, scikit-learn, scipy)
- Point cloud data stored in a table with a 'points' column containing JSON data

### Setup Instructions

1. **Upload the implementation files to your Databricks workspace**:
   - Upload `point_cloud_quality_checks.py` or `point_cloud_quality_checks.sql` to your workspace
   - Choose the implementation that best fits your workflow (Python or SQL)

2. **Create a Delta Live Tables pipeline**:
   - Go to Workflows > Delta Live Tables > Create Pipeline
   - Configure your pipeline settings (cluster size, target schema, etc.)
   - Add the implementation file as a notebook or SQL file
   - Set the development mode to "Production" for production use

3. **Configure the pipeline**:
   - Ensure your point cloud data is available in a table named `point_clouds` with a `points` column
   - Adjust thresholds in the quality check functions if needed
   - Set up appropriate scheduling based on your data ingestion frequency

4. **Run the pipeline**:
   - Start the pipeline to process your point cloud data
   - Monitor the pipeline execution in the DLT UI
   - Check the Data Quality tab to view quality metrics

### Monitoring and Analysis

The pipeline provides several ways to monitor and analyze point cloud quality:

1. **Data Quality Metrics**: View quality metrics directly in the DLT UI under the Data Quality tab.
2. **Valid/Invalid Point Clouds**: Query the `valid_point_clouds` and `invalid_point_clouds` views to see which point clouds passed or failed the quality checks.
3. **Failure Reasons**: The `invalid_point_clouds` view includes detailed failure reasons for each point cloud.
4. **Historical Trends**: The `point_cloud_quality_history` materialized view tracks quality metrics over time.
5. **Failure Patterns**: The `point_cloud_failure_patterns` materialized view identifies common failure patterns.

## Customization

### Adjusting Thresholds

You can adjust the thresholds in the quality check functions to match your specific requirements:

- In `check_ground_plane_orientation()`, adjust the angle threshold (default: 30 degrees)
- In `check_point_density_distribution()`, adjust the density variation threshold (default: 0.2)
- In `check_noise_level()`, adjust the outlier ratio threshold (default: 0.1)
- In `check_arc_distortion()`, adjust the relative difference threshold (default: 0.3)
- In `check_completeness()`, adjust the aspect ratio thresholds (default: 2.0)

### Adding New Checks

To add new quality checks:

1. Implement a new UDF in the Python file or register a new UDF for the SQL implementation
2. Add the new check to the `point_cloud_quality_metrics` table/view
3. Update the `valid_point_clouds` and `invalid_point_clouds` views to include the new check
4. Add the new check to the failure reasons calculation

## Example Usage

### Querying Valid Point Clouds

```sql
SELECT * FROM valid_point_clouds
```

### Querying Invalid Point Clouds with Failure Reasons

```sql
SELECT id, capture_date, failure_reasons 
FROM invalid_point_clouds
ORDER BY capture_date DESC
```

### Analyzing Quality Trends Over Time

```sql
SELECT 
  processing_date,
  total_point_clouds,
  avg_quality_score,
  valid_ground_plane_count / total_point_clouds AS ground_plane_success_rate,
  valid_noise_level_count / total_point_clouds AS noise_success_rate
FROM point_cloud_quality_history
ORDER BY processing_date
```

### Identifying Common Failure Patterns

```sql
SELECT failure_pattern, pattern_count
FROM point_cloud_failure_patterns
ORDER BY pattern_count DESC
```

## Best Practices

1. **Run quality checks early in the pipeline**: Catch issues before expensive processing steps.
2. **Monitor quality metrics over time**: Track trends to identify systematic issues in data collection.
3. **Adjust thresholds based on your data**: Fine-tune thresholds based on your specific use cases and equipment.
4. **Implement feedback loops**: Use failure patterns to improve data collection procedures.
5. **Combine with visual inspection**: For critical applications, combine automated checks with visual inspection of flagged point clouds.

## Troubleshooting

### Common Issues

1. **Performance issues with large point clouds**: 
   - Consider downsampling point clouds before quality checks
   - Increase cluster size for the DLT pipeline

2. **High false positive rate**: 
   - Adjust thresholds to be less strict
   - Add more sophisticated checks for your specific use case

3. **Missing dependencies**:
   - Ensure all required libraries are installed on your Databricks cluster
   - Use an init script to install dependencies if needed

### Getting Help

If you encounter issues with the quality checks, check:
- Databricks logs for error messages
- DLT pipeline execution details
- Data lineage in the DLT UI
