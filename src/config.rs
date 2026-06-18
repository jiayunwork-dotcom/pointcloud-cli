use serde::{Serialize, Deserialize};
use crate::error::{Result, PointCloudError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub input_dir: Option<String>,
    pub pipeline: Vec<PipelineStep>,
    pub output: String,
    #[serde(default)]
    pub output_intermediate: bool,
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub report_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "step", content = "params", rename_all = "snake_case")]
pub enum PipelineStep {
    Filter {
        #[serde(rename = "type", default = "default_filter_type")]
        filter_type: String,
        #[serde(default)]
        k: Option<usize>,
        #[serde(default)]
        std_ratio: Option<f64>,
        #[serde(default)]
        radius: Option<f64>,
        #[serde(default)]
        min_neighbors: Option<usize>,
    },
    Downsample {
        #[serde(rename = "type", default = "default_downsample_type")]
        downsample_type: String,
        #[serde(default)]
        voxel_size: Option<f64>,
    },
    RemoveGround {
        #[serde(default)]
        initial_window: Option<f64>,
        #[serde(default)]
        max_window: Option<f64>,
        #[serde(default)]
        cell_size: Option<f64>,
        #[serde(default)]
        slope_threshold: Option<f64>,
        #[serde(default)]
        height_threshold: Option<f64>,
        #[serde(default)]
        keep_only_non_ground: Option<bool>,
    },
    Normals {
        #[serde(default)]
        k: Option<usize>,
        #[serde(default)]
        orientation_k: Option<usize>,
    },
    Register {
        #[serde(rename = "type", default = "default_register_type")]
        register_type: String,
        #[serde(default)]
        fpfh_radius: Option<f64>,
        #[serde(default)]
        ransac_iterations: Option<usize>,
        #[serde(default)]
        icp_iterations: Option<usize>,
        #[serde(default)]
        icp_threshold: Option<f64>,
    },
    Reconstruct {
        #[serde(rename = "type", default = "default_reconstruct_type")]
        reconstruct_type: String,
        #[serde(default)]
        depth: Option<u32>,
        #[serde(default)]
        min_depth: Option<u32>,
        #[serde(default)]
        ball_radius: Option<f64>,
        #[serde(default)]
        resolution: Option<u32>,
        #[serde(default)]
        iso_value: Option<f64>,
    },
    FillHoles {
        #[serde(default)]
        max_hole_size: Option<usize>,
    },
    Simplify {
        #[serde(rename = "type", default = "default_simplify_type")]
        simplify_type: String,
        #[serde(default)]
        target_faces: Option<usize>,
        #[serde(default)]
        target_ratio: Option<f64>,
    },
    Smooth {
        #[serde(rename = "type", default = "default_smooth_type")]
        smooth_type: String,
        #[serde(default)]
        iterations: Option<u32>,
        #[serde(default)]
        lambda: Option<f64>,
    },
    Segment {
        #[serde(rename = "type", default = "default_segment_type")]
        segment_type: String,
        #[serde(default)]
        max_planes: Option<usize>,
        #[serde(default)]
        plane_distance: Option<f64>,
        #[serde(default)]
        cluster_tolerance: Option<f64>,
        #[serde(default)]
        min_cluster_size: Option<usize>,
    },
    Quality {
        #[serde(default)]
        threshold: Option<f64>,
        #[serde(default)]
        weights: Option<Vec<f64>>,
        #[serde(default)]
        assess_completeness: Option<bool>,
        #[serde(default)]
        auto_fix: Option<bool>,
        #[serde(default)]
        octree_depth: Option<usize>,
        #[serde(default)]
        noise_k: Option<usize>,
        #[serde(default)]
        diff_with: Option<String>,
    },
    Align {
        source: Option<String>,
        target: Option<String>,
        #[serde(default)]
        ransac_iterations: Option<usize>,
        #[serde(default)]
        icp_iterations: Option<usize>,
        #[serde(default)]
        icp_threshold: Option<f64>,
        #[serde(default)]
        voxel_size: Option<f64>,
        #[serde(default)]
        normal_radius: Option<f64>,
        #[serde(default)]
        fpfh_radius: Option<f64>,
        #[serde(default)]
        rmse_warning_threshold: Option<f64>,
        #[serde(default)]
        output_transformed_source: Option<String>,
        #[serde(default)]
        output_matrix: Option<String>,
        #[serde(default)]
        pass_transform_to_next: Option<bool>,
    },
}

fn default_filter_type() -> String { "statistical".to_string() }
fn default_downsample_type() -> String { "voxel".to_string() }
fn default_register_type() -> String { "icp".to_string() }
fn default_reconstruct_type() -> String { "poisson".to_string() }
fn default_simplify_type() -> String { "qem".to_string() }
fn default_smooth_type() -> String { "laplacian".to_string() }
fn default_segment_type() -> String { "planes".to_string() }

impl PipelineConfig {
    pub fn from_yaml_file(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| PointCloudError::ConfigError(format!("无法打开配置文件: {}", e)))?;
        let reader = std::io::BufReader::new(file);
        let config: PipelineConfig = serde_yaml::from_reader(reader)
            .map_err(|e| PointCloudError::YamlError(e))?;
        Ok(config)
    }

    pub fn from_yaml_str(s: &str) -> Result<Self> {
        let config: PipelineConfig = serde_yaml::from_str(s)
            .map_err(|e| PointCloudError::YamlError(e))?;
        Ok(config)
    }
}

pub fn example_config_yaml() -> String {
    r#"input:
  - scan1.ply
  - scan2.ply

pipeline:
  - step: align
    params:
      source: source.ply
      target: target.ply
      ransac_iterations: 1000
      icp_iterations: 50
      icp_threshold: 1e-7
      voxel_size: 0.0
      normal_radius: 0.0
      rmse_warning_threshold: 0.05
      output_transformed_source: aligned_source.ply
      output_matrix: transform_matrix.txt
      pass_transform_to_next: true

  - step: filter
    params:
      type: statistical
      k: 30
      std_ratio: 1.5

  - step: downsample
    params:
      type: voxel
      voxel_size: 0.05

  - step: normals
    params:
      k: 20

  - step: register
    params:
      type: icp
      fpfh_radius: 0.15
      icp_iterations: 100

  - step: reconstruct
    params:
      type: poisson
      depth: 8

  - step: simplify
    params:
      type: qem
      target_ratio: 0.5

  - step: smooth
    params:
      type: laplacian
      iterations: 20
      lambda: 0.5

  - step: quality
    params:
      threshold: 60
      assess_completeness: false
      auto_fix: false

output: result.obj

output_intermediate: true
output_dir: ./output
report_path: report.json
"#.to_string()
}
