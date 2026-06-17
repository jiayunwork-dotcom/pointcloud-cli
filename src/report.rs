use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudStats {
    pub initial_points: usize,
    pub points_after_filter: Option<usize>,
    pub points_after_downsample: Option<usize>,
    pub points_after_ground_removal: Option<usize>,
    pub has_normals: bool,
    pub point_density: Option<f64>,
    pub bounding_box_min: [f64; 3],
    pub bounding_box_max: [f64; 3],
    pub centroid: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStats {
    pub filter_type: String,
    pub removed_count: usize,
    pub removed_ratio: f64,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampleStats {
    pub downsample_type: String,
    pub original_count: usize,
    pub final_count: usize,
    pub compression_ratio: f64,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundRemovalStats {
    pub ground_points: usize,
    pub non_ground_points: usize,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalStats {
    pub k: usize,
    pub mean_curvature: f64,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationStats {
    pub register_type: String,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
    pub transform_matrix: [[f64; 4]; 4],
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionStats {
    pub reconstruct_type: String,
    pub vertices: usize,
    pub faces: usize,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshProcessingStats {
    pub holes_filled: Option<usize>,
    pub triangles_added: Option<usize>,
    pub simplify_original_faces: Option<usize>,
    pub simplify_final_faces: Option<usize>,
    pub simplify_error: Option<f64>,
    pub smooth_iterations: Option<u32>,
    pub smooth_avg_movement: Option<f64>,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationStats {
    pub planes_detected: usize,
    pub clusters: Option<usize>,
    pub largest_cluster: Option<usize>,
    pub time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileProcessingReport {
    pub file_path: String,
    pub success: bool,
    pub error_message: Option<String>,
    pub total_time_ms: u64,
    pub point_cloud_stats: Option<PointCloudStats>,
    pub filter_stats: Option<Vec<FilterStats>>,
    pub downsample_stats: Option<DownsampleStats>,
    pub ground_removal_stats: Option<GroundRemovalStats>,
    pub normal_stats: Option<NormalStats>,
    pub registration_stats: Option<RegistrationStats>,
    pub reconstruction_stats: Option<ReconstructionStats>,
    pub mesh_stats: Option<MeshProcessingStats>,
    pub segmentation_stats: Option<SegmentationStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub generated_at: String,
    pub total_files: usize,
    pub successful_files: usize,
    pub failed_files: usize,
    pub total_time_ms: u64,
    pub input_files: Vec<String>,
    pub output_files: Vec<String>,
    pub per_file: HashMap<String, FileProcessingReport>,
    pub summary: PipelineSummary,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineSummary {
    pub total_initial_points: usize,
    pub total_final_points: usize,
    pub total_mesh_vertices: usize,
    pub total_mesh_faces: usize,
    pub total_planes_detected: usize,
    pub total_clusters_found: usize,
    pub average_processing_time_ms: f64,
}

impl Default for PipelineReport {
    fn default() -> Self {
        PipelineReport {
            generated_at: chrono_like_now(),
            total_files: 0,
            successful_files: 0,
            failed_files: 0,
            total_time_ms: 0,
            input_files: Vec::new(),
            output_files: Vec::new(),
            per_file: HashMap::new(),
            summary: PipelineSummary::default(),
        }
    }
}

fn chrono_like_now() -> String {
    use chrono::Utc;
    Utc::now().to_rfc3339()
}

impl PipelineReport {
    pub fn save_to_json(&self, path: &std::path::Path) -> crate::error::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::PointCloudError::JsonError(e))?;
        std::fs::write(path, json)
            .map_err(|e| crate::error::PointCloudError::IoError(e))?;
        Ok(())
    }

    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Pipeline 处理报告摘要");
        println!("{}", "=".repeat(60));
        println!("生成时间: {}", self.generated_at);
        println!("总文件数: {}", self.total_files);
        println!("成功: {}  失败: {}", self.successful_files, self.failed_files);
        println!("总耗时: {:.2}s", self.total_time_ms as f64 / 1000.0);
        println!();
        println!("统计汇总:");
        println!("  初始总点数: {}", self.summary.total_initial_points);
        println!("  最终总点数: {}", self.summary.total_final_points);
        println!("  总网格顶点: {}", self.summary.total_mesh_vertices);
        println!("  总网格面片: {}", self.summary.total_mesh_faces);
        println!("  检测平面数: {}", self.summary.total_planes_detected);
        println!("  聚类总数: {}", self.summary.total_clusters_found);
        println!("  平均处理时间: {:.2}s", self.summary.average_processing_time_ms / 1000.0);
        println!("{}", "=".repeat(60));

        for (file, report) in &self.per_file {
            println!();
            let status = if report.success { "✓ 成功" } else { "✗ 失败" };
            println!("{} [{}] ({:.2}s)", file, status, report.total_time_ms as f64 / 1000.0);
            if let Some(ref pc) = report.point_cloud_stats {
                println!("  点云: {} -> {} 点", pc.initial_points,
                    pc.points_after_ground_removal
                        .or(pc.points_after_downsample)
                        .or(pc.points_after_filter)
                        .unwrap_or(pc.initial_points));
            }
            if let Some(ref rs) = report.reconstruction_stats {
                println!("  重建: {} 顶点, {} 面片", rs.vertices, rs.faces);
            }
            if let Some(ref e) = report.error_message {
                println!("  错误: {}", e);
            }
        }
    }
}

pub fn duration_to_ms(d: Duration) -> u64 {
    d.as_secs() * 1000 + d.subsec_millis() as u64
}
