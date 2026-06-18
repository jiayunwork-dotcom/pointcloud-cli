use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;
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
pub struct BatchRegistrationFrameStats {
    pub frame_index: usize,
    pub source_file: String,
    pub transform_matrix: [[f64; 4]; 4],
    pub rmse: f64,
    pub converged: bool,
    pub has_warning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRegistrationStats {
    pub total_frames: usize,
    pub average_rmse: f64,
    pub max_rmse: f64,
    pub max_rmse_frame: usize,
    pub failed_frames: usize,
    pub warning_frames: usize,
    pub frames: Vec<BatchRegistrationFrameStats>,
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
pub struct QualityDiffInfo {
    pub reference_file: String,
    pub overall_score_before: f64,
    pub overall_score_after: f64,
    pub overall_change: f64,
    pub density_change: f64,
    pub normal_change: f64,
    pub overlap_change: f64,
    pub noise_change: f64,
    pub completeness_change: f64,
    pub degenerate_items: Vec<String>,
    pub has_degenerate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    pub overall_score: f64,
    pub density_score: f64,
    pub density_cv: f64,
    pub normal_score: f64,
    pub normal_flip_rate: f64,
    pub overlap_score: f64,
    pub overlap_rate: f64,
    pub noise_score: f64,
    pub noise_normalized: f64,
    pub completeness_score: f64,
    pub large_holes: usize,
    pub assessed_completeness: bool,
    pub threshold: f64,
    pub passed: bool,
    pub auto_fix: bool,
    pub points_added: Option<usize>,
    pub points_removed: Option<usize>,
    pub normals_fixed: Option<usize>,
    pub time_ms: u64,
    pub diff_info: Option<QualityDiffInfo>,
    pub has_warnings: bool,
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
    pub batch_registration_stats: Option<BatchRegistrationStats>,
    pub reconstruction_stats: Option<ReconstructionStats>,
    pub mesh_stats: Option<MeshProcessingStats>,
    pub segmentation_stats: Option<SegmentationStats>,
    pub quality_stats: Option<QualityStats>,
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
            if let Some(ref qs) = report.quality_stats {
                let q_color = if qs.overall_score >= 70.0 { "\x1b[32m" } else if qs.overall_score >= 40.0 { "\x1b[33m" } else { "\x1b[31m" };
                let status = if qs.has_warnings {
                    format!("{}警告{}", "\x1b[33m", "\x1b[0m")
                } else if qs.passed {
                    "通过".to_string()
                } else {
                    "未通过".to_string()
                };
                println!("  质量: {}{:.1}{}/100 ({})",
                    q_color, qs.overall_score, "\x1b[0m",
                    status
                );
                if let Some(ref diff) = qs.diff_info {
                    let change_color = if diff.overall_change >= 0.0 { "\x1b[32m" } else { "\x1b[31m" };
                    println!("    对比参考: {}", diff.reference_file);
                    println!("    综合评分: {:.1} → {:.1} ({}{:+.1}{})",
                        diff.overall_score_before, diff.overall_score_after,
                        change_color, diff.overall_change, "\x1b[0m");
                    if diff.has_degenerate {
                        println!("    {}退化项: {:?}{}", "\x1b[31m", diff.degenerate_items, "\x1b[0m");
                    }
                }
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

const SIGNIFICANT_THRESHOLD: f64 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiffItem {
    pub file_path: String,
    pub a_exists: bool,
    pub b_exists: bool,
    pub points_change_rate: Option<f64>,
    pub faces_diff: Option<i64>,
    pub faces_change_rate: Option<f64>,
    pub time_diff_ms: Option<i64>,
    pub time_change_rate: Option<f64>,
    pub success_changed: bool,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDiffSummary {
    pub total_time_ms_a: u64,
    pub total_time_ms_b: u64,
    pub total_time_diff_ms: i64,
    pub total_time_change_rate: f64,
    pub avg_compression_ratio_a: f64,
    pub avg_compression_ratio_b: f64,
    pub avg_compression_ratio_diff: f64,
    pub successful_files_a: usize,
    pub successful_files_b: usize,
    pub successful_diff: i64,
    pub failed_files_a: usize,
    pub failed_files_b: usize,
    pub failed_diff: i64,
    pub total_files_a: usize,
    pub total_files_b: usize,
    pub files_in_common: usize,
    pub files_only_in_a: usize,
    pub files_only_in_b: usize,
    pub significantly_changed_files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDiffResult {
    pub report_a_path: String,
    pub report_b_path: String,
    pub generated_at: String,
    pub summary: PipelineDiffSummary,
    pub per_file: Vec<FileDiffItem>,
}

fn final_points(stats: &PointCloudStats) -> usize {
    stats
        .points_after_ground_removal
        .or(stats.points_after_downsample)
        .or(stats.points_after_filter)
        .unwrap_or(stats.initial_points)
}

fn change_rate(a: f64, b: f64) -> f64 {
    if a.abs() < 1e-10 {
        if b.abs() < 1e-10 {
            0.0
        } else {
            1.0
        }
    } else {
        (b - a) / a
    }
}

fn compression_ratio_for_file(fr: &FileProcessingReport) -> f64 {
    if let Some(ref ds) = fr.downsample_stats {
        if ds.original_count > 0 {
            return ds.compression_ratio;
        }
    }
    if let Some(ref pc) = fr.point_cloud_stats {
        let initial = pc.initial_points as f64;
        let final_p = final_points(pc) as f64;
        if initial > 0.0 {
            return final_p / initial;
        }
    }
    1.0
}

impl PipelineReport {
    pub fn load_from_json(path: &Path) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::PointCloudError::IoError(e))?;
        let report: PipelineReport = serde_json::from_str(&content)
            .map_err(|e| crate::error::PointCloudError::JsonError(e))?;
        Ok(report)
    }
}

pub fn diff_reports(
    report_a: &PipelineReport,
    report_b: &PipelineReport,
    path_a: &str,
    path_b: &str,
) -> PipelineDiffResult {
    use std::collections::HashSet;

    let mut all_keys: Vec<String> = Vec::new();
    let mut key_set = HashSet::new();
    for k in report_a.per_file.keys() {
        if key_set.insert(k.clone()) {
            all_keys.push(k.clone());
        }
    }
    for k in report_b.per_file.keys() {
        if key_set.insert(k.clone()) {
            all_keys.push(k.clone());
        }
    }
    all_keys.sort();

    let mut per_file = Vec::new();
    let mut significantly_changed = 0usize;

    for key in &all_keys {
        let in_a = report_a.per_file.get(key);
        let in_b = report_b.per_file.get(key);
        let a_exists = in_a.is_some();
        let b_exists = in_b.is_some();

        let (points_change_rate, faces_diff, faces_change_rate, time_diff_ms, time_change_rate, success_changed) =
            match (in_a, in_b) {
                (Some(a), Some(b)) => {
                    let pts_a = a
                        .point_cloud_stats
                        .as_ref()
                        .map(|s| final_points(s) as f64);
                    let pts_b = b
                        .point_cloud_stats
                        .as_ref()
                        .map(|s| final_points(s) as f64);
                    let points_change_rate = match (pts_a, pts_b) {
                        (Some(a), Some(b)) => Some(change_rate(a, b)),
                        _ => None,
                    };

                    let faces_a = a
                        .reconstruction_stats
                        .as_ref()
                        .map(|r| r.faces as i64);
                    let faces_b = b
                        .reconstruction_stats
                        .as_ref()
                        .map(|r| r.faces as i64);
                    let (faces_diff, faces_change_rate) = match (faces_a, faces_b) {
                        (Some(fa), Some(fb)) => {
                            let diff = fb - fa;
                            let rate = change_rate(fa as f64, fb as f64);
                            (Some(diff), Some(rate))
                        }
                        (Some(fa), None) => (Some(-fa), None),
                        (None, Some(fb)) => (Some(fb), None),
                        (None, None) => (None, None),
                    };

                    let ta = a.total_time_ms as i64;
                    let tb = b.total_time_ms as i64;
                    let time_diff_ms = Some(tb - ta);
                    let time_change_rate = Some(change_rate(ta as f64, tb as f64));

                    let success_changed = a.success != b.success;

                    (
                        points_change_rate,
                        faces_diff,
                        faces_change_rate,
                        time_diff_ms,
                        time_change_rate,
                        success_changed,
                    )
                }
                _ => (None, None, None, None, None, false),
            };

        let mut significant = false;
        if success_changed {
            significant = true;
        }
        if let Some(r) = points_change_rate {
            if r.abs() > SIGNIFICANT_THRESHOLD {
                significant = true;
            }
        }
        if let Some(r) = faces_change_rate {
            if r.abs() > SIGNIFICANT_THRESHOLD {
                significant = true;
            }
        }
        if let Some(r) = time_change_rate {
            if r.abs() > SIGNIFICANT_THRESHOLD {
                significant = true;
            }
        }
        if significant {
            significantly_changed += 1;
        }

        per_file.push(FileDiffItem {
            file_path: key.clone(),
            a_exists,
            b_exists,
            points_change_rate,
            faces_diff,
            faces_change_rate,
            time_diff_ms,
            time_change_rate,
            success_changed,
            significant,
        });
    }

    let total_files_a = report_a.total_files;
    let total_files_b = report_b.total_files;
    let files_in_common = per_file.iter().filter(|f| f.a_exists && f.b_exists).count();
    let files_only_in_a = per_file.iter().filter(|f| f.a_exists && !f.b_exists).count();
    let files_only_in_b = per_file.iter().filter(|f| !f.a_exists && f.b_exists).count();

    let tta = report_a.total_time_ms as i64;
    let ttb = report_b.total_time_ms as i64;
    let total_time_diff_ms = ttb - tta;
    let total_time_change_rate = change_rate(tta as f64, ttb as f64);

    let avg_cr_a = if !report_a.per_file.is_empty() {
        report_a.per_file.values().map(compression_ratio_for_file).sum::<f64>()
            / report_a.per_file.len() as f64
    } else {
        0.0
    };
    let avg_cr_b = if !report_b.per_file.is_empty() {
        report_b.per_file.values().map(compression_ratio_for_file).sum::<f64>()
            / report_b.per_file.len() as f64
    } else {
        0.0
    };
    let avg_compression_ratio_diff = avg_cr_b - avg_cr_a;

    let successful_diff = report_b.successful_files as i64 - report_a.successful_files as i64;
    let failed_diff = report_b.failed_files as i64 - report_a.failed_files as i64;

    PipelineDiffResult {
        report_a_path: path_a.to_string(),
        report_b_path: path_b.to_string(),
        generated_at: chrono_like_now(),
        summary: PipelineDiffSummary {
            total_time_ms_a: report_a.total_time_ms,
            total_time_ms_b: report_b.total_time_ms,
            total_time_diff_ms,
            total_time_change_rate,
            avg_compression_ratio_a: avg_cr_a,
            avg_compression_ratio_b: avg_cr_b,
            avg_compression_ratio_diff,
            successful_files_a: report_a.successful_files,
            successful_files_b: report_b.successful_files,
            successful_diff,
            failed_files_a: report_a.failed_files,
            failed_files_b: report_b.failed_files,
            failed_diff,
            total_files_a,
            total_files_b,
            files_in_common,
            files_only_in_a,
            files_only_in_b,
            significantly_changed_files: significantly_changed,
        },
        per_file,
    }
}

fn red(s: &str) -> String {
    format!("\x1b[31m{}\x1b[0m", s)
}

fn green(s: &str) -> String {
    format!("\x1b[32m{}\x1b[0m", s)
}

fn yellow(s: &str) -> String {
    format!("\x1b[33m{}\x1b[0m", s)
}

fn cyan(s: &str) -> String {
    format!("\x1b[36m{}\x1b[0m", s)
}

fn bold(s: &str) -> String {
    format!("\x1b[1m{}\x1b[0m", s)
}

fn pct_str(rate: Option<f64>, sig: bool) -> String {
    match rate {
        None => "-".to_string(),
        Some(r) => {
            let s = format!("{:+.2}%", r * 100.0);
            if sig {
                red(&s)
            } else {
                s
            }
        }
    }
}

fn signed_i64(v: Option<i64>) -> String {
    match v {
        None => "-".to_string(),
        Some(v) => {
            if v > 0 {
                format!("+{}", v)
            } else {
                format!("{}", v)
            }
        }
    }
}

fn signed_ms(v: Option<i64>, rate: Option<f64>, sig: bool) -> String {
    match v {
        None => "-".to_string(),
        Some(v) => {
            let s_sec = if v >= 0 {
                format!("+{:.2}s", v as f64 / 1000.0)
            } else {
                format!("{:.2}s", v as f64 / 1000.0)
            };
            let r_str = match rate {
                Some(r) => format!(" ({:+.2}%)", r * 100.0),
                None => "".to_string(),
            };
            let combined = format!("{}{}", s_sec, r_str);
            if sig {
                red(&combined)
            } else {
                combined
            }
        }
    }
}

impl PipelineDiffResult {
    pub fn print_table(&self) {
        let s = &self.summary;

        println!();
        println!("{}", bold(&format!("{}  Pipeline 报告差异对比  {}", "=".repeat(22), "=".repeat(22))));
        println!();
        println!("  报告 A: {}", cyan(&self.report_a_path));
        println!("  报告 B: {}", cyan(&self.report_b_path));
        println!("  生成时间: {}", self.generated_at);
        println!();

        println!("{}", bold("【汇总对比】"));
        println!("{:-<80}", "");
        println!(
            "  {:<30} {:>18}  {:>18}  {:>18}",
            "指标", "A", "B", "差异"
        );
        println!("{:-<80}", "");

        let diff_time_s = s.total_time_diff_ms as f64 / 1000.0;
        let diff_time_sig = s.total_time_change_rate.abs() > SIGNIFICANT_THRESHOLD;
        let diff_time_str = format!(
            "{:+.2}s ({:+.2}%)",
            diff_time_s,
            s.total_time_change_rate * 100.0
        );
        let diff_time_str = if diff_time_sig {
            red(&diff_time_str)
        } else {
            diff_time_str
        };
        println!(
            "  {:<30} {:>18}  {:>18}  {:>18}",
            "总处理时间",
            format!("{:.2}s", s.total_time_ms_a as f64 / 1000.0),
            format!("{:.2}s", s.total_time_ms_b as f64 / 1000.0),
            diff_time_str
        );

        let cr_a_str = format!("{:.2}%", s.avg_compression_ratio_a * 100.0);
        let cr_b_str = format!("{:.2}%", s.avg_compression_ratio_b * 100.0);
        let cr_diff = format!("{:+.2}%", s.avg_compression_ratio_diff * 100.0);
        let cr_sig = s.avg_compression_ratio_diff.abs() > SIGNIFICANT_THRESHOLD;
        let cr_diff = if cr_sig { red(&cr_diff) } else { cr_diff };
        println!(
            "  {:<30} {:>18}  {:>18}  {:>18}",
            "平均压缩比", cr_a_str, cr_b_str, cr_diff
        );

        let succ_change_str = format!("{:+}", s.successful_diff);
        let succ_change_str = if s.successful_diff != 0 {
            if s.successful_diff > 0 {
                green(&succ_change_str)
            } else {
                red(&succ_change_str)
            }
        } else {
            succ_change_str
        };
        println!(
            "  {:<30} {:>18}  {:>18}  {:>18}",
            "成功文件数",
            s.successful_files_a,
            s.successful_files_b,
            succ_change_str
        );

        let fail_change_str = format!("{:+}", s.failed_diff);
        let fail_change_str = if s.failed_diff != 0 {
            if s.failed_diff > 0 {
                red(&fail_change_str)
            } else {
                green(&fail_change_str)
            }
        } else {
            fail_change_str
        };
        println!(
            "  {:<30} {:>18}  {:>18}  {:>18}",
            "失败文件数",
            s.failed_files_a,
            s.failed_files_b,
            fail_change_str
        );

        println!(
            "  {:<30} {:>18}  {:>18}  {:>18}",
            "总文件数",
            s.total_files_a,
            s.total_files_b,
            format!("{:+}", s.total_files_b as i64 - s.total_files_a as i64)
        );

        println!("{:-<80}", "");
        println!(
            "  共有文件: {}    仅在A: {}    仅在B: {}    {}: {}",
            s.files_in_common,
            s.files_only_in_a,
            s.files_only_in_b,
            red("显著变化"),
            s.significantly_changed_files
        );
        println!();

        println!("{}", bold("【逐文件对比】"));
        println!(
            "{:<35} {:>14}  {:>16}  {:>22}  {}",
            bold("文件"),
            bold("点数变化"),
            bold("面片差异"),
            bold("耗时差异"),
            bold("状态")
        );
        println!("{:-<95}", "");

        for f in &self.per_file {
            let sig = f.significant;
            let fname = if sig {
                red(&f.file_path)
            } else {
                f.file_path.clone()
            };
            let fname = format!("{:<35}", fname);
            let pts_sig = sig
                && f.points_change_rate
                    .map(|r| r.abs() > SIGNIFICANT_THRESHOLD)
                    .unwrap_or(false);
            let pts = format!("{:>14}", pct_str(f.points_change_rate, pts_sig));

            let faces_str = match (f.faces_diff, f.faces_change_rate) {
                (Some(d), Some(r)) => {
                    let s = format!("{} ({:+.2}%)", signed_i64(Some(d)), r * 100.0);
                    if sig && r.abs() > SIGNIFICANT_THRESHOLD {
                        format!("{:>16}", red(&s))
                    } else {
                        format!("{:>16}", s)
                    }
                }
                (Some(d), None) => format!("{:>16}", signed_i64(Some(d))),
                _ => format!("{:>16}", "-"),
            };

            let t_sig = sig
                && f.time_change_rate
                    .map(|r| r.abs() > SIGNIFICANT_THRESHOLD)
                    .unwrap_or(false);
            let time_str = format!("{:>22}", signed_ms(f.time_diff_ms, f.time_change_rate, t_sig));

            let status = if !f.a_exists {
                yellow("+新增(B)")
            } else if !f.b_exists {
                yellow("-缺失(B)")
            } else if f.success_changed {
                red("状态变化")
            } else if sig {
                red("显著")
            } else {
                green("正常")
            };

            println!(
                "{} {}  {}  {}  {}",
                fname, pts, faces_str, time_str, status
            );
        }
        println!("{:-<95}", "");
        println!();
        println!("  注: 变化率超过 ±5% 的项目用 {} 高亮", red("红色"));
        println!();
    }

    pub fn to_json_pretty(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::PointCloudError::JsonError(e))
    }
}
