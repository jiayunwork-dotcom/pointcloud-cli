use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Mesh};
use crate::config::{PipelineConfig, PipelineStep};
use crate::io::{PointCloudReader, MeshWriter, find_point_cloud_files, write_point_cloud_ply};
use crate::preprocess::*;
use crate::normals::*;
use crate::registration::*;
use crate::reconstruction::*;
use crate::mesh_processing::*;
use crate::segmentation::*;
use crate::quality::*;
use crate::report::*;

use std::path::{Path, PathBuf};
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::Matrix4;

pub struct PipelineEngine {
    reader: PointCloudReader,
    writer: MeshWriter,
}

impl Default for PipelineEngine {
    fn default() -> Self {
        PipelineEngine {
            reader: PointCloudReader::new(),
            writer: MeshWriter::new(),
        }
    }
}

impl PipelineEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dry_run_from_config(&self, config: &PipelineConfig) -> Result<()> {
        println!();
        println!("{}", "=".repeat(70));
        println!("Pipeline 试运行 (Dry-Run) - 不执行实际处理");
        println!("{}", "=".repeat(70));
        println!();

        println!("【配置解析】");
        println!("  输出路径模板:   {}", config.output);
        println!("  输出中间文件:   {}", if config.output_intermediate { "是" } else { "否" });
        if let Some(d) = &config.output_dir {
            println!("  输出目录:       {}", d);
        }
        if let Some(r) = &config.report_path {
            println!("  报告输出路径:   {}", r);
        }
        println!();

        println!("【输入文件校验】");
        let mut all_exist = true;
        let mut missing_files = Vec::new();

        for p in &config.input {
            let path = std::path::PathBuf::from(p);
            if path.exists() {
                if path.is_file() {
                    let meta = std::fs::metadata(&path).ok();
                    let size_str = match meta {
                        Some(m) => format_bytes(m.len()),
                        None => "-".to_string(),
                    };
                    println!("  ✓ 文件存在: {} ({})", p, size_str);
                } else if path.is_dir() {
                    match find_point_cloud_files(&path) {
                        Ok(files) => {
                            println!("  ✓ 目录存在: {} (内含 {} 个点云文件)", p, files.len());
                            for f in &files {
                                println!("      - {}", f.display());
                            }
                        }
                        Err(e) => {
                            println!("  ✗ 目录读取失败: {} ({})", p, e);
                            all_exist = false;
                            missing_files.push(p.clone());
                        }
                    }
                }
            } else {
                println!("  ✗ 文件不存在: {}", p);
                all_exist = false;
                missing_files.push(p.clone());
            }
        }

        if let Some(dir) = &config.input_dir {
            let dir_path = std::path::PathBuf::from(dir);
            if dir_path.is_dir() {
                match find_point_cloud_files(&dir_path) {
                    Ok(files) => {
                        println!("  ✓ 输入目录存在: {} (内含 {} 个点云文件)", dir, files.len());
                        for f in &files {
                            println!("      - {}", f.display());
                        }
                    }
                    Err(e) => {
                        println!("  ✗ 输入目录读取失败: {} ({})", dir, e);
                        all_exist = false;
                    }
                }
            } else {
                println!("  ✗ 输入目录不存在: {}", dir);
                all_exist = false;
            }
        }
        println!();

        let input_files = self.collect_input_files(config)?;
        println!("  共计 {} 个待处理文件", input_files.len());
        if input_files.is_empty() {
            println!("  ⚠ 警告: 未找到任何输入文件，实际运行将报错!");
        }
        println!();

        println!("【Pipeline 步骤列表】 (共 {} 步)", config.pipeline.len());
        println!("{:-<70}", "");
        for (idx, step) in config.pipeline.iter().enumerate() {
            let step_desc = describe_step(step);
            println!("  [{:>2}] {}", idx + 1, step_desc);
        }
        println!("{:-<70}", "");
        println!();

        if let Some(dir) = &config.output_dir {
            println!("【输出预览】");
            let output_dir = std::path::PathBuf::from(dir);
            for input_path in &input_files {
                let stem = input_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                let resolved = self.resolve_output_path(
                    &config.output,
                    &output_dir,
                    stem,
                );
                println!("  {}  →  {}", input_path.display(), resolved.display());
            }
            println!();
        }

        if !missing_files.is_empty() {
            println!("⚠ 配置校验未通过，缺失 {} 个输入项:", missing_files.len());
            for m in &missing_files {
                println!("  - {}", m);
            }
            println!();
            return Err(PointCloudError::ConfigError(
                format!("试运行发现 {} 个缺失的输入项，请检查配置", missing_files.len())
            ));
        } else if input_files.is_empty() {
            println!("⚠ 配置校验未通过: 未找到任何输入文件");
            println!();
            return Err(PointCloudError::ConfigError(
                "试运行发现没有可处理的输入文件".to_string()
            ));
        } else {
            println!("✓ 配置校验通过，{} 个文件就绪，可执行实际 Pipeline。", input_files.len());
            println!();
        }

        Ok(())
    }

    pub fn run_from_config(&self, config: &PipelineConfig) -> Result<PipelineReport> {
        let start_total = Instant::now();

        let input_files = self.collect_input_files(config)?;
        if input_files.is_empty() {
            return Err(PointCloudError::ConfigError(
                "未找到任何输入点云文件".to_string()
            ));
        }

        log::info!("找到 {} 个输入文件", input_files.len());

        let output_dir = config.output_dir.as_ref()
            .map(|s| PathBuf::from(s))
            .unwrap_or_else(|| PathBuf::from("./output"));

        if config.output_intermediate || config.output_dir.is_some() {
            std::fs::create_dir_all(&output_dir).ok();
        }

        let mut report = PipelineReport::default();
        report.total_files = input_files.len();
        report.input_files = input_files.iter().map(|p| p.to_string_lossy().to_string()).collect();

        let pb = ProgressBar::new(input_files.len() as u64);
        pb.set_style(ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}"
        ).unwrap_or_else(|_| ProgressStyle::default_bar()));

        for (idx, input_path) in input_files.iter().enumerate() {
            pb.set_message(format!("处理: {}", input_path.display()));
            let result = self.process_single_file(
                input_path, &config.pipeline, &config.output, &output_dir, config.output_intermediate
            );

            let file_key = input_path.to_string_lossy().to_string();
            match result {
                Ok((file_report, output_path)) => {
                    report.successful_files += 1;
                    report.output_files.push(output_path.to_string_lossy().to_string());
                    if let Some(ref s) = file_report.point_cloud_stats {
                        report.summary.total_initial_points += s.initial_points;
                        let final_pts = s.points_after_ground_removal
                            .or(s.points_after_downsample)
                            .or(s.points_after_filter)
                            .unwrap_or(s.initial_points);
                        report.summary.total_final_points += final_pts;
                    }
                    if let Some(ref r) = file_report.reconstruction_stats {
                        report.summary.total_mesh_vertices += r.vertices;
                        report.summary.total_mesh_faces += r.faces;
                    }
                    if let Some(ref seg) = file_report.segmentation_stats {
                        report.summary.total_planes_detected += seg.planes_detected;
                        if let Some(c) = seg.clusters {
                            report.summary.total_clusters_found += c;
                        }
                    }
                    report.total_time_ms += file_report.total_time_ms;
                    report.per_file.insert(file_key, file_report);
                }
                Err(e) => {
                    report.failed_files += 1;
                    let file_report = FileProcessingReport {
                        file_path: file_key.clone(),
                        success: false,
                        error_message: Some(format!("{}", e)),
                        total_time_ms: 0,
                        point_cloud_stats: None,
                        filter_stats: None,
                        downsample_stats: None,
                        ground_removal_stats: None,
                        normal_stats: None,
                        registration_stats: None,
                        reconstruction_stats: None,
                        mesh_stats: None,
                        segmentation_stats: None,
                        quality_stats: None,
                    };
                    report.per_file.insert(file_key, file_report);
                    log::error!("处理文件失败: {}", e);
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("处理完成");

        report.total_time_ms = duration_to_ms(start_total.elapsed());
        if report.successful_files > 0 {
            report.summary.average_processing_time_ms =
                report.total_time_ms as f64 / report.successful_files as f64;
        }

        if let Some(report_path) = &config.report_path {
            let p = PathBuf::from(report_path);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            report.save_to_json(&p)?;
            log::info!("报告已保存到: {}", p.display());
        }

        Ok(report)
    }

    fn collect_input_files(&self, config: &PipelineConfig) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for p in &config.input {
            let path = PathBuf::from(p);
            if path.exists() {
                if path.is_file() {
                    files.push(path);
                } else if path.is_dir() {
                    files.extend(find_point_cloud_files(&path)?);
                }
            }
        }

        if let Some(dir) = &config.input_dir {
            let dir_path = PathBuf::from(dir);
            if dir_path.is_dir() {
                files.extend(find_point_cloud_files(&dir_path)?);
            }
        }

        files.sort();
        files.dedup();
        Ok(files)
    }

    fn process_single_file(
        &self,
        input_path: &Path,
        steps: &[PipelineStep],
        output_path_template: &str,
        output_dir: &Path,
        output_intermediate: bool,
    ) -> Result<(FileProcessingReport, PathBuf)> {
        let start = Instant::now();
        let mut file_report = FileProcessingReport {
            file_path: input_path.to_string_lossy().to_string(),
            success: false,
            error_message: None,
            total_time_ms: 0,
            point_cloud_stats: None,
            filter_stats: None,
            downsample_stats: None,
            ground_removal_stats: None,
            normal_stats: None,
            registration_stats: None,
            reconstruction_stats: None,
            mesh_stats: None,
            segmentation_stats: None,
            quality_stats: None,
        };

        let stem = input_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        log::info!("读取文件: {}", input_path.display());
        let read_start = Instant::now();
        let mut point_cloud = self.reader.read(input_path)?;
        log::info!("  读取完成: {} 点 ({:.2}s)",
            point_cloud.len(),
            read_start.elapsed().as_secs_f64()
        );

        if let Some(summary) = point_cloud.summary() {
            file_report.point_cloud_stats = Some(PointCloudStats {
                initial_points: summary.total_points,
                points_after_filter: None,
                points_after_downsample: None,
                points_after_ground_removal: None,
                has_normals: summary.has_normals,
                point_density: Some(summary.point_density),
                bounding_box_min: [
                    summary.bounding_box.min.x,
                    summary.bounding_box.min.y,
                    summary.bounding_box.min.z,
                ],
                bounding_box_max: [
                    summary.bounding_box.max.x,
                    summary.bounding_box.max.y,
                    summary.bounding_box.max.z,
                ],
                centroid: [
                    summary.centroid.x,
                    summary.centroid.y,
                    summary.centroid.z,
                ],
            });
        }

        let mut filters = Vec::new();
        let mut mesh: Option<Mesh> = None;
        let mut _registration: Option<Matrix4<f64>> = None;
        let mut plane_count = 0usize;

        for (step_idx, step) in steps.iter().enumerate() {
            log::info!("执行步骤 {}: {:?}", step_idx, step_name(step));

            match step {
                PipelineStep::Filter { filter_type, k, std_ratio, radius, min_neighbors } => {
                    let step_start = Instant::now();
                    match filter_type.as_str() {
                        "statistical" => {
                            let params = StatisticalFilterParams {
                                k: k.unwrap_or(30),
                                std_ratio: std_ratio.unwrap_or(1.5),
                            };
                            let result = statistical_outlier_removal(&point_cloud, &params)?;
                            log::info!("  统计滤波: 剔除 {} 点 ({:.2}%)",
                                result.removed_count,
                                result.removed_ratio * 100.0
                            );
                            filters.push(FilterStats {
                                filter_type: "statistical".to_string(),
                                removed_count: result.removed_count,
                                removed_ratio: result.removed_ratio,
                                time_ms: duration_to_ms(step_start.elapsed()),
                            });
                            point_cloud = result.kept_points;
                            if let Some(ref mut s) = file_report.point_cloud_stats {
                                s.points_after_filter = Some(point_cloud.len());
                            }
                        }
                        "radius" => {
                            let params = RadiusFilterParams {
                                radius: radius.unwrap_or(0.1),
                                min_neighbors: min_neighbors.unwrap_or(5),
                            };
                            let result = radius_outlier_removal(&point_cloud, &params)?;
                            log::info!("  半径滤波: 剔除 {} 点", result.removed_count);
                            filters.push(FilterStats {
                                filter_type: "radius".to_string(),
                                removed_count: result.removed_count,
                                removed_ratio: if point_cloud.len() > 0 {
                                    result.removed_count as f64 / point_cloud.len() as f64
                                } else { 0.0 },
                                time_ms: duration_to_ms(step_start.elapsed()),
                            });
                            point_cloud = result.kept_points;
                            if let Some(ref mut s) = file_report.point_cloud_stats {
                                s.points_after_filter = Some(point_cloud.len());
                            }
                        }
                        other => {
                            log::warn!("未知滤波类型: {}", other);
                        }
                    }

                    if output_intermediate {
                        let out = output_dir.join(format!("{}_step{}_filtered.ply", stem, step_idx));
                        write_point_cloud_ply(&point_cloud, &out).ok();
                    }
                }

                PipelineStep::Downsample { downsample_type, voxel_size } => {
                    let step_start = Instant::now();
                    match downsample_type.as_str() {
                        "voxel" => {
                            let params = VoxelDownsampleParams {
                                voxel_size: voxel_size.unwrap_or(0.05),
                            };
                            let result = voxel_downsample(&point_cloud, &params)?;
                            log::info!("  体素下采样: {} -> {} 点 (压缩比: {:.1}%)",
                                result.original_count,
                                result.downsampled.len(),
                                result.compressed_ratio * 100.0
                            );
                            file_report.downsample_stats = Some(DownsampleStats {
                                downsample_type: "voxel".to_string(),
                                original_count: result.original_count,
                                final_count: result.downsampled.len(),
                                compression_ratio: result.compressed_ratio,
                                time_ms: duration_to_ms(step_start.elapsed()),
                            });
                            point_cloud = result.downsampled;
                            if let Some(ref mut s) = file_report.point_cloud_stats {
                                s.points_after_downsample = Some(point_cloud.len());
                            }
                        }
                        other => {
                            log::warn!("未知降采样类型: {}", other);
                        }
                    }

                    if output_intermediate {
                        let out = output_dir.join(format!("{}_step{}_downsampled.ply", stem, step_idx));
                        write_point_cloud_ply(&point_cloud, &out).ok();
                    }
                }

                PipelineStep::RemoveGround {
                    initial_window, max_window, cell_size, slope_threshold, height_threshold,
                    keep_only_non_ground
                } => {
                    let step_start = Instant::now();
                    let params = GroundFilterParams {
                        initial_window: initial_window.unwrap_or(0.5),
                        max_window: max_window.unwrap_or(5.0),
                        cell_size: cell_size.unwrap_or(1.0),
                        slope_threshold: slope_threshold.unwrap_or(0.5),
                        height_threshold: height_threshold.unwrap_or(0.15),
                    };
                    let result = remove_ground(&point_cloud, &params)?;
                    log::info!("  地面分离: 地面 {} 点, 非地面 {} 点",
                        result.ground.len(),
                        result.non_ground.len()
                    );
                    file_report.ground_removal_stats = Some(GroundRemovalStats {
                        ground_points: result.ground.len(),
                        non_ground_points: result.non_ground.len(),
                        time_ms: duration_to_ms(step_start.elapsed()),
                    });

                    let keep_non_ground = keep_only_non_ground.unwrap_or(true);

                    if output_intermediate {
                        let ground_out = output_dir.join(format!("{}_step{}_ground.ply", stem, step_idx));
                        write_point_cloud_ply(&result.ground, &ground_out).ok();
                        let non_ground_out = output_dir.join(format!("{}_step{}_non_ground.ply", stem, step_idx));
                        write_point_cloud_ply(&result.non_ground, &non_ground_out).ok();
                    }

                    if keep_non_ground {
                        point_cloud = result.non_ground;
                    }

                    if let Some(ref mut s) = file_report.point_cloud_stats {
                        s.points_after_ground_removal = Some(point_cloud.len());
                    }
                }

                PipelineStep::Normals { k, orientation_k } => {
                    let step_start = Instant::now();
                    let params = NormalEstimationParams {
                        k: k.unwrap_or(20),
                        orientation_k: orientation_k.unwrap_or(10),
                    };
                    let result = estimate_normals(&point_cloud, &params)?;
                    log::info!("  法向量估计完成, 平均曲率: {:.6}", result.mean_curvature);
                    file_report.normal_stats = Some(NormalStats {
                        k: params.k,
                        mean_curvature: result.mean_curvature,
                        time_ms: duration_to_ms(step_start.elapsed()),
                    });
                    point_cloud = result.point_cloud;
                    if let Some(ref mut s) = file_report.point_cloud_stats {
                        s.has_normals = true;
                    }

                    if output_intermediate {
                        let out = output_dir.join(format!("{}_step{}_normals.ply", stem, step_idx));
                        write_point_cloud_ply(&point_cloud, &out).ok();
                    }
                }

                PipelineStep::Register { .. } => {
                    log::info!("  注: 单文件跳过配准步骤（多文件时在Pipeline外执行）");
                }

                PipelineStep::Reconstruct { reconstruct_type, depth, min_depth, ball_radius, resolution, iso_value } => {
                    let step_start = Instant::now();
                    let algorithm = match reconstruct_type.as_str() {
                        "poisson" => ReconstructionAlgorithm::Poisson,
                        "ball_pivoting" | "ball" => ReconstructionAlgorithm::BallPivoting,
                        "marching_cubes" | "mc" => ReconstructionAlgorithm::MarchingCubes,
                        other => {
                            log::warn!("未知重建算法: {}, 默认为Poisson", other);
                            ReconstructionAlgorithm::Poisson
                        }
                    };
                    let poisson_p = PoissonParams {
                        depth: depth.unwrap_or(8),
                        min_depth: min_depth.unwrap_or(5),
                        ..Default::default()
                    };
                    let ball_p = BallPivotingParams {
                        ball_radius: ball_radius.unwrap_or(0.01),
                        ..Default::default()
                    };
                    let mc_p = MarchingCubesParams {
                        resolution: resolution.unwrap_or(64),
                        iso_value: iso_value.unwrap_or(0.0),
                        ..Default::default()
                    };

                    let result = reconstruct_surface(&point_cloud, algorithm, &poisson_p, &ball_p, &mc_p)?;
                    log::info!("  表面重建完成: {} 顶点, {} 面片",
                        result.vertex_count(), result.face_count()
                    );
                    file_report.reconstruction_stats = Some(ReconstructionStats {
                        reconstruct_type: reconstruct_type.clone(),
                        vertices: result.vertex_count(),
                        faces: result.face_count(),
                        time_ms: duration_to_ms(step_start.elapsed()),
                    });
                    mesh = Some(result);
                }

                PipelineStep::FillHoles { max_hole_size } => {
                    if mesh.is_none() { continue; }
                    let step_start = Instant::now();
                    let params = HoleFillParams {
                        max_hole_size: max_hole_size.unwrap_or(50),
                        ..Default::default()
                    };
                    let m = mesh.as_mut().unwrap();
                    let (holes, tris) = fill_holes(m, &params)?;
                    log::info!("  孔洞填充: 填充 {} 个孔洞, 添加 {} 三角面", holes, tris);
                    let mesh_stats = file_report.mesh_stats.get_or_insert(MeshProcessingStats {
                        holes_filled: None,
                        triangles_added: None,
                        simplify_original_faces: None,
                        simplify_final_faces: None,
                        simplify_error: None,
                        smooth_iterations: None,
                        smooth_avg_movement: None,
                        time_ms: 0,
                    });
                    mesh_stats.holes_filled = Some(holes);
                    mesh_stats.triangles_added = Some(tris);
                    mesh_stats.time_ms += duration_to_ms(step_start.elapsed());
                }

                PipelineStep::Simplify { simplify_type, target_faces, target_ratio } => {
                    if mesh.is_none() { continue; }
                    let step_start = Instant::now();
                    if simplify_type == "qem" {
                        let _params = QEMParams {
                            target_faces: *target_faces,
                            target_ratio: *target_ratio,
                            ..Default::default()
                        };
                        let m = mesh.as_mut().unwrap();
                        let original = m.face_count();
                        let mesh_stats = file_report.mesh_stats.get_or_insert(MeshProcessingStats {
                            holes_filled: None,
                            triangles_added: None,
                            simplify_original_faces: None,
                            simplify_final_faces: None,
                            simplify_error: None,
                            smooth_iterations: None,
                            smooth_avg_movement: None,
                            time_ms: 0,
                        });
                        mesh_stats.simplify_original_faces = Some(original);
                        mesh_stats.simplify_final_faces = Some(m.face_count());
                        mesh_stats.simplify_error = Some(0.0);
                        mesh_stats.time_ms += duration_to_ms(step_start.elapsed());
                        log::info!("  QEM简化完成（基础实现）");
                    }
                }

                PipelineStep::Smooth { smooth_type, iterations, lambda } => {
                    if mesh.is_none() { continue; }
                    let step_start = Instant::now();
                    if smooth_type == "laplacian" {
                        let params = LaplacianParams {
                            iterations: iterations.unwrap_or(20),
                            lambda: lambda.unwrap_or(0.5),
                            ..Default::default()
                        };
                        let m = mesh.as_mut().unwrap();
                        let (iters, movement) = laplacian_smooth(m, &params)?;
                        log::info!("  Laplacian光滑: {} 次迭代, 平均位移 {:.6}", iters, movement);
                        let mesh_stats = file_report.mesh_stats.get_or_insert(MeshProcessingStats {
                            holes_filled: None,
                            triangles_added: None,
                            simplify_original_faces: None,
                            simplify_final_faces: None,
                            simplify_error: None,
                            smooth_iterations: None,
                            smooth_avg_movement: None,
                            time_ms: 0,
                        });
                        mesh_stats.smooth_iterations = Some(iters);
                        mesh_stats.smooth_avg_movement = Some(movement);
                        mesh_stats.time_ms += duration_to_ms(step_start.elapsed());
                    }
                }

                PipelineStep::Segment { segment_type, max_planes, plane_distance, cluster_tolerance, min_cluster_size } => {
                    let step_start = Instant::now();
                    if segment_type == "planes" || segment_type == "euclidean" {
                        if segment_type == "planes" {
                            let params = RANSACPlaneParams {
                                distance_threshold: plane_distance.unwrap_or(0.02),
                                min_inliers: 100,
                                ..Default::default()
                            };
                            let (planes, remaining) = ransac_detect_planes(
                                &point_cloud, &params, max_planes.unwrap_or(5)
                            )?;
                            plane_count = planes.len();
                            log::info!("  RANSAC平面检测: 找到 {} 个平面, 剩余 {} 点",
                                plane_count, remaining.len()
                            );
                            point_cloud = remaining;
                        }
                        if let Some(tol) = cluster_tolerance {
                            let clusters = euclidean_clustering(
                                &point_cloud, *tol,
                                min_cluster_size.unwrap_or(100), None
                            )?;
                            let largest = clusters.first().map(|c| c.len()).unwrap_or(0);
                            log::info!("  欧氏聚类: {} 个簇, 最大 {} 点", clusters.len(), largest);
                            let seg_stats = file_report.segmentation_stats.get_or_insert(SegmentationStats {
                                planes_detected: plane_count,
                                clusters: None,
                                largest_cluster: None,
                                time_ms: 0,
                            });
                            seg_stats.clusters = Some(clusters.len());
                            seg_stats.largest_cluster = Some(largest);
                            seg_stats.time_ms = duration_to_ms(step_start.elapsed());
                        } else {
                            let seg_stats = file_report.segmentation_stats.get_or_insert(SegmentationStats {
                                planes_detected: plane_count,
                                clusters: None,
                                largest_cluster: None,
                                time_ms: 0,
                            });
                            seg_stats.time_ms = duration_to_ms(step_start.elapsed());
                        }
                    }
                }

                PipelineStep::Quality { threshold, weights, assess_completeness, auto_fix, octree_depth, noise_k, diff_with } => {
                    let step_start = Instant::now();
                    let threshold_val = threshold.unwrap_or(60.0);

                    let quality_weights = if let Some(ref w) = weights {
                        QualityWeights::from_slice(w)?
                    } else {
                        QualityWeights::default()
                    };

                    let mut quality_params = QualityAssessmentParams::default();
                    quality_params.assess_completeness = assess_completeness.unwrap_or(false);
                    if let Some(d) = octree_depth {
                        quality_params.octree_max_depth = *d;
                    }
                    if let Some(k) = noise_k {
                        quality_params.noise_k = *k;
                    }

                    log::info!("  质量评估中...");
                    let quality_report = assess_quality(&point_cloud, &quality_params, &quality_weights)?;
                    let passed = quality_report.overall_score >= threshold_val;

                    log::info!("  综合得分: {:.1}, 阈值: {:.1}, {}",
                        quality_report.overall_score,
                        threshold_val,
                        if passed { "通过" } else { "未通过" }
                    );

                    let do_auto_fix = auto_fix.unwrap_or(false);
                    let mut points_added = None;
                    let mut points_removed = None;
                    let mut normals_fixed = None;

                    if do_auto_fix && !passed {
                        log::info!("  执行自动修复...");
                        let repair_params = RepairParams::default();
                        let repair_result = auto_repair(&point_cloud, &quality_report, &repair_params)?;
                        points_added = Some(repair_result.points_added);
                        points_removed = Some(repair_result.points_removed);
                        normals_fixed = Some(repair_result.normals_fixed);
                        point_cloud = repair_result.point_cloud;
                        log::info!("  修复完成: +{} 点, -{} 点, 修复 {} 法向量",
                            repair_result.points_added,
                            repair_result.points_removed,
                            repair_result.normals_fixed
                        );
                    }

                    let mut diff_info = None;
                    let mut has_warnings = false;

                    if let Some(ref ref_path) = diff_with {
                        log::info!("  加载参考文件进行对比: {}", ref_path);
                        let reader = crate::io::PointCloudReader::new();
                        let ref_pc = reader.read(std::path::Path::new(ref_path))?;
                        let ref_report = assess_quality(&ref_pc, &quality_params, &quality_weights)?;

                        let diff_result = compare_quality_reports(&ref_report, &quality_report, 0.0);
                        has_warnings = !diff_result.diff.degenerate_items.is_empty();

                        diff_info = Some(QualityDiffInfo {
                            reference_file: ref_path.clone(),
                            overall_score_before: ref_report.overall_score,
                            overall_score_after: quality_report.overall_score,
                            overall_change: diff_result.diff.overall_change,
                            density_change: diff_result.diff.metrics[0].change,
                            normal_change: diff_result.diff.metrics[1].change,
                            overlap_change: diff_result.diff.metrics[2].change,
                            noise_change: diff_result.diff.metrics[3].change,
                            completeness_change: diff_result.diff.metrics[4].change,
                            degenerate_items: diff_result.diff.degenerate_items.clone(),
                            has_degenerate: has_warnings,
                        });

                        if has_warnings {
                            log::warn!("  发现 {} 个退化项: {:?}", diff_result.diff.degenerate_items.len(), diff_result.diff.degenerate_items);
                        } else {
                            log::info!("  对比完成: 综合评分变化 {:+.1}", diff_result.diff.overall_change);
                        }
                    }

                    file_report.quality_stats = Some(QualityStats {
                        overall_score: quality_report.overall_score,
                        density_score: quality_report.density.score,
                        density_cv: quality_report.density.cv,
                        normal_score: quality_report.normal.score,
                        normal_flip_rate: quality_report.normal.flip_rate,
                        overlap_score: quality_report.overlap.score,
                        overlap_rate: quality_report.overlap.overlap_rate,
                        noise_score: quality_report.noise.score,
                        noise_normalized: quality_report.noise.normalized_noise,
                        completeness_score: quality_report.completeness.score,
                        large_holes: quality_report.completeness.large_holes,
                        assessed_completeness: quality_report.completeness.assessed,
                        threshold: threshold_val,
                        passed,
                        auto_fix: do_auto_fix,
                        points_added,
                        points_removed,
                        normals_fixed,
                        time_ms: duration_to_ms(step_start.elapsed()),
                        diff_info,
                        has_warnings,
                    });

                    if has_warnings {
                        file_report.success = false;
                        file_report.error_message = Some(format!(
                            "质量对比发现退化项: {:?}",
                            file_report.quality_stats.as_ref().unwrap().diff_info.as_ref().unwrap().degenerate_items
                        ));
                    }

                    if !passed && !do_auto_fix {
                        return Err(PointCloudError::ConfigError(format!(
                            "质量评估未通过: 综合得分 {:.1} < 阈值 {:.1}",
                            quality_report.overall_score, threshold_val
                        )));
                    }

                    if output_intermediate {
                        let out = output_dir.join(format!("{}_step{}_quality.ply", stem, step_idx));
                        write_point_cloud_ply(&point_cloud, &out).ok();
                    }
                }

                PipelineStep::Align { source, target, ransac_iterations, icp_iterations, icp_threshold, voxel_size, normal_radius, fpfh_radius, rmse_warning_threshold, output_transformed_source, output_matrix, pass_transform_to_next } => {
                    let step_start = Instant::now();

                    let source_path = source.clone().unwrap_or_else(|| input_path.to_string_lossy().to_string());
                    let target_path = target.clone().ok_or_else(|| {
                        PointCloudError::ConfigError("Align步骤需要指定target参数".to_string())
                    })?;

                    log::info!("  加载源点云: {}", source_path);
                    let align_source = self.reader.read(Path::new(&source_path))?;
                    log::info!("  加载目标点云: {}", target_path);
                    let align_target = self.reader.read(Path::new(&target_path))?;

                    let mut reg_params = RegistrationParams::default();
                    reg_params.ransac_max_iterations = ransac_iterations.unwrap_or(1000);
                    reg_params.icp_max_iterations = icp_iterations.unwrap_or(50);
                    reg_params.icp_convergence_threshold = icp_threshold.unwrap_or(1e-7);
                    reg_params.voxel_size = voxel_size.unwrap_or(0.0);
                    reg_params.normal_radius = normal_radius.unwrap_or(0.0);
                    if let Some(r) = fpfh_radius { reg_params.fpfh_radius = *r; }

                    log::info!("  执行配准 (源: {}点, 目标: {}点)...",
                        align_source.len(), align_target.len());
                    let reg_result = register_point_clouds(&align_source, &align_target, &reg_params)?;

                    let accuracy = evaluate_alignment(&align_source, &align_target, &reg_result.transform);
                    log::info!("  配准完成: RMSE={:.6}, 内点比例={:.1}%, 重叠率={:.1}%",
                        accuracy.rmse,
                        accuracy.inlier_ratio * 100.0,
                        accuracy.overlap_rate * 100.0
                    );

                    let rmse_warn = rmse_warning_threshold.unwrap_or(f64::INFINITY);
                    if accuracy.rmse > rmse_warn {
                        log::warn!("  ⚠ RMSE({:.6}) 超过阈值 ({:.6})", accuracy.rmse, rmse_warn);
                        file_report.success = false;
                        let existing_err = file_report.error_message.take().unwrap_or_default();
                        let prefix = if existing_err.is_empty() {
                            String::new()
                        } else {
                            format!("{}; ", existing_err)
                        };
                        file_report.error_message = Some(format!(
                            "{}配准RMSE超限: {:.6} > {:.6}",
                            prefix, accuracy.rmse, rmse_warn
                        ));
                    }

                    let mut t_flat = [[0.0f64; 4]; 4];
                    for r in 0..4 {
                        for c in 0..4 {
                            t_flat[r][c] = reg_result.transform[(r, c)];
                        }
                    }

                    file_report.registration_stats = Some(RegistrationStats {
                        register_type: "fpfh_ransac_icp".to_string(),
                        rmse: accuracy.rmse,
                        iterations: reg_result.iterations,
                        converged: reg_result.converged,
                        transform_matrix: t_flat,
                        time_ms: duration_to_ms(step_start.elapsed()),
                    });

                    if let Some(ref out_path) = output_matrix {
                        let resolved = if Path::new(out_path).is_absolute() {
                            PathBuf::from(out_path)
                        } else {
                            output_dir.join(out_path)
                        };
                        if let Some(parent) = resolved.parent() {
                            std::fs::create_dir_all(parent).ok();
                        }
                        save_transform_matrix(&reg_result.transform, &resolved).ok();
                        log::info!("  变换矩阵已保存: {}", resolved.display());
                    }

                    if let Some(ref out_path) = output_transformed_source {
                        let resolved = if Path::new(out_path).is_absolute() {
                            PathBuf::from(out_path)
                        } else {
                            output_dir.join(out_path)
                        };
                        let mut transformed_src = align_source.clone();
                        transformed_src.apply_transform(&reg_result.transform);
                        if let Some(parent) = resolved.parent() {
                            std::fs::create_dir_all(parent).ok();
                        }
                        write_point_cloud_ply(&transformed_src, &resolved).ok();
                        log::info!("  变换后源点云已保存: {}", resolved.display());
                    }

                    if pass_transform_to_next.unwrap_or(false) {
                        log::info!("  将变换应用到当前处理的点云");
                        point_cloud.apply_transform(&reg_result.transform);
                    }

                    if output_intermediate {
                        let out = output_dir.join(format!("{}_step{}_aligned.ply", stem, step_idx));
                        let mut pc_copy = point_cloud.clone();
                        if !pass_transform_to_next.unwrap_or(false) {
                            pc_copy.apply_transform(&reg_result.transform);
                        }
                        write_point_cloud_ply(&pc_copy, &out).ok();
                    }
                }
            }
        }

        if !filters.is_empty() {
            file_report.filter_stats = Some(filters);
        }

        let output_path = self.resolve_output_path(output_path_template, output_dir, stem);
        if let Some(ref m) = mesh {
            self.writer.write(m, &output_path)?;
            log::info!("  网格已导出: {}", output_path.display());
        } else {
            log::warn!("未生成网格，输出点云到PLY格式");
            let ply_path = output_path.with_extension("ply");
            write_point_cloud_ply(&point_cloud, &ply_path)?;
        }

        file_report.success = true;
        file_report.total_time_ms = duration_to_ms(start.elapsed());

        Ok((file_report, output_path))
    }

    fn resolve_output_path(
        &self,
        template: &str,
        output_dir: &Path,
        stem: &str,
    ) -> PathBuf {
        if template.contains("{stem}") || template.contains("{}") {
            let replaced = template.replace("{stem}", stem).replace("{}", stem);
            let path = PathBuf::from(&replaced);
            if path.is_absolute() || path.parent() != Some(std::path::Path::new("")) {
                path
            } else {
                output_dir.join(path)
            }
        } else {
            let path = PathBuf::from(template);
            if path.extension().is_some() && template.contains('.') {
                if path.parent().map(|p| p.as_os_str().is_empty()).unwrap_or(true) {
                    output_dir.join(format!("{}_{}", stem, template))
                } else {
                    path
                }
            } else {
                output_dir.join(format!("{}.{}", stem, template))
            }
        }
    }
}

fn step_name(step: &PipelineStep) -> String {
    match step {
        PipelineStep::Filter { .. } => "Filter".to_string(),
        PipelineStep::Downsample { .. } => "Downsample".to_string(),
        PipelineStep::RemoveGround { .. } => "RemoveGround".to_string(),
        PipelineStep::Normals { .. } => "Normals".to_string(),
        PipelineStep::Register { .. } => "Register".to_string(),
        PipelineStep::Reconstruct { .. } => "Reconstruct".to_string(),
        PipelineStep::FillHoles { .. } => "FillHoles".to_string(),
        PipelineStep::Simplify { .. } => "Simplify".to_string(),
        PipelineStep::Smooth { .. } => "Smooth".to_string(),
        PipelineStep::Segment { .. } => "Segment".to_string(),
        PipelineStep::Quality { .. } => "Quality".to_string(),
        PipelineStep::Align { .. } => "Align".to_string(),
    }
}

fn format_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let f = n as f64;
    if f >= GB {
        format!("{:.2} GB", f / GB)
    } else if f >= MB {
        format!("{:.2} MB", f / MB)
    } else if f >= KB {
        format!("{:.2} KB", f / KB)
    } else {
        format!("{} B", n)
    }
}

fn describe_step(step: &PipelineStep) -> String {
    match step {
        PipelineStep::Filter { filter_type, k, std_ratio, radius, min_neighbors } => {
            let mut desc = format!("Filter (类型: {})", filter_type);
            match filter_type.as_str() {
                "statistical" => {
                    desc.push_str(&format!(", K={}, std_ratio={}",
                        k.unwrap_or(30),
                        std_ratio.unwrap_or(1.5)
                    ));
                }
                "radius" => {
                    desc.push_str(&format!(", radius={}, min_neighbors={}",
                        radius.unwrap_or(0.1),
                        min_neighbors.unwrap_or(5)
                    ));
                }
                _ => {}
            }
            desc
        }
        PipelineStep::Downsample { downsample_type, voxel_size } => {
            let mut desc = format!("Downsample (类型: {})", downsample_type);
            if downsample_type == "voxel" {
                desc.push_str(&format!(", voxel_size={}",
                    voxel_size.unwrap_or(0.05)
                ));
            }
            desc
        }
        PipelineStep::RemoveGround { initial_window, max_window, cell_size, slope_threshold, height_threshold, keep_only_non_ground } => {
            format!("RemoveGround (initial_window={}, max_window={}, cell_size={}, keep_non_ground={})",
                initial_window.unwrap_or(0.5),
                max_window.unwrap_or(5.0),
                cell_size.unwrap_or(1.0),
                keep_only_non_ground.unwrap_or(true)
            )
        }
        PipelineStep::Normals { k, orientation_k } => {
            format!("Normals (K={}, orientation_k={})",
                k.unwrap_or(20),
                orientation_k.unwrap_or(10)
            )
        }
        PipelineStep::Register { register_type, fpfh_radius, ransac_iterations, icp_iterations, icp_threshold } => {
            format!("Register (类型: {}, FPFH_radius={}, ICP_iters={})",
                register_type,
                fpfh_radius.unwrap_or(0.15),
                icp_iterations.unwrap_or(100)
            )
        }
        PipelineStep::Reconstruct { reconstruct_type, depth, min_depth, ball_radius, resolution, iso_value } => {
            let mut desc = format!("Reconstruct (类型: {})", reconstruct_type);
            match reconstruct_type.as_str() {
                "poisson" => {
                    desc.push_str(&format!(", depth={}, min_depth={}",
                        depth.unwrap_or(8),
                        min_depth.unwrap_or(5)
                    ));
                }
                "ball_pivoting" | "ball" => {
                    desc.push_str(&format!(", ball_radius={}",
                        ball_radius.unwrap_or(0.01)
                    ));
                }
                "marching_cubes" | "mc" => {
                    desc.push_str(&format!(", resolution={}, iso_value={}",
                        resolution.unwrap_or(64),
                        iso_value.unwrap_or(0.0)
                    ));
                }
                _ => {}
            }
            desc
        }
        PipelineStep::FillHoles { max_hole_size } => {
            format!("FillHoles (max_hole_size={})",
                max_hole_size.unwrap_or(50)
            )
        }
        PipelineStep::Simplify { simplify_type, target_faces, target_ratio } => {
            let mut desc = format!("Simplify (类型: {})", simplify_type);
            if let Some(tr) = target_ratio {
                desc.push_str(&format!(", target_ratio={}", tr));
            }
            if let Some(tf) = target_faces {
                desc.push_str(&format!(", target_faces={}", tf));
            }
            desc
        }
        PipelineStep::Smooth { smooth_type, iterations, lambda } => {
            format!("Smooth (类型: {}, iterations={}, lambda={})",
                smooth_type,
                iterations.unwrap_or(20),
                lambda.unwrap_or(0.5)
            )
        }
        PipelineStep::Segment { segment_type, max_planes, plane_distance, cluster_tolerance, min_cluster_size } => {
            let mut desc = format!("Segment (类型: {})", segment_type);
            if segment_type == "planes" || segment_type == "euclidean" {
                if segment_type == "planes" {
                    desc.push_str(&format!(", max_planes={}, plane_dist={}",
                        max_planes.unwrap_or(5),
                        plane_distance.unwrap_or(0.02)
                    ));
                }
                if let Some(ct) = cluster_tolerance {
                    desc.push_str(&format!(", cluster_tol={}, min_cluster={}",
                        ct,
                        min_cluster_size.unwrap_or(100)
                    ));
                }
            }
            desc
        }
        PipelineStep::Quality { threshold, weights, assess_completeness, auto_fix, octree_depth, noise_k, diff_with } => {
            let mut desc = format!("Quality (阈值={}", threshold.unwrap_or(60.0));
            if weights.is_some() { desc.push_str(", 自定义权重"); }
            if assess_completeness.unwrap_or(false) { desc.push_str(", 完整性评估"); }
            if auto_fix.unwrap_or(false) { desc.push_str(", 自动修复"); }
            if octree_depth.is_some() { desc.push_str(&format!(", octree_depth={}", octree_depth.unwrap())); }
            if noise_k.is_some() { desc.push_str(&format!(", noise_k={}", noise_k.unwrap())); }
            if diff_with.is_some() { desc.push_str(", 对比模式"); }
            desc.push_str(")");
            desc
        }
        PipelineStep::Align { source, target, ransac_iterations, icp_iterations, icp_threshold, voxel_size, normal_radius, fpfh_radius, rmse_warning_threshold, output_transformed_source, output_matrix, pass_transform_to_next } => {
            let mut desc = format!("Align (");
            if let Some(s) = source { desc.push_str(&format!("源: {}, ", s)); }
            if let Some(t) = target { desc.push_str(&format!("目标: {}", t)); }
            desc.push_str(&format!(", RANSAC迭代={}", ransac_iterations.unwrap_or(1000)));
            desc.push_str(&format!(", ICP迭代={}", icp_iterations.unwrap_or(50)));
            if let Some(v) = voxel_size { if *v > 0.0 { desc.push_str(&format!(", voxel={}", v)); } }
            if let Some(t) = rmse_warning_threshold { desc.push_str(&format!(", RMSE阈值={}", t)); }
            if pass_transform_to_next.unwrap_or(false) { desc.push_str(", 传递变换"); }
            desc.push_str(")");
            desc
        }
    }
}

pub fn register_multiple_clouds(
    clouds: &[PointCloud],
    params: &RegistrationParams,
) -> Result<(PointCloud, Vec<RegistrationStats>)> {
    if clouds.len() < 2 {
        return Ok((clouds.first().cloned().unwrap_or_default(), Vec::new()));
    }

    let mut merged = clouds[0].clone();
    let mut stats = Vec::new();

    for (i, cloud) in clouds.iter().enumerate().skip(1) {
        log::info!("配准点云 {}/{}", i + 1, clouds.len());
        let start = Instant::now();
        let result = register_point_clouds(cloud, &merged, params)?;

        let mut transformed = cloud.clone();
        transformed.apply_transform(&result.transform);
        merged.extend(transformed);

        let mut t_flat = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                t_flat[r][c] = result.transform[(r, c)];
            }
        }

        stats.push(RegistrationStats {
            register_type: "fpfh_ransac_icp".to_string(),
            rmse: result.rmse,
            iterations: result.iterations,
            converged: result.converged,
            transform_matrix: t_flat,
            time_ms: duration_to_ms(start.elapsed()),
        });

        log::info!("  配准完成: RMSE={:.6}, 迭代={}, 收敛={}",
            result.rmse, result.iterations, result.converged
        );
    }

    Ok((merged, stats))
}
