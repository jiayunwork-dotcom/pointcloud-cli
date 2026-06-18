use clap::{Parser, Subcommand, ValueEnum};
use pointcloud_cli::*;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "pointcloud-cli",
    version = "0.1.0",
    about = "点云数据处理与三维表面重建命令行Pipeline工具",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, global = true, help = "启用详细日志输出")]
    verbose: bool,

    #[arg(short, long, global = true, help = "使用的线程数 (默认: 所有CPU核心)")]
    threads: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run {
        #[arg(short, long, help = "YAML配置文件路径")]
        config: PathBuf,

        #[arg(short, long, help = "覆盖输出路径")]
        output: Option<PathBuf>,

        #[arg(short, long, help = "覆盖报告路径")]
        report: Option<PathBuf>,

        #[arg(long, help = "试运行模式: 解析配置、校验文件、打印步骤但不执行")]
        dry_run: bool,
    },

    Diff {
        #[arg(help = "第一份Pipeline报告 (JSON)")]
        report_a: PathBuf,

        #[arg(help = "第二份Pipeline报告 (JSON)")]
        report_b: PathBuf,

        #[arg(long, help = "以JSON格式输出差异结果")]
        json: bool,
    },

    Info {
        #[arg(help = "点云文件路径")]
        input: PathBuf,
    },

    Convert {
        #[arg(help = "输入点云文件")]
        input: PathBuf,

        #[arg(help = "输出文件 (格式由扩展名推断: .ply/.obj/.stl)")]
        output: PathBuf,

        #[arg(short = 'f', long, help = "预处理: 统计滤波离群点")]
        filter: bool,

        #[arg(short = 'd', long, help = "预处理: 体素降采样大小 (如0.05)")]
        downsample: Option<f64>,

        #[arg(short = 'n', long, help = "法向量估计K近邻数 (启用法向量估计)")]
        normals: Option<usize>,

        #[arg(short = 'r', long, value_enum, help = "表面重建算法 (不指定则只做格式转换)")]
        reconstruct: Option<ReconAlgo>,

        #[arg(long, help = "Poisson重建深度 (默认8)")]
        poisson_depth: Option<u32>,

        #[arg(long, help = "Marching Cubes分辨率 (默认64)")]
        mc_resolution: Option<u32>,

        #[arg(long, help = "输出转换统计摘要到stderr (文件大小、精度损失、包围盒)")]
        stats: bool,
    },

    Batch {
        #[arg(help = "输入目录，包含所有点云文件")]
        input_dir: PathBuf,

        #[arg(help = "输出目录")]
        output_dir: PathBuf,

        #[arg(short, long, help = "使用的Pipeline YAML配置文件")]
        config: Option<PathBuf>,
    },

    Measure {
        #[command(subcommand)]
        action: MeasureActions,
    },

    ExampleConfig {
        #[arg(help = "输出示例配置文件路径")]
        output: Option<PathBuf>,
    },

    Benchmark {
        #[arg(help = "输入点云文件")]
        input: PathBuf,
    },

    Quality {
        #[command(subcommand)]
        action: QualityActions,
    },
}

#[derive(Subcommand, Debug)]
enum QualityActions {
    Assess {
        #[arg(help = "输入点云文件")]
        input: PathBuf,

        #[arg(long, help = "以JSON格式输出评估报告")]
        json: bool,

        #[arg(long, help = "执行自动修复")]
        fix: bool,

        #[arg(long, help = "修复后点云输出路径")]
        fix_output: Option<PathBuf>,

        #[arg(long, help = "五项权重,逗号分隔(密度,法向量,重叠,噪声,完整性),自动归一化")]
        weights: Option<String>,

        #[arg(long, help = "启用完整性评估(封闭曲面)")]
        assess_completeness: bool,

        #[arg(long, help = "八叉树最大深度 (默认: 6)")]
        octree_depth: Option<usize>,

        #[arg(long, help = "噪声评估K近邻数 (默认: 15)")]
        noise_k: Option<usize>,
    },

    Diff {
        #[arg(help = "修复前点云文件")]
        before: PathBuf,

        #[arg(help = "修复后点云文件")]
        after: PathBuf,

        #[arg(long, help = "以JSON格式输出对比结果")]
        json: bool,

        #[arg(long, help = "综合评分变化量阈值,低于此值则修复效果不达标 (默认: 0)")]
        threshold: Option<f64>,

        #[arg(long, help = "五项权重,逗号分隔(密度,法向量,重叠,噪声,完整性),自动归一化")]
        weights: Option<String>,

        #[arg(long, help = "启用完整性评估(封闭曲面)")]
        assess_completeness: bool,

        #[arg(long, help = "八叉树最大深度 (默认: 6)")]
        octree_depth: Option<usize>,

        #[arg(long, help = "噪声评估K近邻数 (默认: 15)")]
        noise_k: Option<usize>,
    },

    Batch {
        #[arg(help = "包含点云文件的目录路径")]
        input_dir: PathBuf,

        #[arg(long, help = "以JSON格式输出批量结果")]
        json: bool,

        #[arg(long, help = "对评分低于60的文件自动执行修复")]
        fix: bool,

        #[arg(long, help = "修复后文件的输出目录 (默认: 原文件同目录)")]
        output_dir: Option<PathBuf>,

        #[arg(long, help = "并发处理数 (默认: 1, 串行)")]
        parallel: Option<usize>,

        #[arg(long, help = "五项权重,逗号分隔(密度,法向量,重叠,噪声,完整性),自动归一化")]
        weights: Option<String>,

        #[arg(long, help = "启用完整性评估(封闭曲面)")]
        assess_completeness: bool,

        #[arg(long, help = "八叉树最大深度 (默认: 6)")]
        octree_depth: Option<usize>,

        #[arg(long, help = "噪声评估K近邻数 (默认: 15)")]
        noise_k: Option<usize>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum ReconAlgo {
    Poisson,
    BallPivoting,
    MarchingCubes,
}

#[derive(Subcommand, Debug)]
enum MeasureActions {
    Distance {
        #[arg(help = "点云文件")]
        input: PathBuf,
        #[arg(long, help = "第一个点的索引")]
        idx1: Option<usize>,
        #[arg(long, help = "第二个点的索引")]
        idx2: Option<usize>,
        #[arg(long, num_args = 3, help = "点A坐标: x y z")]
        p1: Option<Vec<f64>>,
        #[arg(long, num_args = 3, help = "点B坐标: x y z")]
        p2: Option<Vec<f64>>,
    },
    Volume {
        #[arg(help = "输入网格文件或点云文件")]
        input: PathBuf,
    },
    Section {
        #[arg(help = "点云文件")]
        input: PathBuf,
        #[arg(long, num_args = 3, help = "切割平面法向量 nx ny nz")]
        normal: Vec<f64>,
        #[arg(long, num_args = 3, help = "平面上一点 x y z")]
        point: Vec<f64>,
        #[arg(long, help = "截面厚度 (默认0.01)")]
        thickness: Option<f64>,
    },
}

fn main() {
    let cli = Cli::parse();

    let log_level = if cli.verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(format!("pointcloud_cli={}", log_level))
    ).init();

    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok();
    }

    let result = match cli.command {
        Commands::Run { config, output, report, dry_run } => run_pipeline(&config, output, report, dry_run),
        Commands::Diff { report_a, report_b, json } => run_diff(&report_a, &report_b, json),
        Commands::Info { input } => show_info(&input),
        Commands::Convert { input, output, filter, downsample, normals, reconstruct, poisson_depth, mc_resolution, stats }
            => do_convert(&input, &output, filter, downsample, normals, reconstruct, poisson_depth, mc_resolution, stats),
        Commands::Batch { input_dir, output_dir, config } => do_batch(&input_dir, &output_dir, config),
        Commands::Measure { action } => do_measurement(action),
        Commands::ExampleConfig { output } => write_example_config(output),
        Commands::Benchmark { input } => run_benchmark(&input),
        Commands::Quality { action }
            => run_quality(action),
    };

    if let Err(e) = result {
        eprintln!("\n错误: {}", e);
        std::process::exit(1);
    }
}

fn run_pipeline(config_path: &Path, output_override: Option<PathBuf>, report_override: Option<PathBuf>, dry_run: bool) -> Result<()> {
    log::info!("加载配置: {}", config_path.display());
    let mut config = config::PipelineConfig::from_yaml_file(config_path)?;

    if let Some(o) = output_override {
        config.output = o.to_string_lossy().to_string();
    }
    if let Some(r) = report_override {
        config.report_path = Some(r.to_string_lossy().to_string());
    }

    let engine = pipeline::PipelineEngine::new();

    if dry_run {
        engine.dry_run_from_config(&config)?;
    } else {
        let report = engine.run_from_config(&config)?;
        report.print_summary();
    }

    Ok(())
}

fn run_diff(report_a_path: &Path, report_b_path: &Path, as_json: bool) -> Result<()> {
    log::info!("加载报告 A: {}", report_a_path.display());
    let report_a = report::PipelineReport::load_from_json(report_a_path)?;

    log::info!("加载报告 B: {}", report_b_path.display());
    let report_b = report::PipelineReport::load_from_json(report_b_path)?;

    let diff = report::diff_reports(
        &report_a,
        &report_b,
        &report_a_path.to_string_lossy(),
        &report_b_path.to_string_lossy(),
    );

    if as_json {
        let json = diff.to_json_pretty()?;
        println!("{}", json);
    } else {
        diff.print_table();
    }

    Ok(())
}

fn show_info(input: &Path) -> Result<()> {
    println!("点云文件信息: {}", input.display());
    println!("{:-<60}", "");

    let reader = io::PointCloudReader::new();
    let pc = reader.read(input)?;

    if let Some(summary) = pc.summary() {
        println!("总点数:         {}", summary.total_points);
        println!("包围盒最小点:   ({:.4}, {:.4}, {:.4})",
            summary.bounding_box.min.x,
            summary.bounding_box.min.y,
            summary.bounding_box.min.z
        );
        println!("包围盒最大点:   ({:.4}, {:.4}, {:.4})",
            summary.bounding_box.max.x,
            summary.bounding_box.max.y,
            summary.bounding_box.max.z
        );
        let size = summary.bounding_box.size();
        println!("包围盒尺寸:     ({:.4}, {:.4}, {:.4})", size.x, size.y, size.z);
        println!("对角线长度:     {:.4}", summary.bounding_box.diagonal());
        println!("质心坐标:       ({:.4}, {:.4}, {:.4})",
            summary.centroid.x, summary.centroid.y, summary.centroid.z
        );
        println!("点密度估计:     {:.2} 点/立方米", summary.point_density);
        println!("包含颜色:       {}", summary.has_color);
        println!("包含法向量:     {}", summary.has_normals);
    }

    Ok(())
}

fn detect_format_float_precision(path: &Path) -> Option<&'static str> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "ply" => {
            let mut file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(_) => return None,
            };
            let mut reader = std::io::BufReader::new(&mut file);
            let mut line = String::new();
            let mut float_type: Option<&'static str> = None;
            loop {
                line.clear();
                let n = match std::io::BufRead::read_line(&mut reader, &mut line) {
                    Ok(n) => n,
                    Err(_) => break,
                };
                if n == 0 { break; }
                let trimmed = line.trim();
                if trimmed == "end_header" { break; }
                if trimmed.starts_with("property ") && (trimmed.contains(" x") || trimmed.contains(" y") || trimmed.contains(" z")) {
                    if trimmed.contains("float64") || trimmed.contains("double") {
                        float_type = Some("float64");
                        break;
                    } else if trimmed.contains("float32") || trimmed.contains("float ") {
                        float_type = Some("float32");
                        break;
                    }
                }
            }
            float_type
        }
        "xyz" => Some("float64 (ascii)"),
        "pcd" => Some("float32"),
        "las" | "laz" => Some("int32_scaled"),
        "obj" | "stl" => Some("float64 (ascii/混合)"),
        _ => None,
    }
}

fn bbox_almost_equal(a: &types::AABB, b: &types::AABB, eps: f64) -> bool {
    (a.min.x - b.min.x).abs() < eps
        && (a.min.y - b.min.y).abs() < eps
        && (a.min.z - b.min.z).abs() < eps
        && (a.max.x - b.max.x).abs() < eps
        && (a.max.y - b.max.y).abs() < eps
        && (a.max.z - b.max.z).abs() < eps
}

fn format_stats_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let f = n as f64;
    if f >= GB {
        format!("{:.2} GB ({})", f / GB, n)
    } else if f >= MB {
        format!("{:.2} MB ({})", f / MB, n)
    } else if f >= KB {
        format!("{:.2} KB ({})", f / KB, n)
    } else {
        format!("{} B", n)
    }
}

fn do_convert(
    input: &Path,
    output: &Path,
    do_filter: bool,
    downsample: Option<f64>,
    normals_k: Option<usize>,
    recon: Option<ReconAlgo>,
    poisson_depth: Option<u32>,
    mc_resolution: Option<u32>,
    enable_stats: bool,
) -> Result<()> {
    use reconstruction::*;
    use preprocess::*;
    use normals::*;

    let reader = io::PointCloudReader::new();
    let mesh_writer = io::MeshWriter::new();

    let input_size = std::fs::metadata(input).ok().map(|m| m.len());
    let input_precision = detect_format_float_precision(input);
    let input_bbox_orig: Option<types::AABB>;

    let total_start = std::time::Instant::now();
    log::info!("读取: {}", input.display());
    let mut pc = reader.read(input)?;
    let initial = pc.len();
    log::info!("  共 {} 点", initial);

    input_bbox_orig = pc.summary().map(|s| s.bounding_box);

    if do_filter {
        log::info!("统计滤波去噪...");
        let result = statistical_outlier_removal(&pc, &StatisticalFilterParams::default())?;
        log::info!("  剔除 {} 点 ({:.1}%)",
            result.removed_count, result.removed_ratio * 100.0
        );
        pc = result.kept_points;
    }

    if let Some(v) = downsample {
        log::info!("体素下采样 (大小={})...", v);
        let result = voxel_downsample(&pc, &VoxelDownsampleParams { voxel_size: v })?;
        log::info!("  {} -> {} 点", result.original_count, result.downsampled.len());
        pc = result.downsampled;
    }

    let need_reconstruct = recon.is_some();
    let need_normals = if let Some(r) = &recon {
        matches!(r, ReconAlgo::Poisson | ReconAlgo::BallPivoting) || normals_k.is_some()
    } else {
        normals_k.is_some()
    };

    if need_normals && !pc.has_normals() {
        let k = normals_k.unwrap_or(20);
        log::info!("法向量估计 (K={})...", k);
        let result = estimate_normals(&pc, &NormalEstimationParams { k, orientation_k: 10 })?;
        log::info!("  完成, 平均曲率 {:.6}", result.mean_curvature);
        pc = result.point_cloud;
    }

    let output_bbox_after_processing: Option<types::AABB> =
        pc.summary().map(|s| s.bounding_box);
    let output_points_count = pc.len();

    if need_reconstruct {
        let recon = recon.unwrap();
        log::info!("表面重建 ({:?})...", recon);
        let algorithm = match recon {
            ReconAlgo::Poisson => ReconstructionAlgorithm::Poisson,
            ReconAlgo::BallPivoting => ReconstructionAlgorithm::BallPivoting,
            ReconAlgo::MarchingCubes => ReconstructionAlgorithm::MarchingCubes,
        };
        let pp = PoissonParams { depth: poisson_depth.unwrap_or(8), ..Default::default() };
        let bp = BallPivotingParams::default();
        let mcp = MarchingCubesParams { resolution: mc_resolution.unwrap_or(64), ..Default::default() };

        let mesh = reconstruct_surface(&pc, algorithm, &pp, &bp, &mcp)?;
        let mesh_bbox = mesh.aabb();
        log::info!("  生成: {} 顶点, {} 面片", mesh.vertex_count(), mesh.face_count());

        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        mesh_writer.write(&mesh, output)?;
        log::info!("已写入网格: {}", output.display());

        let output_size = std::fs::metadata(output).ok().map(|m| m.len());
        let output_precision = detect_format_float_precision(output);

        let elapsed = total_start.elapsed();
        println!("\n处理完成!");
        println!("  初始点数:   {}", initial);
        println!("  处理后:     {} 点 -> {} 顶点, {} 面片",
            pc.len(), mesh.vertex_count(), mesh.face_count()
        );
        println!("  总耗时:     {:.2}s", elapsed.as_secs_f64());

        if enable_stats {
            eprintln!();
            eprintln!("{}", "=".repeat(60));
            eprintln!("转换统计摘要 (stderr)");
            eprintln!("{}", "=".repeat(60));
            eprintln!("【文件大小】");
            eprintln!("  输入: {}",
                input_size.map(format_stats_bytes).unwrap_or_else(|| "-".to_string()));
            eprintln!("  输出: {}",
                output_size.map(format_stats_bytes).unwrap_or_else(|| "-".to_string()));
            if let (Some(is), Some(os)) = (input_size, output_size) {
                let ratio = if is > 0 {
                    os as f64 / is as f64
                } else { 0.0 };
                let delta = os as i64 - is as i64;
                eprintln!("  变化: {:+} bytes ({:+.2}x)",
                    delta, ratio);
            }
            eprintln!();
            eprintln!("【精度分析】");
            eprintln!("  输入精度:   {}", input_precision.unwrap_or("未知"));
            eprintln!("  输出精度:   {}", output_precision.unwrap_or("未知"));
            let mut precision_loss_notes: Vec<&str> = Vec::new();
            match (input_precision, output_precision) {
                (Some(i), Some(o)) => {
                    if i.contains("64") && (o.contains("32") || o == "float32") {
                        precision_loss_notes.push("float64 → float32 可能存在数值截断");
                    }
                    if (i.contains("64") || i.contains("ascii")) && o.contains("32_scaled") {
                        precision_loss_notes.push("浮点 → 整数缩放编码 存在精度损失");
                    }
                }
                _ => {}
            }
            precision_loss_notes.push("表面重建是近似过程，不复原始点");
            let _ = recon;
            if precision_loss_notes.is_empty() {
                eprintln!("  精度损失:   无（无损或格式兼容）");
            } else {
                eprintln!("  精度损失:   有");
                for note in precision_loss_notes {
                    eprintln!("    ⚠ {}", note);
                }
            }
            eprintln!();
            eprintln!("【包围盒对比】");
            match (input_bbox_orig, mesh_bbox.or(output_bbox_after_processing)) {
                (Some(orig), Some(out)) => {
                    let eps = 1e-6;
                    let eq = bbox_almost_equal(&orig, &out, eps);
                    eprintln!("  输入包围盒:");
                    eprintln!("    min=({:.6}, {:.6}, {:.6})",
                        orig.min.x, orig.min.y, orig.min.z);
                    eprintln!("    max=({:.6}, {:.6}, {:.6})",
                        orig.max.x, orig.max.y, orig.max.z);
                    eprintln!("  输出包围盒:");
                    eprintln!("    min=({:.6}, {:.6}, {:.6})",
                        out.min.x, out.min.y, out.min.z);
                    eprintln!("    max=({:.6}, {:.6}, {:.6})",
                        out.max.x, out.max.y, out.max.z);
                    eprintln!("  一致性: {}",
                        if eq { "✓ 完全一致" } else { "✗ 存在差异（滤波/重建等原因）" });
                }
                _ => {
                    eprintln!("  无法对比包围盒信息");
                }
            }
            eprintln!("{}", "=".repeat(60));
            eprintln!();
        }
    } else {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let output_ext = output.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match output_ext.as_str() {
            "ply" => {
                io::write_point_cloud_ply(&pc, output)?;
                log::info!("已写入点云: {}", output.display());
            }
            _ => {
                return Err(crate::error::PointCloudError::ConfigError(
                    format!("不支持的点云输出格式: .{} (不做重建时仅支持 .ply 点云输出)", output_ext)
                ));
            }
        }

        let output_size = std::fs::metadata(output).ok().map(|m| m.len());
        let output_precision = detect_format_float_precision(output);
        let output_pc = reader.read(output).ok();
        let output_bbox_final = output_pc.as_ref().and_then(|p| p.summary().map(|s| s.bounding_box));

        let elapsed = total_start.elapsed();
        println!("\n格式转换完成!");
        println!("  点数:       {}", pc.len());
        println!("  总耗时:     {:.2}s", elapsed.as_secs_f64());

        if enable_stats {
            eprintln!();
            eprintln!("{}", "=".repeat(60));
            eprintln!("转换统计摘要 (stderr)");
            eprintln!("{}", "=".repeat(60));
            eprintln!("【文件大小】");
            eprintln!("  输入: {}",
                input_size.map(format_stats_bytes).unwrap_or_else(|| "-".to_string()));
            eprintln!("  输出: {}",
                output_size.map(format_stats_bytes).unwrap_or_else(|| "-".to_string()));
            if let (Some(is), Some(os)) = (input_size, output_size) {
                let ratio = if is > 0 {
                    os as f64 / is as f64
                } else { 0.0 };
                let delta = os as i64 - is as i64;
                eprintln!("  变化: {:+} bytes ({:+.2}x)",
                    delta, ratio);
            }
            eprintln!();
            eprintln!("【精度分析】");
            eprintln!("  输入精度:   {}", input_precision.unwrap_or("未知"));
            eprintln!("  输出精度:   {}", output_precision.unwrap_or("未知"));
            let mut precision_loss_notes: Vec<&str> = Vec::new();
            match (input_precision, output_precision) {
                (Some(i), Some(o)) => {
                    if i.contains("64") && (o.contains("32") || o == "float32") {
                        precision_loss_notes.push("float64 → float32 可能存在数值截断");
                    }
                    if (i.contains("64") || i.contains("ascii")) && o.contains("32_scaled") {
                        precision_loss_notes.push("浮点 → 整数缩放编码 存在精度损失");
                    }
                }
                _ => {}
            }
            if precision_loss_notes.is_empty() {
                eprintln!("  精度损失:   无（无损或格式兼容）");
            } else {
                eprintln!("  精度损失:   有");
                for note in precision_loss_notes {
                    eprintln!("    ⚠ {}", note);
                }
            }
            eprintln!();
            eprintln!("【包围盒对比】 (点数: {} → {})",
                initial, output_points_count);
            match (input_bbox_orig, output_bbox_final.or(output_bbox_after_processing)) {
                (Some(orig), Some(out)) => {
                    let eps = 1e-6;
                    let eq = bbox_almost_equal(&orig, &out, eps);
                    eprintln!("  输入包围盒:");
                    eprintln!("    min=({:.6}, {:.6}, {:.6})",
                        orig.min.x, orig.min.y, orig.min.z);
                    eprintln!("    max=({:.6}, {:.6}, {:.6})",
                        orig.max.x, orig.max.y, orig.max.z);
                    eprintln!("  输出包围盒:");
                    eprintln!("    min=({:.6}, {:.6}, {:.6})",
                        out.min.x, out.min.y, out.min.z);
                    eprintln!("    max=({:.6}, {:.6}, {:.6})",
                        out.max.x, out.max.y, out.max.z);
                    eprintln!("  一致性: {}",
                        if eq { "✓ 完全一致" } else { "✗ 存在差异（滤波/降采样原因）" });
                }
                _ => {
                    eprintln!("  无法对比包围盒信息");
                }
            }
            eprintln!("{}", "=".repeat(60));
            eprintln!();
        }
    }

    Ok(())
}

fn do_batch(input_dir: &Path, output_dir: &Path, config_file: Option<PathBuf>) -> Result<()> {
    let mut config = if let Some(cf) = config_file {
        config::PipelineConfig::from_yaml_file(&cf)?
    } else {
        config::PipelineConfig::from_yaml_str(&config::example_config_yaml())?
    };

    config.input_dir = Some(input_dir.to_string_lossy().to_string());
    config.output = "{stem}.obj".to_string();
    config.output_dir = Some(output_dir.to_string_lossy().to_string());
    config.output_intermediate = false;
    config.report_path = Some(output_dir.join("report.json").to_string_lossy().to_string());

    std::fs::create_dir_all(output_dir).ok();

    let engine = pipeline::PipelineEngine::new();
    let report = engine.run_from_config(&config)?;
    report.print_summary();

    Ok(())
}

fn do_measurement(action: MeasureActions) -> Result<()> {
    use measurement::*;
    use nalgebra::Point3;

    match action {
        MeasureActions::Distance { input, idx1, idx2, p1, p2 } => {
            let reader = io::PointCloudReader::new();
            let pc = reader.read(&input)?;

            let point_a = if let Some(c) = p1 {
                if c.len() == 3 { Point3::new(c[0], c[1], c[2]) } else {
                    return Err(PointCloudError::InvalidParameter("p1需要3个坐标".into()));
                }
            } else if let Some(i) = idx1 {
                if i >= pc.len() {
                    return Err(PointCloudError::InvalidParameter(format!("idx1超出范围 ({}/{})", i, pc.len())));
                }
                pc[i].position
            } else {
                return Err(PointCloudError::InvalidParameter("需要指定--idx1或--p1".into()));
            };

            let point_b = if let Some(c) = p2 {
                if c.len() == 3 { Point3::new(c[0], c[1], c[2]) } else {
                    return Err(PointCloudError::InvalidParameter("p2需要3个坐标".into()));
                }
            } else if let Some(i) = idx2 {
                if i >= pc.len() {
                    return Err(PointCloudError::InvalidParameter(format!("idx2超出范围 ({}/{})", i, pc.len())));
                }
                pc[i].position
            } else {
                return Err(PointCloudError::InvalidParameter("需要指定--idx2或--p2".into()));
            };

            let dist = distance_between_points(&point_a, &point_b);
            println!("两点距离:");
            println!("  A: ({:.4}, {:.4}, {:.4})", point_a.x, point_a.y, point_a.z);
            println!("  B: ({:.4}, {:.4}, {:.4})", point_b.x, point_b.y, point_b.z);
            println!("  欧氏距离: {:.6}", dist);
        }

        MeasureActions::Volume { input } => {
            let mesh_reader = io::MeshReader::new();
            let pc_reader = io::PointCloudReader::new();

            let is_mesh = matches!(crate::utils::detect_mesh_format(&input), Ok(_))
                && input.extension().and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase() == "obj" || e.to_lowercase() == "stl")
                    .unwrap_or(false);

            if is_mesh {
                let mesh = mesh_reader.read(&input)?;
                let result = measurement::estimate_mesh_volume(&mesh)?;
                println!("体积计算 (网格):");
                println!("  顶点数: {}", mesh.vertex_count());
                println!("  面片数: {}", mesh.face_count());
                println!("  体积: {:.6} 立方米", result.volume);
                println!("  注: 基于散度定理，适用于封闭流形网格");
            } else {
                match mesh_reader.read(&input) {
                    Ok(mesh) if mesh.face_count() > 0 => {
                        let result = measurement::estimate_mesh_volume(&mesh)?;
                        println!("体积计算 (网格):");
                        println!("  顶点数: {}", mesh.vertex_count());
                        println!("  面片数: {}", mesh.face_count());
                        println!("  体积: {:.6} 立方米", result.volume);
                        println!("  注: 基于散度定理，适用于封闭流形网格");
                    }
                    _ => {
                        let pc = pc_reader.read(&input)?;
                        let density = measurement::estimate_point_density(&pc, 10)?;
                        let vol = measurement::point_cloud_volume_convex_hull(&pc)?;
                        println!("体积估算 (点云，近似值):");
                        println!("  点云总点数: {}", pc.len());
                        println!("  平均密度: {:.2} 点/立方米", density.average_density);
                        println!("  近似体积 (AABB): {:.6} 立方米", vol);
                        println!("  注: 此为AABB体积近似，封闭网格体积更准确");
                    }
                }
            }
        }

        MeasureActions::Section { input, normal, point, thickness } => {
            if normal.len() != 3 {
                return Err(PointCloudError::InvalidParameter("normal需要3个值".into()));
            }
            if point.len() != 3 {
                return Err(PointCloudError::InvalidParameter("point需要3个值".into()));
            }
            let reader = io::PointCloudReader::new();
            let pc = reader.read(&input)?;
            let plane = CrossSectionPlane {
                normal: nalgebra::Vector3::new(normal[0], normal[1], normal[2]),
                point: Point3::new(point[0], point[1], point[2]),
            };
            let thick = thickness.unwrap_or(0.01);
            let result = cross_section_area(&pc, &plane, thick)?;
            println!("截面积计算:");
            println!("  平面法向量: ({:.4}, {:.4}, {:.4})", normal[0], normal[1], normal[2]);
            println!("  平面点:     ({:.4}, {:.4}, {:.4})", point[0], point[1], point[2]);
            println!("  截面厚度:   {}", thick);
            println!("  截面内点数: {}", result.boundary_points.len());
            println!("  凸包顶点数: {}", result.hull_points.len());
            println!("  截面积:     {:.6}", result.area);
            println!("  凸包周长:   {:.6}", result.perimeter);
        }
    }
    Ok(())
}

fn write_example_config(output: Option<PathBuf>) -> Result<()> {
    let yaml = config::example_config_yaml();
    match output {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&p, &yaml)?;
            println!("示例配置已写入: {}", p.display());
        }
        None => {
            println!("{}", yaml);
        }
    }
    Ok(())
}

fn run_benchmark(input: &Path) -> Result<()> {
    use preprocess::*;
    use normals::*;
    use reconstruction::*;

    println!("{:=<70}", "");
    println!("点云处理Pipeline 性能基准测试");
    println!("{:=<70}", "");

    let reader = io::PointCloudReader::new();
    let pc = reader.read(input)?;
    let n = pc.len();
    println!("输入文件: {}", input.display());
    println!("点数量:   {}", n);
    if let Some(s) = pc.summary() {
        println!("包围盒:   {:.2} x {:.2} x {:.2}", s.bounding_box.size().x, s.bounding_box.size().y, s.bounding_box.size().z);
    }
    println!();

    let benchmarks: Vec<(&str, Box<dyn Fn() -> Result<std::time::Duration>>)> = vec![
        ("KD-tree构建", Box::new(|| {
            let t = std::time::Instant::now();
            let _ = types::KdTree::from_point_cloud(&pc);
            Ok(t.elapsed())
        })),

        ("统计滤波(K=30)", Box::new(|| {
            let t = std::time::Instant::now();
            let _ = statistical_outlier_removal(&pc, &StatisticalFilterParams::default())?;
            Ok(t.elapsed())
        })),

        ("体素下采样(0.05)", Box::new(|| {
            let t = std::time::Instant::now();
            let _ = voxel_downsample(&pc, &VoxelDownsampleParams { voxel_size: 0.05 })?;
            Ok(t.elapsed())
        })),

        ("法向量估计(K=20)", Box::new(|| {
            let t = std::time::Instant::now();
            let _ = estimate_normals(&pc, &NormalEstimationParams { k: 20, orientation_k: 10 })?;
            Ok(t.elapsed())
        })),

        ("Poisson重建(深度=6)", Box::new(|| {
            let mut normal_pc = pc.clone();
            if !normal_pc.has_normals() {
                let r = estimate_normals(&normal_pc, &NormalEstimationParams { k: 20, orientation_k: 10 })?;
                normal_pc = r.point_cloud;
            }
            let t = std::time::Instant::now();
            let _ = poisson_reconstruction(&normal_pc, &PoissonParams { depth: 6, ..Default::default() })?;
            Ok(t.elapsed())
        })),

        ("MarchingCubes(分辨率=64)", Box::new(|| {
            let t = std::time::Instant::now();
            let _ = marching_cubes(&pc, &MarchingCubesParams { resolution: 64, ..Default::default() })?;
            Ok(t.elapsed())
        })),
    ];

    println!("{:<30} {:>15} {:>15}", "操作", "耗时", "吞吐量");
    println!("{:-<65}", "");

    for (name, f) in benchmarks {
        match f() {
            Ok(dur) => {
                let secs = dur.as_secs_f64();
                let throughput = if secs > 0.0 { n as f64 / secs / 1000.0 } else { f64::INFINITY };
                let dur_str = if secs < 1.0 {
                    format!("{:.1}ms", secs * 1000.0)
                } else {
                    format!("{:.2}s", secs)
                };
                let tp_str = if throughput.is_finite() {
                    format!("{:.1}K点/s", throughput)
                } else {
                    "N/A".to_string()
                };
                println!("{:<30} {:>15} {:>15}", name, dur_str, tp_str);
            }
            Err(e) => {
                println!("{:<30} 错误: {}", name, e);
            }
        }
    }

    println!("{:=<70}", "");
    Ok(())
}

fn parse_weights(weights_str: Option<String>) -> Result<quality::QualityWeights> {
    use quality::QualityWeights;
    if let Some(ws) = weights_str {
        let parts: Vec<&str> = ws.split(',').collect();
        if parts.len() != 5 {
            return Err(PointCloudError::InvalidParameter(
                "权重需要恰好5个值,用逗号分隔".to_string()
            ));
        }
        let parsed: std::result::Result<Vec<f64>, _> = parts.iter().map(|s| s.parse::<f64>()).collect();
        match parsed {
            Ok(w) => QualityWeights::from_slice(&w),
            Err(_) => {
                return Err(PointCloudError::InvalidParameter(
                    "权重值解析失败".to_string()
                ));
            }
        }
    } else {
        Ok(QualityWeights::default())
    }
}

fn build_quality_params(
    assess_completeness_flag: bool,
    octree_depth: Option<usize>,
    noise_k: Option<usize>,
) -> quality::QualityAssessmentParams {
    use quality::QualityAssessmentParams;
    let mut params = QualityAssessmentParams::default();
    params.assess_completeness = assess_completeness_flag;
    if let Some(d) = octree_depth {
        params.octree_max_depth = d;
    }
    if let Some(k) = noise_k {
        params.noise_k = k;
    }
    params
}

fn run_quality(action: QualityActions) -> Result<()> {
    match action {
        QualityActions::Assess { input, json, fix, fix_output, weights, assess_completeness, octree_depth, noise_k } => {
            run_quality_assess(&input, json, fix, fix_output, weights, assess_completeness, octree_depth, noise_k)
        }
        QualityActions::Diff { before, after, json, threshold, weights, assess_completeness, octree_depth, noise_k } => {
            run_quality_diff(&before, &after, json, threshold.unwrap_or(0.0), weights, assess_completeness, octree_depth, noise_k)
        }
        QualityActions::Batch { input_dir, json, fix, output_dir, parallel, weights, assess_completeness, octree_depth, noise_k } => {
            run_quality_batch(&input_dir, json, fix, output_dir.as_deref(), parallel.unwrap_or(1), weights, assess_completeness, octree_depth, noise_k)
        }
    }
}

fn run_quality_assess(
    input: &Path,
    as_json: bool,
    do_fix: bool,
    fix_output: Option<PathBuf>,
    weights_str: Option<String>,
    assess_completeness_flag: bool,
    octree_depth: Option<usize>,
    noise_k: Option<usize>,
) -> Result<()> {
    use quality::{
        QualityAssessmentParams, RepairParams,
        assess_quality, auto_repair, print_quality_report, quality_report_to_json,
    };

    let weights = parse_weights(weights_str)?;
    let params = build_quality_params(assess_completeness_flag, octree_depth, noise_k);

    log::info!("读取点云: {}", input.display());
    let reader = io::PointCloudReader::new();
    let pc = reader.read(input)?;
    log::info!("  共 {} 点", pc.len());

    log::info!("执行质量评估...");
    let report = assess_quality(&pc, &params, &weights)?;

    if as_json {
        let json = quality_report_to_json(&report)?;
        println!("{}", json);
    } else {
        print_quality_report(&report);
    }

    if do_fix {
        log::info!("执行自动修复...");
        let repair_params = RepairParams::default();
        let repair_result = auto_repair(&pc, &report, &repair_params)?;

        println!();
        println!("{}", "\x1b[1m修复统计:\x1b[0m");
        println!("  添加点数:   {}", repair_result.points_added);
        println!("  去除点数:   {}", repair_result.points_removed);
        println!("  修复法向量: {}", repair_result.normals_fixed);
        println!("  滤波迭代:   {}", repair_result.iterations);

        let output_path = fix_output.unwrap_or_else(|| {
            let stem = input.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            input.with_file_name(format!("{}_fixed.ply", stem))
        });

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        io::write_point_cloud_ply(&repair_result.point_cloud, &output_path)?;
        println!();
        println!("修复后点云已保存到: {}", output_path.display());

        if !as_json {
            log::info!("重新评估修复后点云质量...");
            let report_after = assess_quality(&repair_result.point_cloud, &params, &weights)?;
            println!();
            println!("{}", "\x1b[1m修复后质量:\x1b[0m");
            print_quality_report(&report_after);
        }
    }

    Ok(())
}

fn run_quality_diff(
    before_path: &Path,
    after_path: &Path,
    as_json: bool,
    threshold: f64,
    weights_str: Option<String>,
    assess_completeness_flag: bool,
    octree_depth: Option<usize>,
    noise_k: Option<usize>,
) -> Result<()> {
    use quality::{
        assess_quality, compare_quality_reports, print_quality_diff, quality_diff_to_json,
    };

    let weights = parse_weights(weights_str)?;
    let params = build_quality_params(assess_completeness_flag, octree_depth, noise_k);

    log::info!("读取修复前点云: {}", before_path.display());
    let reader = io::PointCloudReader::new();
    let pc_before = reader.read(before_path)?;
    log::info!("  共 {} 点", pc_before.len());

    log::info!("读取修复后点云: {}", after_path.display());
    let pc_after = reader.read(after_path)?;
    log::info!("  共 {} 点", pc_after.len());

    log::info!("评估修复前质量...");
    let report_before = assess_quality(&pc_before, &params, &weights)?;

    log::info!("评估修复后质量...");
    let report_after = assess_quality(&pc_after, &params, &weights)?;

    let diff = compare_quality_reports(&report_before, &report_after, threshold);

    if as_json {
        let json = quality_diff_to_json(&diff)?;
        println!("{}", json);
    } else {
        print_quality_diff(&diff);
    }

    if !diff.meets_threshold {
        if !as_json {
            eprintln!("修复效果不达标");
        }
        std::process::exit(1);
    }

    Ok(())
}

fn run_quality_batch(
    dir: &Path,
    as_json: bool,
    do_fix: bool,
    output_dir: Option<&Path>,
    parallel: usize,
    weights_str: Option<String>,
    assess_completeness_flag: bool,
    octree_depth: Option<usize>,
    noise_k: Option<usize>,
) -> Result<()> {
    use quality::{
        run_quality_batch as quality_batch, print_batch_summary, batch_result_to_json,
    };

    let weights = parse_weights(weights_str)?;
    let params = build_quality_params(assess_completeness_flag, octree_depth, noise_k);

    let result = quality_batch(dir, do_fix, output_dir, parallel, &params, &weights)?;

    if as_json {
        let json = batch_result_to_json(&result)?;
        println!("{}", json);
    } else {
        print_batch_summary(&result);
    }

    Ok(())
}
