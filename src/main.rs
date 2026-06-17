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

        #[arg(short = 'r', long, value_enum, help = "表面重建算法", default_value_t = ReconAlgo::Poisson)]
        reconstruct: ReconAlgo,

        #[arg(long, help = "Poisson重建深度 (默认8)")]
        poisson_depth: Option<u32>,

        #[arg(long, help = "Marching Cubes分辨率 (默认64)")]
        mc_resolution: Option<u32>,
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
        Commands::Run { config, output, report } => run_pipeline(&config, output, report),
        Commands::Info { input } => show_info(&input),
        Commands::Convert { input, output, filter, downsample, normals, reconstruct, poisson_depth, mc_resolution }
            => do_convert(&input, &output, filter, downsample, normals, reconstruct, poisson_depth, mc_resolution),
        Commands::Batch { input_dir, output_dir, config } => do_batch(&input_dir, &output_dir, config),
        Commands::Measure { action } => do_measurement(action),
        Commands::ExampleConfig { output } => write_example_config(output),
        Commands::Benchmark { input } => run_benchmark(&input),
    };

    if let Err(e) = result {
        eprintln!("\n错误: {}", e);
        std::process::exit(1);
    }
}

fn run_pipeline(config_path: &Path, output_override: Option<PathBuf>, report_override: Option<PathBuf>) -> Result<()> {
    log::info!("加载配置: {}", config_path.display());
    let mut config = config::PipelineConfig::from_yaml_file(config_path)?;

    if let Some(o) = output_override {
        config.output = o.to_string_lossy().to_string();
    }
    if let Some(r) = report_override {
        config.report_path = Some(r.to_string_lossy().to_string());
    }

    let engine = pipeline::PipelineEngine::new();
    let report = engine.run_from_config(&config)?;
    report.print_summary();

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

fn do_convert(
    input: &Path,
    output: &Path,
    do_filter: bool,
    downsample: Option<f64>,
    normals_k: Option<usize>,
    recon: ReconAlgo,
    poisson_depth: Option<u32>,
    mc_resolution: Option<u32>,
) -> Result<()> {
    use reconstruction::*;
    use preprocess::*;
    use normals::*;

    let reader = io::PointCloudReader::new();
    let writer = io::MeshWriter::new();

    let total_start = std::time::Instant::now();
    log::info!("读取: {}", input.display());
    let mut pc = reader.read(input)?;
    let initial = pc.len();
    log::info!("  共 {} 点", initial);

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

    let need_normals = matches!(recon, ReconAlgo::Poisson | ReconAlgo::BallPivoting)
        || normals_k.is_some();

    if need_normals && !pc.has_normals() {
        let k = normals_k.unwrap_or(20);
        log::info!("法向量估计 (K={})...", k);
        let result = estimate_normals(&pc, &NormalEstimationParams { k, orientation_k: 10 })?;
        log::info!("  完成, 平均曲率 {:.6}", result.mean_curvature);
        pc = result.point_cloud;
    }

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
    log::info!("  生成: {} 顶点, {} 面片", mesh.vertex_count(), mesh.face_count());

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    writer.write(&mesh, output)?;
    log::info!("已写入: {}", output.display());

    let elapsed = total_start.elapsed();
    println!("\n处理完成!");
    println!("  初始点数:   {}", initial);
    println!("  处理后:     {} 点 -> {} 顶点, {} 面片",
        pc.len(), mesh.vertex_count(), mesh.face_count()
    );
    println!("  总耗时:     {:.2}s", elapsed.as_secs_f64());

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
            let format = crate::utils::detect_mesh_format(&input);
            match format {
                Ok(_) => {
                    log::warn!("暂不支持从文件读取网格，尝试作为点云读取估计体积");
                }
                Err(_) => {}
            }
            let reader = io::PointCloudReader::new();
            let pc = reader.read(&input)?;
            let density = estimate_point_density(&pc, 10)?;
            let vol = point_cloud_volume_convex_hull(&pc)?;
            println!("体积估算 (基于AABB，近似值):");
            println!("  点云总点数: {}", pc.len());
            println!("  平均密度: {:.2} 点/立方米", density.average_density);
            println!("  近似体积 (AABB): {:.6} 立方米", vol);
            println!("  注: 此为AABB体积近似，对封闭网格使用Convert生成后计算更准确");
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
