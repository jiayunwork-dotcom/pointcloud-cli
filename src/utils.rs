use std::time::Instant;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub fn time_it<F, R>(label: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    log::info!("{} 耗时: {:.3?}", label, duration);
    result
}

pub fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{:.1}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else {
        let mins = (secs / 60.0) as u64;
        let s = secs % 60.0;
        format!("{}m{:.1}s", mins, s)
    }
}

pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

pub fn random_unit_vector() -> nalgebra::Vector3<f64> {
    let mut rng = rand::thread_rng();
    loop {
        let v = nalgebra::Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        let len = v.norm();
        if len > 0.0 && len <= 1.0 {
            return v / len;
        }
    }
}

pub fn random_sample_indices(n: usize, k: usize) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rand::thread_rng());
    indices.truncate(k);
    indices
}

pub fn random_normal_f64(mean: f64, std: f64) -> f64 {
    let normal = Normal::new(mean, std).unwrap();
    normal.sample(&mut rand::thread_rng())
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

pub fn clamp(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

pub fn hash_point_to_grid(x: f64, y: f64, z: f64, cell_size: f64) -> (i64, i64, i64) {
    (
        (x / cell_size).floor() as i64,
        (y / cell_size).floor() as i64,
        (z / cell_size).floor() as i64,
    )
}

pub fn detect_file_format(path: &std::path::Path) -> crate::error::Result<crate::io::PointCloudFormat> {
    use crate::io::PointCloudFormat;
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            match ext.to_lowercase().as_str() {
                "ply" => Ok(PointCloudFormat::PLY),
                "pcd" => Ok(PointCloudFormat::PCD),
                "las" => Ok(PointCloudFormat::LAS),
                "laz" => Ok(PointCloudFormat::LAS),
                "xyz" => Ok(PointCloudFormat::XYZ),
                "txt" => Ok(PointCloudFormat::XYZ),
                other => Err(crate::error::PointCloudError::UnsupportedFormat(other.to_string())),
            }
        }
        None => Err(crate::error::PointCloudError::ParseError(
            "无法识别文件扩展名".to_string()
        )),
    }
}

pub fn detect_mesh_format(path: &std::path::Path) -> crate::error::Result<crate::io::MeshFormat> {
    use crate::io::MeshFormat;
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            match ext.to_lowercase().as_str() {
                "ply" => Ok(MeshFormat::PLY),
                "obj" => Ok(MeshFormat::OBJ),
                "stl" => Ok(MeshFormat::STL),
                other => Err(crate::error::PointCloudError::UnsupportedFormat(other.to_string())),
            }
        }
        None => Err(crate::error::PointCloudError::ParseError(
            "无法识别文件扩展名".to_string()
        )),
    }
}

pub fn union_find_parent(parent: &mut Vec<usize>, i: usize) -> usize {
    if parent[i] != i {
        parent[i] = union_find_parent(parent, parent[i]);
    }
    parent[i]
}

pub fn union_find_merge(parent: &mut Vec<usize>, rank: &mut Vec<usize>, a: usize, b: usize) {
    let ra = union_find_parent(parent, a);
    let rb = union_find_parent(parent, b);
    if ra == rb {
        return;
    }
    if rank[ra] < rank[rb] {
        parent[ra] = rb;
    } else if rank[ra] > rank[rb] {
        parent[rb] = ra;
    } else {
        parent[rb] = ra;
        rank[ra] += 1;
    }
}
