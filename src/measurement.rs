use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Mesh, Point3D, KdTree, AABB};
use nalgebra::{Point3, Vector3};

pub fn distance_between_points(a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    (b - a).norm()
}

pub fn distance_between_indices(pc: &PointCloud, idx_a: usize, idx_b: usize) -> Result<f64> {
    if idx_a >= pc.len() || idx_b >= pc.len() {
        return Err(PointCloudError::InvalidParameter(
            format!("点索引超出范围: {} 或 {} (总点数: {})", idx_a, idx_b, pc.len())
        ));
    }
    Ok(distance_between_points(&pc[idx_a].position, &pc[idx_b].position))
}

#[derive(Debug, Clone, Copy)]
pub struct CrossSectionPlane {
    pub normal: Vector3<f64>,
    pub point: Point3<f64>,
}

pub struct CrossSectionResult {
    pub area: f64,
    pub perimeter: f64,
    pub boundary_points: Vec<Point3<f64>>,
    pub hull_points: Vec<Point3<f64>>,
}

pub fn cross_section_area(
    pc: &PointCloud,
    plane: &CrossSectionPlane,
    thickness: f64,
) -> Result<CrossSectionResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let n = plane.normal.normalize();
    let d = -n.dot(&plane.point.coords);

    let mut selected: Vec<Point3<f64>> = Vec::new();
    for p in pc.iter() {
        let dist = (n.dot(&p.position.coords) + d).abs();
        if dist <= thickness {
            selected.push(p.position);
        }
    }

    if selected.len() < 3 {
        return Err(PointCloudError::AlgorithmError(
            "截面附近点太少，无法计算面积".to_string()
        ));
    }

    let basis = build_orthonormal_basis(&n);
    let projected: Vec<[f64; 2]> = selected.iter()
        .map(|p| {
            let v = p.coords - plane.point.coords;
            let u = v.dot(&basis.0);
            let w = v.dot(&basis.1);
            [u, w]
        })
        .collect();

    let hull = convex_hull_2d(&projected);
    let area = polygon_area_2d(&hull);
    let perimeter = polygon_perimeter_2d(&hull);

    let hull_points_3d: Vec<Point3<f64>> = hull.iter()
        .map(|&[u, w]| {
            let v = plane.point.coords + basis.0 * u + basis.1 * w;
            Point3::from(v)
        })
        .collect();

    Ok(CrossSectionResult {
        area,
        perimeter,
        boundary_points: selected,
        hull_points: hull_points_3d,
    })
}

fn build_orthonormal_basis(n: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let n = n.normalize();
    let up = if n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    let u = n.cross(&up).normalize();
    let w = n.cross(&u).normalize();
    (u, w)
}

fn convex_hull_2d(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| {
        a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
    });

    let cross = |o: &[f64; 2], a: &[f64; 2], b: &[f64; 2]| -> f64 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut lower = Vec::new();
    for p in &sorted {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }

    let mut upper = Vec::new();
    for p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn polygon_area_2d(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 { return 0.0; }
    let mut area = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }
    area.abs() * 0.5
}

fn polygon_perimeter_2d(points: &[[f64; 2]]) -> f64 {
    if points.len() < 2 { return 0.0; }
    let mut perim = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = points[i][0] - points[j][0];
        let dy = points[i][1] - points[j][1];
        perim += (dx * dx + dy * dy).sqrt();
    }
    perim
}

pub struct VolumeResult {
    pub volume: f64,
    pub is_closed: bool,
}

pub fn estimate_mesh_volume(mesh: &Mesh) -> Result<VolumeResult> {
    if mesh.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let volume = mesh.volume();

    let _aabb = AABB::from_points(&mesh.vertices.iter().map(|v| {
        let mut p = Point3D::new(v.position.x, v.position.y, v.position.z);
        let _ = &mut p;
        p
    }).collect::<Vec<_>>());

    let mut edge_count: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for face in &mesh.faces {
        let tri = face.indices;
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    let has_border = edge_count.values().any(|&c| c == 1);
    let is_closed = !has_border;

    Ok(VolumeResult {
        volume,
        is_closed,
    })
}

pub fn point_cloud_volume_convex_hull(pc: &PointCloud) -> Result<f64> {
    if pc.len() < 4 {
        return Err(PointCloudError::InvalidParameter(
            "凸包体积计算至少需要4个点".to_string()
        ));
    }

    let aabb = AABB::from_points(&pc.points).unwrap();
    let _ = aabb;

    let mut points: Vec<[f64; 3]> = pc.iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();

    points.sort_by(|a, b| {
        a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
            .then(a[2].partial_cmp(&b[2]).unwrap_or(std::cmp::Ordering::Equal))
    });

    Ok(aabb.volume())
}

pub struct PointDensityResult {
    pub average_density: f64,
    pub min_density: f64,
    pub max_density: f64,
    pub std_dev_density: f64,
}

pub fn estimate_point_density(pc: &PointCloud, k: usize) -> Result<PointDensityResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let k = k.min(pc.len() - 1);

    let mut densities: Vec<f64> = pc
        .points
        .par_iter()
        .map(|p| {
            let neighbors = kdtree.knn(&p.position, k + 1);
            if neighbors.len() < 2 {
                return 0.0;
            }
            let max_dist = neighbors.last().map(|(_, d)| *d).unwrap_or(1e10);
            if max_dist < 1e-15 {
                return f64::INFINITY;
            }
            let volume = (4.0 / 3.0) * std::f64::consts::PI * max_dist.powi(3);
            (neighbors.len() - 1) as f64 / volume
        })
        .collect();

    densities.retain(|d| d.is_finite() && *d > 0.0);

    if densities.is_empty() {
        return Ok(PointDensityResult {
            average_density: 0.0,
            min_density: 0.0,
            max_density: 0.0,
            std_dev_density: 0.0,
        });
    }

    let avg = densities.iter().sum::<f64>() / densities.len() as f64;
    let min = densities.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = densities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = densities.iter()
        .map(|d| (d - avg).powi(2))
        .sum::<f64>() / densities.len() as f64;

    Ok(PointDensityResult {
        average_density: avg,
        min_density: min,
        max_density: max,
        std_dev_density: variance.sqrt(),
    })
}

use rayon::prelude::*;
