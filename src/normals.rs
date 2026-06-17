use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Point3D, KdTree};
use nalgebra::{Vector3, Matrix3, SymmetricEigen};
use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct NormalEstimationParams {
    pub k: usize,
    pub orientation_k: usize,
}

impl Default for NormalEstimationParams {
    fn default() -> Self {
        NormalEstimationParams {
            k: 20,
            orientation_k: 10,
        }
    }
}

pub struct NormalEstimationResult {
    pub point_cloud: PointCloud,
    pub mean_curvature: f64,
}

pub fn estimate_normals(
    pc: &PointCloud,
    params: &NormalEstimationParams,
) -> Result<NormalEstimationResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let n = pc.len();

    let normal_results: Vec<(Option<Vector3<f64>>, f64)> = pc
        .points
        .par_iter()
        .map(|p| compute_pca_normal(p, &kdtree, params.k))
        .collect();

    let mut new_points: Vec<Point3D> = pc.iter().cloned().collect();
    let mut curvatures: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if let Some(normal) = normal_results[i].0 {
            new_points[i].normal = Some(normal);
        }
        new_points[i].curvature = Some(normal_results[i].1);
        curvatures.push(normal_results[i].1);
    }

    let oriented = orient_normals_consistently(&new_points, &kdtree, params.orientation_k)?;

    let mean_curvature = if curvatures.is_empty() {
        0.0
    } else {
        curvatures.iter().sum::<f64>() / curvatures.len() as f64
    };

    Ok(NormalEstimationResult {
        point_cloud: oriented,
        mean_curvature,
    })
}

fn compute_pca_normal(
    point: &Point3D,
    kdtree: &KdTree,
    k: usize,
) -> (Option<Vector3<f64>>, f64) {
    let neighbors = kdtree.knn(&point.position, k);
    if neighbors.len() < 3 {
        return (None, 0.0);
    }

    let pts: Vec<&[f64; 3]> = Vec::new();
    let _ = pts;

    let mut mean = Vector3::zeros();
    for (idx, _) in &neighbors {
        let p = kdtree_point(kdtree, *idx);
        mean += p.coords;
    }
    mean /= neighbors.len() as f64;

    let mut cov = Matrix3::zeros();
    for (idx, _) in &neighbors {
        let p = kdtree_point(kdtree, *idx);
        let d = p.coords - mean;
        cov += d * d.transpose();
    }
    cov /= (neighbors.len() - 1) as f64;

    let eigen = SymmetricEigen::new(cov);
    let mut eig_pairs: Vec<(f64, Vector3<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(v, col)| (*v, col.clone_owned()))
        .collect();
    eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let normal = eig_pairs[0].1.normalize();
    let lambda_min = eig_pairs[0].0.max(0.0);
    let lambda_total = eig_pairs.iter().map(|(v, _)| v.max(0.0)).sum::<f64>().max(1e-15);
    let curvature = lambda_min / lambda_total;

    (Some(normal), curvature)
}

fn kdtree_point(kdtree: &KdTree, idx: usize) -> nalgebra::Point3<f64> {
    kdtree.points[idx]
}

fn orient_normals_consistently(
    points: &[Point3D],
    kdtree: &KdTree,
    orientation_k: usize,
) -> Result<PointCloud> {
    let n = points.len();
    let mut result: Vec<Point3D> = points.to_vec();

    let mut visited = vec![false; n];
    let mut component_id = 0usize;
    let mut components: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if visited[start] || result[start].normal.is_none() {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        let mut comp = vec![start];

        while let Some(idx) = queue.pop_front() {
            let neighbors = kdtree.knn(&points[idx].position, orientation_k + 1);
            for (nidx, _) in neighbors.iter().skip(1) {
                let nidx = *nidx;
                if visited[nidx] || result[nidx].normal.is_none() {
                    continue;
                }
                visited[nidx] = true;
                comp.push(nidx);
                queue.push_back(nidx);
            }
        }
        component_id += 1;
        components.push(comp);
    }

    for comp in &components {
        let &seed = comp.first().unwrap();
        if result[seed].normal.is_none() {
            continue;
        }

        let mut queue = VecDeque::new();
        queue.push_back(seed);
        let mut prop_visited = vec![false; n];
        prop_visited[seed] = true;

        let viewpoint = find_viewpoint(points, comp);

        if let Some(seed_normal) = result[seed].normal {
            let to_view = viewpoint - points[seed].position;
            if seed_normal.dot(&to_view) < 0.0 {
                if let Some(ref mut n) = result[seed].normal {
                    *n = -*n;
                }
            }
        }

        while let Some(idx) = queue.pop_front() {
            let current_normal = result[idx].normal.unwrap();
            let neighbors = kdtree.knn(&points[idx].position, orientation_k + 1);
            for (nidx, _) in neighbors.iter().skip(1) {
                let nidx = *nidx;
                if prop_visited[nidx] || result[nidx].normal.is_none() {
                    continue;
                }
                prop_visited[nidx] = true;
                if let Some(ref mut neighbor_normal) = result[nidx].normal {
                    if current_normal.dot(neighbor_normal) < 0.0 {
                        *neighbor_normal = -*neighbor_normal;
                    }
                }
                queue.push_back(nidx);
            }
        }
    }

    Ok(PointCloud::from_points(result))
}

fn find_viewpoint(points: &[Point3D], component: &[usize]) -> nalgebra::Point3<f64> {
    let mut centroid = Vector3::zeros();
    for &idx in component {
        centroid += points[idx].position.coords;
    }
    centroid /= component.len() as f64;

    let mut max_dist = 0.0f64;
    let mut far_point = centroid;
    for &idx in component {
        let d = (points[idx].position.coords - centroid).norm();
        if d > max_dist {
            max_dist = d;
            far_point = points[idx].position.coords;
        }
    }

    let direction = (far_point - centroid).normalize();
    nalgebra::Point3::from(centroid + direction * (max_dist + 10.0))
}

pub fn flip_normals_toward_viewpoint(
    pc: &mut PointCloud,
    viewpoint: nalgebra::Point3<f64>,
) {
    for p in pc.iter_mut() {
        if let Some(ref mut n) = p.normal {
            let to_view = viewpoint - p.position;
            if n.dot(&to_view) < 0.0 {
                *n = -*n;
            }
        }
    }
}
