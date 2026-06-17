use crate::error::Result;
use crate::types::{PointCloud, Point3D, KdTree, AABB};
use nalgebra::{Point3, Vector3, Matrix3};
use rand::seq::SliceRandom;
use std::collections::{VecDeque, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct RANSACPlaneParams {
    pub distance_threshold: f64,
    pub max_iterations: usize,
    pub probability: f64,
    pub min_inliers: usize,
}

impl Default for RANSACPlaneParams {
    fn default() -> Self {
        RANSACPlaneParams {
            distance_threshold: 0.02,
            max_iterations: 10000,
            probability: 0.99,
            min_inliers: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlaneModel {
    pub normal: Vector3<f64>,
    pub d: f64,
    pub inlier_indices: Vec<usize>,
    pub confidence: f64,
}

impl PlaneModel {
    pub fn distance(&self, p: &Point3<f64>) -> f64 {
        (self.normal.dot(&p.coords) + self.d).abs() / self.normal.norm()
    }
}

pub fn ransac_detect_planes(
    pc: &PointCloud,
    params: &RANSACPlaneParams,
    max_planes: usize,
) -> Result<(Vec<PlaneModel>, PointCloud)> {
    if pc.is_empty() {
        return Ok((Vec::new(), PointCloud::new()));
    }

    let mut remaining_indices: Vec<usize> = (0..pc.len()).collect();
    let mut planes = Vec::new();

    for _ in 0..max_planes {
        if remaining_indices.len() < params.min_inliers { break; }

        if let Some(plane) = ransac_single_plane(pc, &remaining_indices, params) {
            if plane.inlier_indices.len() >= params.min_inliers {
                let inlier_set: HashSet<usize> = plane.inlier_indices.iter().cloned().collect();
                remaining_indices = remaining_indices
                    .into_iter()
                    .filter(|i| !inlier_set.contains(i))
                    .collect();
                planes.push(plane);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let remaining_points: Vec<Point3D> = remaining_indices
        .iter()
        .map(|&i| pc[i].clone())
        .collect();

    Ok((planes, PointCloud::from_points(remaining_points)))
}

fn ransac_single_plane(
    pc: &PointCloud,
    indices: &[usize],
    params: &RANSACPlaneParams,
) -> Option<PlaneModel> {
    let mut rng = rand::thread_rng();
    let mut best_inliers = Vec::new();
    let mut best_normal = Vector3::zeros();
    let mut best_d = 0.0f64;

    let mut iterations = params.max_iterations;
    let mut iter_count = 0usize;

    while iter_count < iterations {
        iter_count += 1;

        let sample: Vec<usize> = indices
            .choose_multiple(&mut rng, 3)
            .cloned()
            .collect();
        if sample.len() < 3 { continue; }

        let p0 = &pc[sample[0]].position;
        let p1 = &pc[sample[1]].position;
        let p2 = &pc[sample[2]].position;

        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let normal = v1.cross(&v2);
        let len = normal.norm();
        if len < 1e-15 { continue; }
        let normal = normal / len;

        if normal.norm() < 1e-10 { continue; }

        let d = -normal.dot(&p0.coords);

        let mut inliers = Vec::new();
        for &idx in indices {
            let dist = (normal.dot(&pc[idx].position.coords) + d).abs() / normal.norm();
            if dist <= params.distance_threshold {
                inliers.push(idx);
            }
        }

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_normal = normal;
            best_d = d;

            let inlier_ratio = best_inliers.len() as f64 / indices.len() as f64;
            if inlier_ratio > 0.0 {
                let w = inlier_ratio;
                let prob_no_outliers = 1.0 - w.powi(3);
                if prob_no_outliers > 0.0 {
                    iterations = ((1.0 - params.probability).ln() / prob_no_outliers.ln()).ceil() as usize;
                    iterations = iterations.min(params.max_iterations);
                }
            }
        }
    }

    if best_inliers.is_empty() {
        return None;
    }

    let confidence = best_inliers.len() as f64 / indices.len() as f64;
    Some(PlaneModel {
        normal: best_normal,
        d: best_d,
        inlier_indices: best_inliers,
        confidence,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct RANSACCylinderParams {
    pub distance_threshold: f64,
    pub normal_threshold: f64,
    pub max_iterations: usize,
    pub min_radius: f64,
    pub max_radius: f64,
    pub min_inliers: usize,
}

impl Default for RANSACCylinderParams {
    fn default() -> Self {
        RANSACCylinderParams {
            distance_threshold: 0.02,
            normal_threshold: 0.1,
            max_iterations: 50000,
            min_radius: 0.01,
            max_radius: 1.0,
            min_inliers: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CylinderModel {
    pub axis_direction: Vector3<f64>,
    pub axis_point: Point3<f64>,
    pub radius: f64,
    pub inlier_indices: Vec<usize>,
    pub confidence: f64,
}

pub fn euclidean_clustering(
    pc: &PointCloud,
    tolerance: f64,
    min_cluster_size: usize,
    max_cluster_size: Option<usize>,
) -> Result<Vec<Vec<usize>>> {
    if pc.is_empty() {
        return Ok(Vec::new());
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let mut visited = vec![false; pc.len()];
    let mut clusters = Vec::new();

    for start in 0..pc.len() {
        if visited[start] { continue; }

        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        let mut cluster = vec![start];

        while let Some(idx) = queue.pop_front() {
            let neighbors = kdtree.radius_search(&pc[idx].position, tolerance);
            for (nidx, _) in neighbors {
                if !visited[nidx] {
                    visited[nidx] = true;
                    cluster.push(nidx);
                    if let Some(max) = max_cluster_size {
                        if cluster.len() > max { break; }
                    }
                    queue.push_back(nidx);
                }
            }
        }

        if cluster.len() >= min_cluster_size {
            if let Some(max) = max_cluster_size {
                if cluster.len() <= max {
                    clusters.push(cluster);
                }
            } else {
                clusters.push(cluster);
            }
        }
    }

    clusters.sort_by(|a, b| b.len().cmp(&a.len()));
    Ok(clusters)
}

#[derive(Debug, Clone, Copy)]
pub struct RegionGrowingParams {
    pub curvature_threshold: f64,
    pub smoothness_threshold_deg: f64,
    pub min_cluster_size: usize,
    pub max_cluster_size: Option<usize>,
    pub neighbor_k: usize,
}

impl Default for RegionGrowingParams {
    fn default() -> Self {
        RegionGrowingParams {
            curvature_threshold: 0.1,
            smoothness_threshold_deg: 30.0,
            min_cluster_size: 100,
            max_cluster_size: None,
            neighbor_k: 30,
        }
    }
}

pub fn region_growing_segmentation(
    pc: &PointCloud,
    params: &RegionGrowingParams,
) -> Result<Vec<Vec<usize>>> {
    if pc.is_empty() {
        return Ok(Vec::new());
    }
    if !pc.has_normals() {
        return Err(crate::error::PointCloudError::NormalsNotComputed);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let smoothness_cos = params.smoothness_threshold_deg.to_radians().cos();

    let mut sorted_indices: Vec<usize> = (0..pc.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        let ca = pc[a].curvature.unwrap_or(1.0);
        let cb = pc[b].curvature.unwrap_or(1.0);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut visited = vec![false; pc.len()];
    let mut clusters = Vec::new();

    for &seed in &sorted_indices {
        if visited[seed] { continue; }
        if let Some(curv) = pc[seed].curvature {
            if curv > params.curvature_threshold { continue; }
        }

        let mut queue = VecDeque::new();
        queue.push_back(seed);
        visited[seed] = true;
        let mut cluster = vec![seed];
        let mut reached_max = false;

        while let Some(current) = queue.pop_front() {
            if let Some(max) = params.max_cluster_size {
                if cluster.len() >= max { reached_max = true; break; }
            }

            let neighbors = kdtree.knn(&pc[current].position, params.neighbor_k + 1);
            let current_normal = pc[current].normal.unwrap_or(Vector3::z());

            for (nidx, _) in neighbors.iter().skip(1) {
                if visited[*nidx] { continue; }

                if let Some(curv) = pc[*nidx].curvature {
                    if curv > params.curvature_threshold { continue; }
                }

                let neighbor_normal = pc[*nidx].normal.unwrap_or(Vector3::z());
                let angle_cos = current_normal.dot(&neighbor_normal).abs();

                if angle_cos >= smoothness_cos {
                    visited[*nidx] = true;
                    cluster.push(*nidx);
                    queue.push_back(*nidx);
                }
            }
        }

        if cluster.len() >= params.min_cluster_size && !reached_max {
            clusters.push(cluster);
        }
    }

    clusters.sort_by(|a, b| b.len().cmp(&a.len()));
    Ok(clusters)
}
