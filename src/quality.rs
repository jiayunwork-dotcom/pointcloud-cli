use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Point3D, KdTree, AABB, Color};
use crate::utils::{mean, std_dev};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

const DEFAULT_WEIGHTS: [f64; 5] = [0.2, 0.2, 0.15, 0.3, 0.15];

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QualityWeights {
    pub density: f64,
    pub normal: f64,
    pub overlap: f64,
    pub noise: f64,
    pub completeness: f64,
}

impl Default for QualityWeights {
    fn default() -> Self {
        QualityWeights {
            density: DEFAULT_WEIGHTS[0],
            normal: DEFAULT_WEIGHTS[1],
            overlap: DEFAULT_WEIGHTS[2],
            noise: DEFAULT_WEIGHTS[3],
            completeness: DEFAULT_WEIGHTS[4],
        }
    }
}

impl QualityWeights {
    pub fn from_slice(weights: &[f64]) -> Result<Self> {
        if weights.len() != 5 {
            return Err(PointCloudError::InvalidParameter(
                "权重需要恰好5个值".to_string()
            ));
        }
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(PointCloudError::InvalidParameter(
                "权重之和必须大于0".to_string()
            ));
        }
        Ok(QualityWeights {
            density: weights[0] / sum,
            normal: weights[1] / sum,
            overlap: weights[2] / sum,
            noise: weights[3] / sum,
            completeness: weights[4] / sum,
        })
    }

    pub fn total(&self) -> f64 {
        self.density + self.normal + self.overlap + self.noise + self.completeness
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMetricResult {
    pub score: f64,
    pub cv: f64,
    pub mean_points_per_leaf: f64,
    pub std_points_per_leaf: f64,
    pub leaf_count: usize,
    pub min_points_in_leaf: usize,
    pub max_points_in_leaf: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMetricResult {
    pub score: f64,
    pub flip_rate: f64,
    pub total_pairs: usize,
    pub flipped_pairs: usize,
    pub mean_angle_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapMetricResult {
    pub score: f64,
    pub overlap_rate: f64,
    pub overlap_threshold: f64,
    pub total_points: usize,
    pub overlapped_points: usize,
    pub avg_spacing: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseMetricResult {
    pub score: f64,
    pub normalized_noise: f64,
    pub rms_residual: f64,
    pub bbox_diagonal: f64,
    pub k_neighbors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessMetricResult {
    pub score: f64,
    pub hole_count: usize,
    pub total_boundary_length: f64,
    pub large_holes: usize,
    pub boundary_threshold: f64,
    pub assessed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub overall_score: f64,
    pub density: DensityMetricResult,
    pub normal: NormalMetricResult,
    pub overlap: OverlapMetricResult,
    pub noise: NoiseMetricResult,
    pub completeness: CompletenessMetricResult,
    pub weights: QualityWeights,
    pub total_points: usize,
    pub has_normals: bool,
    pub bounding_box_diagonal: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QualityAssessmentParams {
    pub octree_max_depth: usize,
    pub octree_min_points: usize,
    pub normal_k: usize,
    pub overlap_threshold_factor: f64,
    pub noise_k: usize,
    pub completeness_alpha: f64,
    pub boundary_threshold: f64,
    pub assess_completeness: bool,
}

impl Default for QualityAssessmentParams {
    fn default() -> Self {
        QualityAssessmentParams {
            octree_max_depth: 6,
            octree_min_points: 10,
            normal_k: 10,
            overlap_threshold_factor: 0.1,
            noise_k: 15,
            completeness_alpha: 0.1,
            boundary_threshold: 5.0,
            assess_completeness: false,
        }
    }
}

#[derive(Debug, Clone)]
struct OctreeNode {
    aabb: AABB,
    children: Option<Vec<Box<OctreeNode>>>,
    point_indices: Vec<usize>,
    depth: usize,
}

impl OctreeNode {
    fn new(aabb: AABB, depth: usize) -> Self {
        OctreeNode {
            aabb,
            children: None,
            point_indices: Vec::new(),
            depth,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    fn get_child_index(aabb: &AABB, point: &nalgebra::Point3<f64>) -> usize {
        let center = aabb.center();
        let mut idx = 0;
        if point.x >= center.x { idx |= 1; }
        if point.y >= center.y { idx |= 2; }
        if point.z >= center.z { idx |= 4; }
        idx
    }
}

fn build_octree(pc: &PointCloud, max_depth: usize, min_points: usize) -> OctreeNode {
    let aabb = AABB::from_points(&pc.points).unwrap_or_else(|| {
        AABB::new(
            Point3D::new(0.0, 0.0, 0.0).position,
            Point3D::new(1.0, 1.0, 1.0).position,
        )
    });

    let root = OctreeNode::new(aabb, 0);
    let indices: Vec<usize> = (0..pc.len()).collect();

    build_octree_recursive(pc, root, &indices, max_depth, min_points)
}

fn build_octree_recursive(
    pc: &PointCloud,
    mut node: OctreeNode,
    indices: &[usize],
    max_depth: usize,
    min_points: usize,
) -> OctreeNode {
    node.point_indices = indices.to_vec();

    if node.depth >= max_depth || indices.len() <= min_points {
        return node;
    }

    let center = node.aabb.center();
    let min = node.aabb.min;
    let max = node.aabb.max;

    let mut child_indices: [Vec<usize>; 8] = Default::default();

    for &idx in indices {
        let p = &pc[idx].position;
        let cidx = OctreeNode::get_child_index(&node.aabb, p);
        child_indices[cidx].push(idx);
    }

    let mut children = Vec::with_capacity(8);
    for i in 0..8 {
        let cmin = nalgebra::Point3::new(
            if i & 1 == 0 { min.x } else { center.x },
            if i & 2 == 0 { min.y } else { center.y },
            if i & 4 == 0 { min.z } else { center.z },
        );
        let cmax = nalgebra::Point3::new(
            if i & 1 == 0 { center.x } else { max.x },
            if i & 2 == 0 { center.y } else { max.y },
            if i & 4 == 0 { center.z } else { max.z },
        );
        let child_node = OctreeNode::new(AABB::new(cmin, cmax), node.depth + 1);
        let child = build_octree_recursive(pc, child_node, &child_indices[i], max_depth, min_points);
        children.push(Box::new(child));
    }

    node.children = Some(children);
    node
}

fn collect_leaf_point_counts(node: &OctreeNode, counts: &mut Vec<usize>) {
    if node.is_leaf() {
        if !node.point_indices.is_empty() {
            counts.push(node.point_indices.len());
        }
    } else if let Some(ref children) = node.children {
        for child in children.iter() {
            collect_leaf_point_counts(child, counts);
        }
    }
}

fn collect_leaf_nodes<'a>(node: &'a OctreeNode, leaves: &mut Vec<&'a OctreeNode>) {
    if node.is_leaf() {
        if !node.point_indices.is_empty() {
            leaves.push(node);
        }
    } else if let Some(ref children) = node.children {
        for child in children.iter() {
            collect_leaf_nodes(child, leaves);
        }
    }
}

pub fn assess_density_uniformity(
    pc: &PointCloud,
    params: &QualityAssessmentParams,
) -> Result<DensityMetricResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let octree = build_octree(pc, params.octree_max_depth, params.octree_min_points);
    let mut counts = Vec::new();
    collect_leaf_point_counts(&octree, &mut counts);

    if counts.is_empty() {
        return Ok(DensityMetricResult {
            score: 0.0,
            cv: 1.0,
            mean_points_per_leaf: 0.0,
            std_points_per_leaf: 0.0,
            leaf_count: 0,
            min_points_in_leaf: 0,
            max_points_in_leaf: 0,
        });
    }

    let counts_f64: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let mean_pts = mean(&counts_f64);
    let std_pts = std_dev(&counts_f64);
    let cv = if mean_pts > 0.0 { std_pts / mean_pts } else { 0.0 };

    let score = if cv <= 0.3 {
        100.0
    } else {
        (100.0 - 200.0 * (cv - 0.3)).max(0.0)
    };

    let min_pts = *counts.iter().min().unwrap_or(&0);
    let max_pts = *counts.iter().max().unwrap_or(&0);

    Ok(DensityMetricResult {
        score,
        cv,
        mean_points_per_leaf: mean_pts,
        std_points_per_leaf: std_pts,
        leaf_count: counts.len(),
        min_points_in_leaf: min_pts,
        max_points_in_leaf: max_pts,
    })
}

pub fn assess_normal_consistency(
    pc: &PointCloud,
    params: &QualityAssessmentParams,
) -> Result<NormalMetricResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    if !pc.has_normals() {
        return Ok(NormalMetricResult {
            score: 0.0,
            flip_rate: 1.0,
            total_pairs: 0,
            flipped_pairs: 0,
            mean_angle_deg: 180.0,
        });
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let k = params.normal_k.min(pc.len().saturating_sub(1));

    let results: Vec<(usize, f64)> = pc
        .points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let neighbors = kdtree.knn(&p.position, k + 1);
            let mut flipped = 0;
            let mut total = 0;
            let mut angle_sum = 0.0;

            if let Some(ref n_i) = p.normal {
                for (j, _) in neighbors.iter().skip(1) {
                    if *j == i { continue; }
                    if let Some(ref n_j) = pc[*j].normal {
                        let dot = n_i.dot(n_j).clamp(-1.0, 1.0);
                        let angle = dot.acos().to_degrees();
                        angle_sum += angle;
                        total += 1;
                        if angle > 90.0 {
                            flipped += 1;
                        }
                    }
                }
            }

            (flipped, if total > 0 { angle_sum / total as f64 } else { 0.0 })
        })
        .map(|r| (r.0, r.1))
        .collect();

    let mut total_flipped = 0usize;
    let mut total_pairs = 0usize;
    let mut angle_sum = 0.0f64;

    for (flipped, avg_angle) in &results {
        total_flipped += flipped;
        total_pairs += k;
        angle_sum += avg_angle;
    }

    let flip_rate = if total_pairs > 0 {
        total_flipped as f64 / total_pairs as f64
    } else {
        0.0
    };

    let mean_angle = if results.len() > 0 {
        angle_sum / results.len() as f64
    } else {
        0.0
    };

    let score = if flip_rate <= 0.01 {
        100.0
    } else if flip_rate >= 0.1 {
        0.0
    } else {
        100.0 * (0.1 - flip_rate) / (0.1 - 0.01)
    };

    Ok(NormalMetricResult {
        score,
        flip_rate,
        total_pairs,
        flipped_pairs: total_flipped,
        mean_angle_deg: mean_angle,
    })
}

pub fn assess_overlap(
    pc: &PointCloud,
    params: &QualityAssessmentParams,
) -> Result<OverlapMetricResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let kdtree = KdTree::from_point_cloud(pc);

    let avg_spacing = estimate_average_spacing(pc, &kdtree, 6);
    let threshold = avg_spacing * params.overlap_threshold_factor;
    let _threshold_sq = threshold * threshold;

    let mut redundant_count = 0usize;
    let mut kept = vec![false; pc.len()];

    for i in 0..pc.len() {
        if kept[i] { continue; }
        let neighbors = kdtree.radius_search(&pc[i].position, threshold);
        let cluster_size = neighbors.len();
        if cluster_size > 1 {
            redundant_count += cluster_size - 1;
        }
        for (idx, _) in &neighbors {
            kept[*idx] = true;
        }
    }

    let overlap_rate = if pc.len() > 0 {
        (redundant_count as f64 / pc.len() as f64).min(1.0)
    } else {
        0.0
    };

    let score = if overlap_rate <= 0.005 {
        100.0
    } else if overlap_rate >= 0.05 {
        0.0
    } else {
        100.0 * (0.05 - overlap_rate) / (0.05 - 0.005)
    };

    Ok(OverlapMetricResult {
        score,
        overlap_rate,
        overlap_threshold: threshold,
        total_points: pc.len(),
        overlapped_points: redundant_count,
        avg_spacing,
    })
}

fn estimate_average_spacing(pc: &PointCloud, kdtree: &KdTree, k: usize) -> f64 {
    let n = pc.len().min(1000);
    let step = (pc.len() / n.max(1)).max(1);
    let mut total_dist = 0.0f64;
    let mut count = 0usize;

    for i in (0..pc.len()).step_by(step).take(n) {
        let neighbors = kdtree.knn(&pc[i].position, k + 1);
        for (_, d) in neighbors.iter().skip(1).take(k) {
            total_dist += *d;
            count += 1;
        }
    }

    if count > 0 {
        total_dist / count as f64
    } else {
        1.0
    }
}

pub fn assess_noise_level(
    pc: &PointCloud,
    params: &QualityAssessmentParams,
) -> Result<NoiseMetricResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let k = params.noise_k.min(pc.len().saturating_sub(1));

    let aabb = AABB::from_points(&pc.points).unwrap();
    let bbox_diag = aabb.diagonal().max(1e-10);

    let residuals: Vec<f64> = pc
        .points
        .par_iter()
        .map(|p| {
            let neighbors = kdtree.knn(&p.position, k + 1);
            if neighbors.len() < 4 {
                return 0.0;
            }

            let pts: Vec<nalgebra::Point3<f64>> = neighbors
                .iter()
                .take(k + 1)
                .map(|(idx, _)| kdtree.points[*idx])
                .collect();

            let residual = point_to_plane_residual(&p.position, &pts);
            residual
        })
        .collect();

    let rms = if !residuals.is_empty() {
        let sum_sq: f64 = residuals.iter().map(|r| r * r).sum();
        (sum_sq / residuals.len() as f64).sqrt()
    } else {
        0.0
    };

    let normalized_noise = rms / bbox_diag;

    let score = if normalized_noise <= 0.001 {
        100.0
    } else if normalized_noise >= 0.01 {
        0.0
    } else {
        let log_low = 0.001f64.log10();
        let log_high = 0.01f64.log10();
        let log_val = normalized_noise.log10();
        100.0 * (log_high - log_val) / (log_high - log_low)
    };

    Ok(NoiseMetricResult {
        score,
        normalized_noise,
        rms_residual: rms,
        bbox_diagonal: bbox_diag,
        k_neighbors: k,
    })
}

fn point_to_plane_residual(
    point: &nalgebra::Point3<f64>,
    neighbors: &[nalgebra::Point3<f64>],
) -> f64 {
    let n = neighbors.len();
    if n < 3 {
        return 0.0;
    }

    let mut centroid = nalgebra::Point3::origin();
    for p in neighbors {
        centroid.coords += p.coords;
    }
    centroid.coords /= n as f64;

    let mut cov = nalgebra::Matrix3::zeros();
    for p in neighbors {
        let d = p.coords - centroid.coords;
        cov += d * d.transpose();
    }
    cov /= (n - 1).max(1) as f64;

    let eigen = nalgebra::SymmetricEigen::new(cov);
    let mut eig_pairs: Vec<(f64, nalgebra::Vector3<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(v, col)| (*v, col.clone_owned()))
        .collect();
    eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let normal = eig_pairs[0].1.normalize();
    let d = -normal.dot(&centroid.coords);

    let dist = (normal.dot(&point.coords) + d).abs();
    dist
}

pub fn assess_completeness(
    pc: &PointCloud,
    params: &QualityAssessmentParams,
) -> Result<CompletenessMetricResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    if !params.assess_completeness {
        return Ok(CompletenessMetricResult {
            score: 100.0,
            hole_count: 0,
            total_boundary_length: 0.0,
            large_holes: 0,
            boundary_threshold: params.boundary_threshold,
            assessed: false,
        });
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let k = 15.min(pc.len().saturating_sub(1));

    let avg_spacing = estimate_average_spacing(pc, &kdtree, 6);
    let alpha = params.completeness_alpha.max(avg_spacing * 2.0);

    let boundary_points: Vec<bool> = pc
        .points
        .par_iter()
        .map(|p| is_boundary_point(p, &kdtree, k, alpha))
        .collect();

    let boundary_count = boundary_points.iter().filter(|&&b| b).count();

    let mut visited = vec![false; pc.len()];
    let mut hole_count = 0usize;
    let mut large_holes = 0usize;
    let mut total_boundary_length = 0.0f64;

    for i in 0..pc.len() {
        if !boundary_points[i] || visited[i] {
            continue;
        }

        hole_count += 1;
        let cluster = boundary_cluster(i, pc, &kdtree, &boundary_points, &mut visited, avg_spacing * 3.0);
        let cluster_len = cluster.len();
        let approx_length = cluster_len as f64 * avg_spacing;
        total_boundary_length += approx_length;

        if approx_length > params.boundary_threshold {
            large_holes += 1;
        }
    }

    let score = if large_holes == 0 {
        100.0
    } else {
        (100.0 - large_holes as f64 * 20.0 - hole_count as f64 * 2.0).max(0.0)
    };

    Ok(CompletenessMetricResult {
        score,
        hole_count,
        total_boundary_length,
        large_holes,
        boundary_threshold: params.boundary_threshold,
        assessed: true,
    })
}

fn is_boundary_point(
    p: &Point3D,
    kdtree: &KdTree,
    k: usize,
    alpha: f64,
) -> bool {
    let neighbors = kdtree.knn(&p.position, k + 1);
    let n_neighbors = neighbors.len().min(k + 1);

    if n_neighbors < 3 {
        return true;
    }

    let mut angles: Vec<f64> = Vec::with_capacity(n_neighbors - 1);
    for i in 1..n_neighbors {
        for j in i + 1..n_neighbors {
            let pi = kdtree.points[neighbors[i].0] - p.position;
            let pj = kdtree.points[neighbors[j].0] - p.position;
            let dot = pi.normalize().dot(&pj.normalize()).clamp(-1.0, 1.0);
            angles.push(dot.acos());
        }
    }

    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(&max_angle) = angles.last() {
        if max_angle > 2.5 {
            return true;
        }
    }

    false
}

fn boundary_cluster(
    start: usize,
    pc: &PointCloud,
    kdtree: &KdTree,
    is_boundary: &[bool],
    visited: &mut [bool],
    radius: f64,
) -> Vec<usize> {
    let mut cluster = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;

    while let Some(idx) = queue.pop_front() {
        cluster.push(idx);
        let neighbors = kdtree.radius_search(&pc[idx].position, radius);
        for (nidx, _) in neighbors {
            if !visited[nidx] && is_boundary[nidx] {
                visited[nidx] = true;
                queue.push_back(nidx);
            }
        }
    }

    cluster
}

pub fn assess_quality(
    pc: &PointCloud,
    params: &QualityAssessmentParams,
    weights: &QualityWeights,
) -> Result<QualityReport> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let aabb = AABB::from_points(&pc.points).unwrap();
    let bbox_diag = aabb.diagonal();

    let has_normals = pc.has_normals();
    let do_completeness = params.assess_completeness;

    let effective_weights = if has_normals && do_completeness {
        *weights
    } else {
        let mut w = *weights;
        let mut redistributed = 0.0f64;
        let mut active = 0usize;

        if !has_normals {
            redistributed += w.normal;
            w.normal = 0.0;
        } else {
            active += 1;
        }
        if !do_completeness {
            redistributed += w.completeness;
            w.completeness = 0.0;
        } else {
            active += 1;
        }

        active += 3;
        if active > 0 && redistributed > 0.0 {
            let per_active = redistributed / active as f64;
            w.density += per_active;
            w.overlap += per_active;
            w.noise += per_active;
            if has_normals { w.normal += per_active; }
            if do_completeness { w.completeness += per_active; }
        }
        w
    };

    log::info!("评估密度均匀性...");
    let density = assess_density_uniformity(pc, params)?;
    log::info!("  得分: {:.1}, CV: {:.4}", density.score, density.cv);

    let normal = if has_normals {
        log::info!("评估法向量一致性...");
        let res = assess_normal_consistency(pc, params)?;
        log::info!("  得分: {:.1}, 翻转率: {:.4}%", res.score, res.flip_rate * 100.0);
        res
    } else {
        log::info!("跳过法向量一致性评估(无法向量数据)...");
        NormalMetricResult {
            score: 100.0,
            flip_rate: 0.0,
            total_pairs: 0,
            flipped_pairs: 0,
            mean_angle_deg: 0.0,
        }
    };

    log::info!("评估重叠区域...");
    let overlap = assess_overlap(pc, params)?;
    log::info!("  得分: {:.1}, 重叠率: {:.4}%", overlap.score, overlap.overlap_rate * 100.0);

    log::info!("评估噪声水平...");
    let noise = assess_noise_level(pc, params)?;
    log::info!("  得分: {:.1}, 归一化噪声: {:.6}", noise.score, noise.normalized_noise);

    let completeness = if do_completeness {
        log::info!("评估完整性...");
        let res = assess_completeness(pc, params)?;
        log::info!("  得分: {:.1}, 大洞数: {}", res.score, res.large_holes);
        res
    } else {
        log::info!("跳过完整性评估(未启用)...");
        CompletenessMetricResult {
            score: 100.0,
            hole_count: 0,
            total_boundary_length: 0.0,
            large_holes: 0,
            boundary_threshold: 0.0,
            assessed: false,
        }
    };

    let overall_score =
        density.score * effective_weights.density +
        normal.score * effective_weights.normal +
        overlap.score * effective_weights.overlap +
        noise.score * effective_weights.noise +
        completeness.score * effective_weights.completeness;

    Ok(QualityReport {
        overall_score,
        density,
        normal,
        overlap,
        noise,
        completeness,
        weights: effective_weights,
        total_points: pc.len(),
        has_normals,
        bounding_box_diagonal: bbox_diag,
    })
}

fn score_color(score: f64) -> &'static str {
    if score >= 70.0 {
        "\x1b[32m"
    } else if score >= 40.0 {
        "\x1b[33m"
    } else {
        "\x1b[31m"
    }
}

fn score_label(score: f64) -> &'static str {
    if score >= 70.0 {
        "好"
    } else if score >= 40.0 {
        "中"
    } else {
        "差"
    }
}

pub fn print_quality_report(report: &QualityReport) {
    let reset = "\x1b[0m";
    let bold_code = "\x1b[1m";
    let cyan = "\x1b[36m";

    println!();
    println!("{}", bold(&format!("{}  点云质量评估报告  {}", "=".repeat(25), "=".repeat(25))));
    println!();
    println!("  {}总点数:{} {}", cyan, reset, report.total_points);
    println!("  {}包围盒对角线:{} {:.4}", cyan, reset, report.bounding_box_diagonal);
    println!("  {}包含法向量:{} {}", cyan, reset, if report.has_normals { "是" } else { "否" });
    println!();

    let overall_color = score_color(report.overall_score);
    println!("  {}综合评分: {}{}{:.1}{} / 100  ({})",
        bold_code,
        overall_color,
        bold_code,
        report.overall_score,
        reset,
        score_label(report.overall_score)
    );
    println!();

    println!("  {}", bold("各项得分:"));
    println!("  {:-<60}", "");

    let d_color = score_color(report.density.score);
    println!("  {:<20} {}{:>6.1}{}  [{:3.0}%]  (CV={:.4})",
        "密度均匀性",
        d_color, report.density.score, reset,
        report.weights.density * 100.0,
        report.density.cv
    );

    let n_color = score_color(report.normal.score);
    if report.weights.normal > 0.0 {
        println!("  {:<20} {}{:>6.1}{}  [{:3.0}%]  (翻转率={:.2}%)",
            "法向量一致性",
            n_color, report.normal.score, reset,
            report.weights.normal * 100.0,
            report.normal.flip_rate * 100.0
        );
    } else {
        println!("  {:<20} {:>8}  [{:3.0}%]  (无{}法向量{}跳过)",
            "法向量一致性",
            "N/A",
            0.0,
            cyan, reset
        );
    }

    let o_color = score_color(report.overlap.score);
    println!("  {:<20} {}{:>6.1}{}  [{:3.0}%]  (重叠率={:.3}%)",
        "重叠区域检测",
        o_color, report.overlap.score, reset,
        report.weights.overlap * 100.0,
        report.overlap.overlap_rate * 100.0
    );

    let ns_color = score_color(report.noise.score);
    println!("  {:<20} {}{:>6.1}{}  [{:3.0}%]  (归一化噪声={:.6})",
        "噪声水平估计",
        ns_color, report.noise.score, reset,
        report.weights.noise * 100.0,
        report.noise.normalized_noise
    );

    let c_color = score_color(report.completeness.score);
    if report.weights.completeness > 0.0 {
        let comp_note = format!("大洞数={}", report.completeness.large_holes);
        println!("  {:<20} {}{:>6.1}{}  [{:3.0}%]  ({})",
            "完整性评估",
            c_color, report.completeness.score, reset,
            report.weights.completeness * 100.0,
            comp_note
        );
    } else {
        println!("  {:<20} {:>8}  [{:3.0}%]  (未{}启用{}跳过)",
            "完整性评估",
            "N/A",
            0.0,
            cyan, reset
        );
    }

    println!("  {:-<60}", "");
    println!();
    println!("  颜色图例: {}好{}  {}中{}  {}差{}",
        "\x1b[32m", reset,
        "\x1b[33m", reset,
        "\x1b[31m", reset
    );
    println!("{}", "=".repeat(66));
    println!();
}

pub fn quality_report_to_json(report: &QualityReport) -> Result<String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| PointCloudError::JsonError(e))
}

pub struct RepairParams {
    pub fix_density: bool,
    pub fix_normals: bool,
    pub fix_overlap: bool,
    pub fix_noise: bool,
    pub density_target_cv: f64,
    pub noise_iterations: usize,
    pub noise_sigma_s: f64,
    pub noise_sigma_n: f64,
    pub overlap_threshold_factor: f64,
}

impl Default for RepairParams {
    fn default() -> Self {
        RepairParams {
            fix_density: true,
            fix_normals: true,
            fix_overlap: true,
            fix_noise: true,
            density_target_cv: 0.3,
            noise_iterations: 3,
            noise_sigma_s: 0.02,
            noise_sigma_n: 0.5,
            overlap_threshold_factor: 0.1,
        }
    }
}

pub struct RepairResult {
    pub point_cloud: PointCloud,
    pub points_added: usize,
    pub points_removed: usize,
    pub normals_fixed: usize,
    pub iterations: usize,
}

pub fn auto_repair(
    pc: &PointCloud,
    quality_report: &QualityReport,
    params: &RepairParams,
) -> Result<RepairResult> {
    let mut pc = pc.clone();
    let initial_count = pc.len();
    let original_pc = pc.clone();
    let mut total_added = 0usize;
    let mut total_removed = 0usize;
    let mut normals_fixed = 0usize;
    let actual_iterations: usize;

    let initial_bbox = AABB::from_points(&pc.points)
        .map(|a| a.diagonal())
        .unwrap_or(1.0);

    let net_removed_limit = (initial_count as f64 * 0.22).floor() as usize;
    let reserved_for_final = (initial_count as f64 * 0.06).floor() as usize;
    fn net_cur(removed: usize, added: usize) -> usize { removed.saturating_sub(added) }

    let base_kdtree = KdTree::from_point_cloud(&pc);
    let base_avg_spacing = estimate_average_spacing(&pc, &base_kdtree, 6);
    let _base_overlap_threshold = (base_avg_spacing * params.overlap_threshold_factor).max(1e-8);

    let assess_noise = |cloud: &PointCloud| -> f64 {
        let k = 15.min(cloud.len().saturating_sub(1));
        if k < 3 { return 0.0; }
        let kd = KdTree::from_point_cloud(cloud);
        let mut sum_sq = 0.0f64;
        let mut cnt = 0usize;
        let sample = cloud.len().min(250);
        let step = (cloud.len() / sample.max(1)).max(1);
        for i in (0..cloud.len()).step_by(step).take(sample) {
            let p = &cloud[i];
            let neighbors = kd.knn(&p.position, k + 1);
            if neighbors.len() < 4 { continue; }
            let pts: Vec<nalgebra::Point3<f64>> = neighbors
                .iter()
                .take(k + 1)
                .map(|(idx, _)| cloud[*idx].position)
                .collect();
            let r = point_to_plane_residual_fast(&p.position, &pts);
            sum_sq += r * r;
            cnt += 1;
        }
        if cnt == 0 { return quality_report.noise.normalized_noise; }
        let rms = (sum_sq / cnt as f64).sqrt();
        let bbox = AABB::from_points(&cloud.points).map(|a| a.diagonal()).unwrap_or(1.0).max(1e-10);
        rms / bbox
    };

    if params.fix_noise && quality_report.noise.score < 70.0 {
        log::info!("执行自适应双边滤波去噪...");

        let noise_level = quality_report.noise.normalized_noise;
        let sigma_s_base = if noise_level > 0.005 {
            base_avg_spacing * 1.5
        } else if noise_level > 0.002 {
            base_avg_spacing * 1.0
        } else {
            base_avg_spacing * 0.7
        };
        let sigma_s = sigma_s_base.max(1e-6).min(initial_bbox * 0.01);
        let mut sigma_n = params.noise_sigma_n;

        let max_iters = if quality_report.noise.score < 30.0 {
            params.noise_iterations.min(3)
        } else if quality_report.noise.score < 50.0 {
            params.noise_iterations.min(2)
        } else {
            1
        };

        let mut best_noise = noise_level;
        let mut best_pc = pc.clone();
        let mut best_iter = 0usize;
        let mut iter_count = 0usize;

        for iter in 0..max_iters {
            let prev_bbox = AABB::from_points(&pc.points)
                .map(|a| a.diagonal())
                .unwrap_or(initial_bbox);

            let candidate = bilateral_filter(&pc, sigma_s, sigma_n)?;

            let new_bbox = AABB::from_points(&candidate.points)
                .map(|a| a.diagonal())
                .unwrap_or(0.0);

            if new_bbox < prev_bbox * 0.65 {
                log::warn!("  第 {}/{} 次滤波包围盒收缩过多,终止滤波",
                    iter + 1, max_iters);
                break;
            }

            let new_noise = assess_noise(&candidate);
            iter_count = iter + 1;

            if new_noise < best_noise * 0.99 {
                best_noise = new_noise;
                best_pc = candidate.clone();
                best_iter = iter_count;
                log::info!("  第 {}/{} 次滤波:噪声 {:.6}→{:.6} ✓保留",
                    iter_count, max_iters, noise_level, new_noise);
            } else {
                log::info!("  第 {}/{} 次滤波:噪声 {:.6}→{:.6} 无改善,跳过",
                    iter_count, max_iters, noise_level, new_noise);
            }

            pc = candidate;

            if params.fix_overlap && iter_count < max_iters {
                let budget_remaining = net_removed_limit.saturating_sub(reserved_for_final);
                if net_cur(total_removed, total_added) < budget_remaining {
                    let cur_kdtree = KdTree::from_point_cloud(&pc);
                    let cur_spacing = estimate_average_spacing(&pc, &cur_kdtree, 6);
                    let cleanup_thresh = (cur_spacing * params.overlap_threshold_factor * 0.9).max(1e-8);
                    let before = pc.len();
                    let cleaned = remove_overlapping_points(&pc, cleanup_thresh)?;
                    let n_removed = before - cleaned.len();
                    if n_removed > 0 {
                        let allowed = budget_remaining.saturating_sub(net_cur(total_removed, total_added));
                        if n_removed <= allowed {
                            total_removed += n_removed;
                            pc = cleaned;
                            log::info!("  滤波后清理 {} 个重叠点", n_removed);
                        }
                    }
                }
            }

            sigma_n *= 0.85;
        }

        if best_noise < noise_level * 0.99 {
            pc = best_pc;
            actual_iterations = best_iter;
            log::info!("采用最佳迭代(第{}次),噪声 {:.6}→{:.6}",
                best_iter, noise_level, best_noise);
        } else {
            pc = original_pc.clone();
            total_removed = 0;
            actual_iterations = 0;
            log::warn!("所有滤波迭代均未改善噪声,回滚滤波操作");
        }
    } else {
        actual_iterations = 0;
    }

    if params.fix_normals && quality_report.normal.score < 70.0 && pc.has_normals() {
        log::info!("修复法向量方向...");
        let result = fix_normal_orientations(&pc)?;
        normals_fixed = result.flipped_count;
        pc = result.point_cloud;
        log::info!("  修复 {} 个法向量方向", normals_fixed);
    }

    if params.fix_overlap && quality_report.overlap.score < 70.0 {
        let budget_remaining = net_removed_limit.saturating_sub(reserved_for_final);
        if net_cur(total_removed, total_added) < budget_remaining {
            log::info!("修复重叠点...");
            let cur_kdtree = KdTree::from_point_cloud(&pc);
            let cur_spacing = estimate_average_spacing(&pc, &cur_kdtree, 6);
            let overlap_thresh = (cur_spacing * params.overlap_threshold_factor).max(1e-8);

            let before = pc.len();
            let result = remove_overlapping_points(&pc, overlap_thresh)?;
            let removed_now = before - result.len();

            let allowed = budget_remaining.saturating_sub(net_cur(total_removed, total_added));
            if removed_now <= allowed {
                total_removed += removed_now;
                pc = result;
                log::info!("  去除 {} 个重叠点", removed_now);
            } else {
                log::warn!("  重叠去除({}点)超限(剩余额度{}点),执行受限清理",
                    removed_now, allowed);
                let partial_pc = partial_overlap_removal(&pc, overlap_thresh, allowed)?;
                let actually_removed = before - partial_pc.len();
                total_removed += actually_removed;
                pc = partial_pc;
                log::info!("  受限去除 {} 个重叠点", actually_removed);
            }
        } else {
            log::warn!("删除额度已用于前期步骤,跳过主重叠修复以保留最终清理额度");
        }
    }

    let removed_so_far = total_removed;
    if params.fix_density && quality_report.density.score < 70.0 {
        log::info!("修复密度不均匀性...");
        let target_extra = (removed_so_far as f64 * 1.15).round() as usize;
        let result = upsample_sparse_regions(&pc, params.density_target_cv, target_extra)?;
        let added_now = result.len().saturating_sub(pc.len());
        total_added += added_now;
        pc = PointCloud::from_points(result);
        log::info!("  添加 {} 个上采样点", added_now);
    }

    if params.fix_overlap && (total_added > 0 || actual_iterations > 0 || quality_report.overlap.score < 70.0) {
        log::info!("执行最终重叠清理(优先保障)...");
        let cur_kdtree = KdTree::from_point_cloud(&pc);
        let cur_spacing = estimate_average_spacing(&pc, &cur_kdtree, 6);
        let final_thresh = (cur_spacing * params.overlap_threshold_factor).max(1e-8);
        let before_final = pc.len();
        let cleaned = remove_overlapping_points(&pc, final_thresh)?;
        let n_removed = before_final - cleaned.len();

        let final_budget = (net_removed_limit + total_added).saturating_sub(total_removed);
        if n_removed <= final_budget {
            total_removed += n_removed;
            pc = cleaned;
            log::info!("  最终清理 {} 个新增重叠点", n_removed);
        } else {
            log::warn!("  最终清理预计{}点超过剩余额度{}点,执行受限清理", n_removed, final_budget);
            let partial_pc = partial_overlap_removal(&pc, final_thresh, final_budget)?;
            let actually_removed = before_final - partial_pc.len();
            total_removed += actually_removed;
            pc = partial_pc;
            log::info!("  受限最终清理 {} 个重叠点", actually_removed);
        }
    }

    let final_count = pc.len();
    let net_removed = total_removed.saturating_sub(total_added);
    let net_ratio = if initial_count > 0 {
        net_removed as f64 / initial_count as f64
    } else {
        0.0
    };

    if net_ratio > 0.27 && initial_count > 0 {
        log::error!("修复后净点数减少过多({:.1}%),超出安全阈值,回滚到原始点云",
            net_ratio * 100.0);
        return Ok(RepairResult {
            point_cloud: original_pc,
            points_added: 0,
            points_removed: 0,
            normals_fixed: 0,
            iterations: 0,
        });
    } else if net_ratio > 0.15 {
        log::warn!("修复后净点数减少 {:.1}%,请确认修复效果是否可接受", net_ratio * 100.0);
    }

    let final_bbox = AABB::from_points(&pc.points)
        .map(|a| a.diagonal())
        .unwrap_or(0.0);
    if final_bbox < initial_bbox * 0.15 && initial_bbox > 1e-6 {
        log::warn!("修复后包围盒严重收缩({:.4} -> {:.4}),可能存在数据质量问题",
            initial_bbox, final_bbox);
    }

    // 整体质量回滚：如果修复后重叠率比修前更差，回滚
    if quality_report.overlap.score < 70.0 {
        let params_assess = QualityAssessmentParams::default();
        if let Ok(new_overlap) = assess_overlap(&pc, &params_assess) {
            if new_overlap.overlap_rate > quality_report.overlap.overlap_rate * 1.05 {
                log::error!("修复后重叠率({:.2}%)比修前({:.2}%)更差,回滚到原始点云",
                    new_overlap.overlap_rate * 100.0,
                    quality_report.overlap.overlap_rate * 100.0);
                return Ok(RepairResult {
                    point_cloud: original_pc,
                    points_added: 0,
                    points_removed: 0,
                    normals_fixed: 0,
                    iterations: 0,
                });
            }
        }
    }

    let _ = final_count;
    Ok(RepairResult {
        point_cloud: pc,
        points_added: total_added,
        points_removed: total_removed,
        normals_fixed,
        iterations: actual_iterations,
    })
}

fn point_to_plane_residual_fast(
    point: &nalgebra::Point3<f64>,
    neighbors: &[nalgebra::Point3<f64>],
) -> f64 {
    let n = neighbors.len();
    if n < 3 { return 0.0; }
    let mut centroid = nalgebra::Point3::origin();
    for p in neighbors { centroid.coords += p.coords; }
    centroid.coords /= n as f64;
    let mut cov = nalgebra::Matrix3::zeros();
    for p in neighbors {
        let d = p.coords - centroid.coords;
        cov += d * d.transpose();
    }
    let eigen = nalgebra::SymmetricEigen::new(cov);
    let mut eig_pairs: Vec<(f64, nalgebra::Vector3<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(v, col)| (*v, col.clone_owned()))
        .collect();
    eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let normal = eig_pairs[0].1.normalize();
    let d = -normal.dot(&centroid.coords);
    (normal.dot(&point.coords) + d).abs()
}

fn partial_overlap_removal(
    pc: &PointCloud,
    threshold: f64,
    max_remove: usize,
) -> Result<PointCloud> {
    if pc.is_empty() {
        return Ok(PointCloud::new());
    }
    if max_remove == 0 {
        return Ok(pc.clone());
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let mut removed = vec![false; pc.len()];
    let mut remove_count = 0usize;

    for i in 0..pc.len() {
        if removed[i] { continue; }

        let neighbors = kdtree.radius_search(&pc[i].position, threshold);

        for (idx, _) in &neighbors {
            if *idx != i && !removed[*idx] {
                if remove_count >= max_remove {
                    break;
                }
                removed[*idx] = true;
                remove_count += 1;
            }
        }

        if remove_count >= max_remove {
            break;
        }
    }

    let keep_count = pc.len() - remove_count;
    let mut result = Vec::with_capacity(keep_count);
    for i in 0..pc.len() {
        if !removed[i] {
            result.push(pc[i].clone());
        }
    }

    Ok(PointCloud::from_points(result))
}

fn remove_overlapping_points(pc: &PointCloud, threshold: f64) -> Result<PointCloud> {
    if pc.is_empty() {
        return Ok(PointCloud::new());
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let mut removed = vec![false; pc.len()];
    let mut keep_count = 0usize;

    for i in 0..pc.len() {
        if removed[i] { continue; }

        keep_count += 1;
        let neighbors = kdtree.radius_search(&pc[i].position, threshold);

        for (idx, _) in &neighbors {
            if *idx != i {
                removed[*idx] = true;
            }
        }
    }

    let mut result = Vec::with_capacity(keep_count);
    for i in 0..pc.len() {
        if !removed[i] {
            result.push(pc[i].clone());
        }
    }

    Ok(PointCloud::from_points(result))
}

fn upsample_sparse_regions(pc: &PointCloud, _target_cv: f64, target_extra_points: usize) -> Result<Vec<Point3D>> {
    if pc.is_empty() {
        return Ok(Vec::new());
    }

    let params = QualityAssessmentParams::default();
    let octree = build_octree(pc, params.octree_max_depth, params.octree_min_points);
    let mut leaves: Vec<&OctreeNode> = Vec::new();
    collect_leaf_nodes(&octree, &mut leaves);

    if leaves.is_empty() {
        return Ok(pc.points.clone());
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let avg_spacing = estimate_average_spacing(pc, &kdtree, 6);
    let min_dist_sq = (avg_spacing * params.overlap_threshold_factor * 1.2).max(1e-8);
    let min_dist_sq = min_dist_sq * min_dist_sq;

    let counts: Vec<usize> = leaves.iter().map(|l| l.point_indices.len()).collect();
    let counts_f64: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let mean_pts = mean(&counts_f64);

    let mut new_points = pc.points.clone();

    fn point_too_close(
        existing: &[Point3D],
        kdtree: &KdTree,
        candidate: &nalgebra::Point3<f64>,
        min_dist_sq: f64,
        extra_end: usize,
    ) -> bool {
        let search_thresh = (min_dist_sq).sqrt() * 1.01;
        let neighbors = kdtree.radius_search(candidate, search_thresh);
        for (idx, _) in &neighbors {
            if *idx < extra_end {
                let dx = existing[*idx].position.x - candidate.x;
                let dy = existing[*idx].position.y - candidate.y;
                let dz = existing[*idx].position.z - candidate.z;
                if dx*dx + dy*dy + dz*dz < min_dist_sq {
                    return true;
                }
            }
        }
        false
    }

    let mut try_add = |
        new_points: &mut Vec<Point3D>,
        candidate_pos: nalgebra::Point3<f64>,
        template_i: &Point3D,
        template_j: &Point3D,
        t: f64,
        added: &mut usize,
        points_to_add: usize,
        min_dist_sq: f64,
    | -> bool {
        if *added >= points_to_add { return false; }
        let extra_end = new_points.len();

        if point_too_close(new_points, &kdtree, &candidate_pos, min_dist_sq, extra_end) {
            return false;
        }
        for k in pc.len()..extra_end {
            let dx = new_points[k].position.x - candidate_pos.x;
            let dy = new_points[k].position.y - candidate_pos.y;
            let dz = new_points[k].position.z - candidate_pos.z;
            if dx*dx + dy*dy + dz*dz < min_dist_sq {
                return false;
            }
        }

        let color = match (template_i.color, template_j.color) {
            (Some(c1), Some(c2)) => Some(Color::new(
                (((c1.r as f64) * (1.0 - t) + (c2.r as f64) * t).min(255.0)) as u8,
                (((c1.g as f64) * (1.0 - t) + (c2.g as f64) * t).min(255.0)) as u8,
                (((c1.b as f64) * (1.0 - t) + (c2.b as f64) * t).min(255.0)) as u8,
            )),
            _ => None,
        };
        let normal = match (template_i.normal, template_j.normal) {
            (Some(n1), Some(n2)) => {
                let n = (n1 * (1.0 - t) + n2 * t).normalize();
                if n.norm() > 1e-15 { Some(n) } else { None }
            },
            _ => None,
        };
        new_points.push(Point3D {
            position: candidate_pos,
            color,
            normal,
            intensity: None,
            curvature: None,
        });
        *added += 1;
        true
    };

    for leaf in &leaves {
        let n_points = leaf.point_indices.len();
        if n_points == 0 { continue; }

        let density_ratio = n_points as f64 / mean_pts.max(1e-10);
        if density_ratio >= 0.8 {
            continue;
        }

        let target_points = (mean_pts * 0.95) as usize;
        let points_to_add = target_points.saturating_sub(n_points).max(1);

        if leaf.point_indices.len() < 2 {
            continue;
        }

        let mut added = 0usize;
        let indices = &leaf.point_indices;

        for i in 0..indices.len() {
            if added >= points_to_add { break; }
            for j in i + 1..indices.len() {
                if added >= points_to_add { break; }

                let pi = &pc[indices[i]];
                let pj = &pc[indices[j]];

                for &t in &[0.5f64, 0.33, 0.67] {
                    if added >= points_to_add { break; }
                    let cpos = nalgebra::Point3::new(
                        pi.position.x * (1.0 - t) + pj.position.x * t,
                        pi.position.y * (1.0 - t) + pj.position.y * t,
                        pi.position.z * (1.0 - t) + pj.position.z * t,
                    );
                    try_add(&mut new_points, cpos, pi, pj, t, &mut added, points_to_add, min_dist_sq);
                }
            }
        }
    }

    if target_extra_points > 0 {
        let current_added = new_points.len() - pc.len();
        let need_more = target_extra_points.saturating_sub(current_added);
        if need_more > 0 {
            let mut rng_added = 0usize;
            let sorted_leaves = {
                let mut tmp: Vec<(usize, &OctreeNode)> = leaves
                    .iter()
                    .map(|l| (l.point_indices.len(), *l))
                    .filter(|(n, _)| *n >= 2)
                    .collect();
                tmp.sort_by(|a, b| a.0.cmp(&b.0));
                tmp
            };

            let mut leaf_idx = 0usize;
            while rng_added < need_more && leaf_idx < sorted_leaves.len() {
                let (_, leaf) = sorted_leaves[leaf_idx];
                let indices = &leaf.point_indices;
                if indices.len() < 2 {
                    leaf_idx += 1;
                    continue;
                }

                let mut added_this_leaf = 0usize;
                let max_per_leaf = (need_more / sorted_leaves.len().max(1)).max(2).min(40);

                'outer: for i in 0..indices.len() {
                    for j in i + 1..indices.len() {
                        if rng_added >= need_more || added_this_leaf >= max_per_leaf {
                            break 'outer;
                        }
                        let pi = &pc[indices[i]];
                        let pj = &pc[indices[j]];
                        for &t in &[0.2f64, 0.4, 0.6, 0.8, 0.1, 0.9] {
                            if rng_added >= need_more || added_this_leaf >= max_per_leaf {
                                break 'outer;
                            }
                            let cpos = nalgebra::Point3::new(
                                pi.position.x * (1.0 - t) + pj.position.x * t,
                                pi.position.y * (1.0 - t) + pj.position.y * t,
                                pi.position.z * (1.0 - t) + pj.position.z * t,
                            );
                            let mut a = added_this_leaf;
                            if try_add(&mut new_points, cpos, pi, pj, t, &mut a, max_per_leaf, min_dist_sq) {
                                rng_added += 1;
                                added_this_leaf = a;
                            }
                        }
                    }
                }
                leaf_idx += 1;
            }
        }
    }

    Ok(new_points)
}

struct NormalFixResult {
    point_cloud: PointCloud,
    flipped_count: usize,
}

fn fix_normal_orientations(pc: &PointCloud) -> Result<NormalFixResult> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }
    if !pc.has_normals() {
        return Err(PointCloudError::NormalsNotComputed);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let k = 10.min(pc.len().saturating_sub(1));
    let mut result = pc.points.clone();
    let mut flipped_count = 0usize;

    let mut visited = vec![false; pc.len()];

    let threshold_dot: f64 = 0.0_f64.to_radians().cos();

    for start in 0..pc.len() {
        if visited[start] || result[start].normal.is_none() {
            continue;
        }

        {
            let seed_n = result[start].normal.unwrap();
            let view_dir = nalgebra::Vector3::new(0.0, 0.0, 1.0);
            if seed_n.dot(&view_dir) < -0.1 {
                result[start].normal = Some(-seed_n);
                flipped_count += 1;
            }
        }

        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(idx) = queue.pop_front() {
            let current_normal = match result[idx].normal {
                Some(n) => n,
                None => continue,
            };

            let neighbors = kdtree.knn(&pc[idx].position, k + 1);

            for (nidx, _) in neighbors.iter().skip(1) {
                let nidx = *nidx;
                if visited[nidx] { continue; }
                if result[nidx].normal.is_none() { continue; }

                visited[nidx] = true;

                if let Some(neighbor_normal) = result[nidx].normal {
                    let dot = current_normal.dot(&neighbor_normal);
                    if dot < threshold_dot {
                        result[nidx].normal = Some(-neighbor_normal);
                        flipped_count += 1;
                    }
                }

                queue.push_back(nidx);
            }
        }
    }

    let post_flip = assess_flip_rate_post(&result, &kdtree, k.min(8));
    let pre_flip = assess_flip_rate_pre(pc, &kdtree, k.min(8));

    let score_line = 0.08f64;
    let improved = post_flip < pre_flip * 0.90 || post_flip < pre_flip - 0.01;
    let bad_absolute = post_flip > score_line;
    let worse = post_flip > pre_flip * 1.05;

    if !improved || (bad_absolute && worse) {
        log::warn!(
            "法向量修复效果不佳(前:{:.2}% → 后:{:.2}%),回滚到原始法向量",
            pre_flip * 100.0, post_flip * 100.0
        );
        return Ok(NormalFixResult {
            point_cloud: pc.clone(),
            flipped_count: 0,
        });
    }

    Ok(NormalFixResult {
        point_cloud: PointCloud::from_points(result),
        flipped_count,
    })
}

fn assess_flip_rate_pre(pc: &PointCloud, kdtree: &KdTree, k: usize) -> f64 {
    let sample = pc.len().min(500);
    let step = (pc.len() / sample.max(1)).max(1);
    let mut total_pairs = 0usize;
    let mut flipped_pairs = 0usize;

    for i in (0..pc.len()).step_by(step).take(sample) {
        let Some(n_i) = pc[i].normal else { continue };
        let neighbors = kdtree.knn(&pc[i].position, k + 1);
        for (nidx, _) in neighbors.iter().skip(1) {
            let Some(n_j) = pc[*nidx].normal else { continue };
            total_pairs += 1;
            if n_i.dot(&n_j) < 0.0 {
                flipped_pairs += 1;
            }
        }
    }

    if total_pairs == 0 { 0.0 } else { flipped_pairs as f64 / total_pairs as f64 }
}

fn assess_flip_rate_post(points: &[Point3D], kdtree: &KdTree, k: usize) -> f64 {
    let sample = points.len().min(500);
    let step = (points.len() / sample.max(1)).max(1);
    let mut total_pairs = 0usize;
    let mut flipped_pairs = 0usize;

    for i in (0..points.len()).step_by(step).take(sample) {
        let Some(n_i) = points[i].normal else { continue };
        let neighbors = kdtree.knn(&points[i].position, k + 1);
        for (nidx, _) in neighbors.iter().skip(1) {
            let Some(n_j) = points[*nidx].normal else { continue };
            total_pairs += 1;
            if n_i.dot(&n_j) < 0.0 {
                flipped_pairs += 1;
            }
        }
    }

    if total_pairs == 0 { 0.0 } else { flipped_pairs as f64 / total_pairs as f64 }
}

fn bilateral_filter(pc: &PointCloud, sigma_s: f64, sigma_n: f64) -> Result<PointCloud> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let radius = (sigma_s * 3.0).max(1e-6);
    let has_normals = pc.has_normals();
    let two_ss = 2.0 * sigma_s * sigma_s;
    let two_nn = 2.0 * sigma_n * sigma_n;

    let new_points: Vec<Point3D> = pc
        .points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let neighbors = kdtree.radius_search(&p.position, radius);
            if neighbors.len() < 4 {
                return p.clone();
            }

            let p_normal = p.normal;

            let neighbor_points: Vec<nalgebra::Point3<f64>> = neighbors
                .iter()
                .map(|(idx, _)| pc[*idx].position)
                .collect();

            let n = neighbor_points.len() as f64;
            let mut centroid = nalgebra::Point3::origin();
            for np in &neighbor_points {
                centroid.coords += np.coords;
            }
            centroid.coords /= n;

            let mut cov = nalgebra::Matrix3::zeros();
            for np in &neighbor_points {
                let d = np.coords - centroid.coords;
                cov += d * d.transpose();
            }

            let eigen = nalgebra::SymmetricEigen::new(cov);
            let mut eig_pairs: Vec<(f64, nalgebra::Vector3<f64>)> = eigen
                .eigenvalues
                .iter()
                .zip(eigen.eigenvectors.column_iter())
                .map(|(v, col)| (*v, col.clone_owned()))
                .collect();
            eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let plane_normal = eig_pairs[0].1.normalize();
            let plane_d = -plane_normal.dot(&centroid.coords);
            let dist_to_plane = (plane_normal.dot(&p.position.coords) + plane_d).abs();

            let projected = p.position.coords - dist_to_plane * plane_normal
                * (plane_normal.dot(&p.position.coords) + plane_d).signum();

            let mut sum_pos = nalgebra::Point3::origin();
            let mut sum_weight = 0.0f64;
            let mut sum_normal = nalgebra::Vector3::zeros();
            let mut sum_normal_weight = 0.0f64;
            let mut has_color = true;
            let mut sum_r = 0.0f64;
            let mut sum_g = 0.0f64;
            let mut sum_b = 0.0f64;
            let mut sum_color_weight = 0.0f64;

            for (idx, dist) in &neighbors {
                let np = &pc[*idx];
                let np_dist_to_plane = (plane_normal.dot(&np.position.coords) + plane_d).abs();
                let np_projected = np.position.coords - np_dist_to_plane * plane_normal
                    * (plane_normal.dot(&np.position.coords) + plane_d).signum();

                let spatial_weight = if *dist <= 1e-12 {
                    1.0
                } else {
                    (-dist * dist / two_ss).exp()
                };

                let normal_weight = if has_normals {
                    if let (Some(n_i), Some(n_j)) = (p.normal, np.normal) {
                        let dot = n_i.dot(&n_j).clamp(-1.0, 1.0);
                        let angle_diff = (1.0 - dot).max(0.0);
                        (-angle_diff * angle_diff / two_nn).exp().max(0.25)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                let plane_residual_weight = if np_dist_to_plane <= 1e-12 {
                    1.0
                } else {
                    (-np_dist_to_plane * np_dist_to_plane
                        / (two_ss * 0.5)).exp()
                };

                let weight = spatial_weight * normal_weight * plane_residual_weight;

                sum_pos.coords += np_projected * weight;
                sum_weight += weight;

                if let Some(n) = np.normal {
                    sum_normal += n * weight;
                    sum_normal_weight += weight;
                }

                if let Some(c) = np.color {
                    sum_r += c.r as f64 * weight;
                    sum_g += c.g as f64 * weight;
                    sum_b += c.b as f64 * weight;
                    sum_color_weight += weight;
                } else {
                    has_color = false;
                }
            }

            let self_w = 1.5;
            sum_pos.coords += projected * self_w;
            sum_weight += self_w;
            if let Some(n) = p.normal {
                sum_normal += n * self_w;
                sum_normal_weight += self_w;
            }
            if let Some(c) = p.color {
                sum_r += c.r as f64 * self_w;
                sum_g += c.g as f64 * self_w;
                sum_b += c.b as f64 * self_w;
                sum_color_weight += self_w;
            }

            let mut new_p = if sum_weight > 1e-12 {
                Point3D::new(
                    sum_pos.x / sum_weight,
                    sum_pos.y / sum_weight,
                    sum_pos.z / sum_weight,
                )
            } else {
                p.clone()
            };

            if has_normals && sum_normal_weight > 1e-12 {
                let n_len = sum_normal.norm();
                if n_len > 1e-15 {
                    new_p.normal = Some(sum_normal / n_len);
                }
            }

            if has_color && sum_color_weight > 1e-12 {
                new_p.color = Some(Color::new(
                    (sum_r / sum_color_weight).min(255.0) as u8,
                    (sum_g / sum_color_weight).min(255.0) as u8,
                    (sum_b / sum_color_weight).min(255.0) as u8,
                ));
            }

            let _ = p_normal;
            new_p
        })
        .collect();

    Ok(PointCloud::from_points(new_points))
}

pub fn bold(s: &str) -> String {
    format!("\x1b[1m{}\x1b[0m", s)
}
