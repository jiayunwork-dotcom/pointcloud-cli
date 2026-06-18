use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Point3D, KdTree, Transform, AABB};
use nalgebra::{Point3, Vector3, Matrix3, Matrix4, DMatrix};
use rand::seq::SliceRandom;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegistrationParams {
    pub fpfh_radius: f64,
    pub ransac_max_iterations: usize,
    pub ransac_inlier_threshold_factor: f64,
    pub icp_max_iterations: usize,
    pub icp_convergence_threshold: f64,
    pub icp_max_correspondence_distance: f64,
    pub voxel_size: f64,
    pub normal_radius: f64,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        RegistrationParams {
            fpfh_radius: 0.0,
            ransac_max_iterations: 1000,
            ransac_inlier_threshold_factor: 2.0,
            icp_max_iterations: 50,
            icp_convergence_threshold: 1e-7,
            icp_max_correspondence_distance: 1.0,
            voxel_size: 0.0,
            normal_radius: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationResult {
    pub transform: Matrix4<f64>,
    pub rmse: f64,
    pub inlier_count: usize,
    pub iterations: usize,
    pub converged: bool,
    pub coarse_transform: Matrix4<f64>,
    pub coarse_inlier_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentAccuracyReport {
    pub rmse: f64,
    pub max_error: f64,
    pub inlier_ratio: f64,
    pub overlap_rate: f64,
    pub rotation_angle_deg: f64,
    pub translation_distance: f64,
    pub mean_distance: f64,
    pub median_distance: f64,
    pub source_points: usize,
    pub target_points: usize,
    pub corresponding_pairs: usize,
    pub inlier_threshold: f64,
    pub overlap_threshold: f64,
}

pub fn compute_average_point_spacing(pc: &PointCloud, sample_size: usize) -> f64 {
    if pc.len() < 2 {
        return 0.01;
    }
    let n = pc.len().min(sample_size);
    let step = (pc.len() as f64 / n as f64).ceil() as usize;
    let kdtree = KdTree::from_point_cloud(pc);
    let mut sum = 0.0;
    let mut count = 0usize;

    for i in (0..pc.len()).step_by(step.max(1)) {
        if count >= n { break; }
        let neighbors = kdtree.knn(&pc[i].position, 2);
        if neighbors.len() >= 2 {
            sum += neighbors[1].1;
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.01
    }
}

pub fn ensure_normals(pc: &mut PointCloud, normal_radius: f64) -> Result<()> {
    if pc.has_normals() {
        return Ok(());
    }
    use crate::normals::{estimate_normals, NormalEstimationParams};

    let avg_spacing = if normal_radius <= 0.0 {
        compute_average_point_spacing(pc, 500) * 3.0
    } else {
        normal_radius
    };

    let k_estimate = estimate_k_from_radius(pc, avg_spacing);
    let params = NormalEstimationParams {
        k: k_estimate.max(5).min(50),
        orientation_k: 10,
    };

    let result = estimate_normals(pc, &params)?;
    *pc = result.point_cloud;
    Ok(())
}

fn estimate_k_from_radius(pc: &PointCloud, radius: f64) -> usize {
    if pc.len() < 2 {
        return 20;
    }
    let kdtree = KdTree::from_point_cloud(pc);
    let sample = pc.len().min(50);
    let step = (pc.len() / sample.max(1)).max(1);
    let mut total = 0usize;
    let mut count = 0usize;

    for i in (0..pc.len()).step_by(step) {
        if count >= sample { break; }
        let neighbors = kdtree.radius_search(&pc[i].position, radius);
        total += neighbors.len();
        count += 1;
    }

    if count > 0 {
        (total as f64 / count as f64).round() as usize
    } else {
        20
    }
}

pub fn register_point_clouds(
    source: &PointCloud,
    target: &PointCloud,
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    let mut source_work = source.clone();
    let mut target_work = target.clone();

    if params.voxel_size > 0.0 {
        use crate::preprocess::{voxel_downsample, VoxelDownsampleParams};
        let ds_params = VoxelDownsampleParams { voxel_size: params.voxel_size };
        source_work = voxel_downsample(&source_work, &ds_params)?.downsampled;
        target_work = voxel_downsample(&target_work, &ds_params)?.downsampled;
    }

    ensure_normals(&mut source_work, params.normal_radius)?;
    ensure_normals(&mut target_work, params.normal_radius)?;

    let avg_spacing = compute_average_point_spacing(&target_work, 500);
    let fpfh_radius = if params.fpfh_radius <= 0.0 {
        avg_spacing * 10.0
    } else {
        params.fpfh_radius
    };

    let inlier_threshold = avg_spacing * params.ransac_inlier_threshold_factor;
    let mut effective_params = *params;
    if effective_params.icp_max_correspondence_distance <= 0.0
        || effective_params.icp_max_correspondence_distance < inlier_threshold {
        effective_params.icp_max_correspondence_distance = inlier_threshold * 1.5;
    }

    log::info!("平均点间距: {:.6}", avg_spacing);
    log::info!("FPFH半径: {:.6}", fpfh_radius);
    log::info!("RANSAC内点阈值: {:.6}", inlier_threshold);
    log::info!("ICP最大对应距离: {:.6}", effective_params.icp_max_correspondence_distance);

    log::info!("计算源点云FPFH特征 ({}点)...", source_work.len());
    let source_fpfh = compute_fpfh(&source_work, fpfh_radius)?;
    log::info!("计算目标点云FPFH特征 ({}点)...", target_work.len());
    let target_fpfh = compute_fpfh(&target_work, fpfh_radius)?;

    log::info!("执行RANSAC粗配准 (迭代{}次)...", effective_params.ransac_max_iterations);
    let coarse = ransac_registration(
        &source_work, &target_work, &source_fpfh, &target_fpfh,
        &effective_params, inlier_threshold
    )?;
    log::info!("  粗配准完成: 内点数={}, 初始RMSE={:.6}",
        coarse.inlier_count, coarse.rmse);

    log::info!("执行Point-to-Plane ICP精配准 (最大{}次)...", effective_params.icp_max_iterations);
    let icp = point_to_plane_icp(
        &source_work, &target_work, &coarse.transform,
        &effective_params, inlier_threshold
    )?;
    log::info!("  精配准完成: 迭代={}, 收敛={}, 最终RMSE={:.6}",
        icp.iterations, icp.converged, icp.rmse);

    Ok(RegistrationResult {
        transform: icp.transform,
        rmse: icp.rmse,
        inlier_count: icp.inlier_count,
        iterations: icp.iterations,
        converged: icp.converged,
        coarse_transform: coarse.transform,
        coarse_inlier_count: coarse.inlier_count,
    })
}

pub const FPFH_DIM: usize = 33;

pub fn compute_fpfh(pc: &PointCloud, radius: f64) -> Result<Vec<[f64; FPFH_DIM]>> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }
    if !pc.has_normals() {
        return Err(PointCloudError::NormalsNotComputed);
    }

    let kdtree = KdTree::from_point_cloud(pc);
    let n = pc.len();
    let mut spfh: Vec<[f64; FPFH_DIM]> = vec![[0.0; FPFH_DIM]; n];

    for i in 0..n {
        let neighbors = kdtree.radius_search(&pc[i].position, radius);
        if neighbors.len() < 2 { continue; }

        let mut hist = [0.0f64; 33];
        let mut count = 0.0;

        for &(j, _) in &neighbors {
            if i == j { continue; }
            let (alpha, phi, theta) = compute_pfh_features(
                &pc[i], &pc[j]
            );
            let b1 = ((alpha + 1.0) * 0.5 * 11.0).floor().min(10.0) as usize;
            let b2 = ((phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 11.0).floor().min(10.0) as usize;
            let b3 = ((theta + 1.0) * 0.5 * 11.0).floor().min(10.0) as usize;
            let b1 = b1.min(10);
            let b2 = b2.min(10);
            let b3 = b3.min(10);
            hist[b1] += 1.0;
            hist[11 + b2] += 1.0;
            hist[22 + b3] += 1.0;
            count += 1.0;
        }

        if count > 0.0 {
            for k in 0..FPFH_DIM {
                spfh[i][k] = hist[k] / count;
            }
        }
    }

    let mut fpfh: Vec<[f64; FPFH_DIM]> = vec![[0.0; FPFH_DIM]; n];
    for i in 0..n {
        let neighbors = kdtree.radius_search(&pc[i].position, radius);
        if neighbors.is_empty() {
            fpfh[i] = spfh[i];
            continue;
        }

        let mut weights_sum = 0.0;
        for &(j, dist) in &neighbors {
            let w = if dist > 1e-15 { 1.0 / dist } else { 1.0 };
            for k in 0..FPFH_DIM {
                fpfh[i][k] += w * spfh[j][k];
            }
            weights_sum += w;
        }

        if weights_sum > 0.0 {
            for k in 0..FPFH_DIM {
                fpfh[i][k] = spfh[i][k] + fpfh[i][k] / weights_sum;
            }
        }

        let sum = fpfh[i].iter().sum::<f64>().max(1e-15);
        for k in 0..FPFH_DIM {
            fpfh[i][k] /= sum;
        }
    }

    Ok(fpfh)
}

fn compute_pfh_features(pi: &Point3D, pj: &Point3D) -> (f64, f64, f64) {
    let ni = pi.normal.unwrap_or(Vector3::new(0.0, 0.0, 1.0));
    let nj = pj.normal.unwrap_or(Vector3::new(0.0, 0.0, 1.0));

    let u = ni;
    let diff = pj.position - pi.position;
    let diff_norm = diff.norm();
    let v = if diff_norm > 1e-15 {
        u.cross(&diff).normalize()
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let w = u.cross(&v);

    let alpha = v.dot(&nj).clamp(-1.0, 1.0);
    let phi = if diff_norm > 1e-15 {
        u.dot(&diff)
    } else {
        0.0
    };
    let theta = w.dot(&nj).clamp(-1.0, 1.0);

    (alpha, phi, theta)
}

struct SimpleFeatureKdTree {
    nodes: Vec<KdNode>,
    points: Vec<[f64; FPFH_DIM]>,
    indices: Vec<usize>,
}

struct KdNode {
    split_dim: usize,
    split_val: f64,
    left: isize,
    right: isize,
    point_idx: usize,
}

impl SimpleFeatureKdTree {
    fn from_features(f: &[[f64; FPFH_DIM]]) -> Self {
        let n = f.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(n);
        let points = f.to_vec();

        Self::build_recursive(&points, &mut indices, &mut nodes, 0, n);

        SimpleFeatureKdTree { nodes, points, indices }
    }

    fn build_recursive(
        points: &[[f64; FPFH_DIM]],
        indices: &mut [usize],
        nodes: &mut Vec<KdNode>,
        depth: usize,
        total: usize,
    ) -> isize {
        if indices.is_empty() {
            return -1;
        }

        let dim = depth % FPFH_DIM;
        let mid = indices.len() / 2;

        indices.select_nth_unstable_by(mid, |&a, &b| {
            points[a][dim].partial_cmp(&points[b][dim]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let point_idx = indices[mid];
        let split_val = points[point_idx][dim];

        let node_idx = nodes.len();
        nodes.push(KdNode {
            split_dim: dim,
            split_val,
            left: -1,
            right: -1,
            point_idx,
        });

        if indices.len() > 1 {
            let (left_part, right_part) = indices.split_at_mut(mid);
            let right_part = &mut right_part[1..];

            let left_child = Self::build_recursive(
                points, left_part, nodes, depth + 1, total
            );
            let right_child = Self::build_recursive(
                points, right_part, nodes, depth + 1, total
            );

            nodes[node_idx].left = left_child;
            nodes[node_idx].right = right_child;
        }

        node_idx as isize
    }

    fn nearest(&self, query: &[f64; FPFH_DIM]) -> (usize, f64) {
        if self.nodes.is_empty() {
            return (0, f64::INFINITY);
        }

        let mut best_i = self.nodes[0].point_idx;
        let mut best_d = fpfh_distance(query, &self.points[best_i]);

        self.nearest_recursive(query, 0, &mut best_i, &mut best_d);
        (best_i, best_d)
    }

    fn nearest_recursive(
        &self,
        query: &[f64; FPFH_DIM],
        node_idx: isize,
        best_i: &mut usize,
        best_d: &mut f64,
    ) {
        if node_idx < 0 || (node_idx as usize) >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_idx as usize];
        let d = fpfh_distance(query, &self.points[node.point_idx]);

        if d < *best_d {
            *best_d = d;
            *best_i = node.point_idx;
        }

        let dim = node.split_dim;
        let diff = query[dim] - node.split_val;

        let (primary, secondary) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.nearest_recursive(query, primary, best_i, best_d);

        if diff.abs() < *best_d {
            self.nearest_recursive(query, secondary, best_i, best_d);
        }
    }
}

fn ransac_registration(
    source: &PointCloud,
    target: &PointCloud,
    source_fpfh: &[[f64; FPFH_DIM]],
    target_fpfh: &[[f64; FPFH_DIM]],
    params: &RegistrationParams,
    inlier_threshold: f64,
) -> Result<RegistrationResult> {
    let mut rng = rand::thread_rng();
    let n_source = source.len();
    let n_target = target.len();
    let threshold_sq = inlier_threshold * inlier_threshold;

    let target_kdtree = KdTree::from_point_cloud(target);
    let target_feature_tree = SimpleFeatureKdTree::from_features(target_fpfh);

    let feature_matches: Vec<(usize, usize)> = (0..n_source)
        .filter_map(|i| {
            let (best_j, _) = target_feature_tree.nearest(&source_fpfh[i]);
            if best_j < n_target {
                Some((i, best_j))
            } else {
                None
            }
        })
        .collect();

    let sample_count = 3usize;
    if feature_matches.len() < sample_count || n_target < sample_count {
        return Err(PointCloudError::RegistrationFailed("特征匹配点太少".to_string()));
    }

    let mut best_inliers: Vec<(usize, usize)> = Vec::new();
    let mut best_transform = Matrix4::identity();
    let mut best_rmse = f64::INFINITY;

    for _ in 0..params.ransac_max_iterations {
        let sample_matches: Vec<(usize, usize)> = feature_matches
            .choose_multiple(&mut rng, sample_count)
            .cloned()
            .collect();

        if sample_matches.len() < sample_count { continue; }

        let src_pts: Vec<Point3<f64>> = sample_matches
            .iter()
            .map(|&(s, _)| source[s].position)
            .collect();
        let tgt_pts: Vec<Point3<f64>> = sample_matches
            .iter()
            .map(|&(_, t)| target[t].position)
            .collect();

        let transform_opt = estimate_rigid_transform(&src_pts, &tgt_pts);
        if transform_opt.is_none() { continue; }
        let transform = transform_opt.unwrap();

        let (inliers, rmse) = count_inliers(
            source, &target_kdtree, &transform, threshold_sq,
            Some(params.icp_max_correspondence_distance)
        );

        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && rmse < best_rmse)
        {
            best_inliers = inliers;
            best_transform = transform;
            best_rmse = rmse;
        }
    }

    let inlier_count = best_inliers.len();
    if !best_inliers.is_empty() {
        let src_pts: Vec<_> = best_inliers.iter().map(|&(s, _)| source[s].position).collect();
        let tgt_pts: Vec<_> = best_inliers.iter().map(|&(_, t)| target[t].position).collect();
        if let Some(t) = estimate_rigid_transform(&src_pts, &tgt_pts) {
            best_transform = t;
            let (_, final_rmse) = count_inliers(
                source, &target_kdtree, &best_transform, threshold_sq,
                Some(params.icp_max_correspondence_distance)
            );
            best_rmse = final_rmse;
        }
    }

    Ok(RegistrationResult {
        transform: best_transform,
        rmse: best_rmse,
        inlier_count,
        iterations: params.ransac_max_iterations,
        converged: inlier_count > sample_count * 3,
        coarse_transform: Matrix4::identity(),
        coarse_inlier_count: 0,
    })
}

fn fpfh_distance(a: &[f64; FPFH_DIM], b: &[f64; FPFH_DIM]) -> f64 {
    let mut d = 0.0;
    for k in 0..FPFH_DIM {
        let diff = a[k] - b[k];
        d += diff * diff;
    }
    d.sqrt()
}

fn count_inliers(
    source: &PointCloud,
    target_kdtree: &KdTree,
    transform: &Matrix4<f64>,
    threshold_sq: f64,
    max_distance: Option<f64>,
) -> (Vec<(usize, usize)>, f64) {
    let mut inliers = Vec::new();
    let mut sum_sq = 0.0;
    let max_dist_sq = max_distance.map(|d| d * d).unwrap_or(f64::INFINITY);
    let effective_threshold = threshold_sq.min(max_dist_sq);

    for (i, sp) in source.iter().enumerate() {
        let p_homog = transform * sp.position.to_homogeneous();
        let p = Point3::from_homogeneous(p_homog).unwrap_or(sp.position);

        if let Some((j, dist)) = target_kdtree.nearest_neighbor(&p) {
            let dsq = dist * dist;
            if dsq <= effective_threshold {
                inliers.push((i, j));
                sum_sq += dsq;
            }
        }
    }

    let rmse = if inliers.is_empty() {
        f64::INFINITY
    } else {
        (sum_sq / inliers.len() as f64).sqrt()
    };

    (inliers, rmse)
}

pub fn estimate_rigid_transform(
    source: &[Point3<f64>],
    target: &[Point3<f64>],
) -> Option<Matrix4<f64>> {
    if source.len() != target.len() || source.len() < 3 {
        return None;
    }
    let n = source.len() as f64;

    let mut src_centroid = Point3::origin();
    let mut tgt_centroid = Point3::origin();
    for i in 0..source.len() {
        src_centroid.coords += source[i].coords;
        tgt_centroid.coords += target[i].coords;
    }
    src_centroid.coords /= n;
    tgt_centroid.coords /= n;

    let mut h = Matrix3::zeros();
    for i in 0..source.len() {
        let a = source[i].coords - src_centroid.coords;
        let b = target[i].coords - tgt_centroid.coords;
        h += a * b.transpose();
    }

    let svd = h.svd(true, true);
    let v = svd.v_t?.transpose();
    let u = svd.u?;
    let det_sign = (v * u.transpose()).determinant().signum();

    let mut corr = Matrix3::identity();
    corr[(2, 2)] = det_sign;
    let rotation = v * corr * u.transpose();
    let translation = tgt_centroid.coords - rotation * src_centroid.coords;

    let mut result = Matrix4::identity();
    result.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    result.fixed_view_mut::<3, 1>(0, 3).copy_from(&translation);
    Some(result)
}

pub fn point_to_plane_icp(
    source: &PointCloud,
    target: &PointCloud,
    initial_transform: &Matrix4<f64>,
    params: &RegistrationParams,
    inlier_threshold: f64,
) -> Result<RegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }
    if !target.has_normals() {
        return Err(PointCloudError::NormalsNotComputed);
    }

    let target_kdtree = KdTree::from_point_cloud(target);
    let mut current_transform = *initial_transform;
    let mut prev_rmse = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut final_rmse = 0.0;
    let mut final_inlier_count = 0;

    for iter in 0..params.icp_max_iterations {
        iterations = iter + 1;
        let mut correspondences = Vec::new();
        let mut sum_sq = 0.0;
        let mut inliers = 0usize;
        let inlier_threshold_sq = inlier_threshold * inlier_threshold;

        for sp in source.iter() {
            let p_homog = current_transform * sp.position.to_homogeneous();
            let p = Point3::from_homogeneous(p_homog).unwrap_or(sp.position);

            if let Some((ti, dist)) = target_kdtree.nearest_neighbor(&p) {
                if dist <= params.icp_max_correspondence_distance {
                    let tn = target[ti].normal.unwrap_or(Vector3::new(0.0, 0.0, 1.0));
                    correspondences.push((p, target[ti].position, tn));
                    let dsq = dist * dist;
                    sum_sq += dsq;
                    if dsq <= inlier_threshold_sq {
                        inliers += 1;
                    }
                }
            }
        }

        if correspondences.len() < 6 {
            break;
        }

        let rmse = (sum_sq / correspondences.len() as f64).sqrt();
        final_rmse = rmse;
        final_inlier_count = inliers;

        if iter > 0 {
            let delta_rmse = (prev_rmse - rmse).abs();
            if delta_rmse < params.icp_convergence_threshold {
                converged = true;
                break;
            }
        }
        prev_rmse = rmse;

        let delta_transform = solve_point_to_plane(&correspondences);
        current_transform = delta_transform * current_transform;
    }

    Ok(RegistrationResult {
        transform: current_transform,
        rmse: final_rmse,
        inlier_count: final_inlier_count,
        iterations,
        converged,
        coarse_transform: *initial_transform,
        coarse_inlier_count: 0,
    })
}

fn solve_point_to_plane(
    correspondences: &[(Point3<f64>, Point3<f64>, Vector3<f64>)],
) -> Matrix4<f64> {
    let n = correspondences.len();
    if n < 6 {
        return Matrix4::identity();
    }

    let mut a = DMatrix::zeros(n, 6);
    let mut b_vec = DMatrix::zeros(n, 1);

    for (i, (src, tgt, normal)) in correspondences.iter().enumerate() {
        let cross = src.coords.cross(normal);
        a[(i, 0)] = cross.x;
        a[(i, 1)] = cross.y;
        a[(i, 2)] = cross.z;
        a[(i, 3)] = normal.x;
        a[(i, 4)] = normal.y;
        a[(i, 5)] = normal.z;

        let diff = tgt.coords - src.coords;
        b_vec[(i, 0)] = normal.dot(&diff);
    }

    let ata = a.transpose() * &a;
    let atb = a.transpose() * &b_vec;

    let x = ata.lu().solve(&atb).unwrap_or_else(|| DMatrix::zeros(6, 1));

    let rx = x[(0, 0)];
    let ry = x[(1, 0)];
    let rz = x[(2, 0)];

    let rot_x = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, rx.cos(), -rx.sin(),
        0.0, rx.sin(), rx.cos(),
    );
    let rot_y = Matrix3::new(
        ry.cos(), 0.0, ry.sin(),
        0.0, 1.0, 0.0,
        -ry.sin(), 0.0, ry.cos(),
    );
    let rot_z = Matrix3::new(
        rz.cos(), -rz.sin(), 0.0,
        rz.sin(), rz.cos(), 0.0,
        0.0, 0.0, 1.0,
    );
    let rotation = rot_z * rot_y * rot_x;

    let translation = Vector3::new(x[(3, 0)], x[(4, 0)], x[(5, 0)]);

    let mut result = Matrix4::identity();
    result.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    result.fixed_view_mut::<3, 1>(0, 3).copy_from(&translation);
    result
}

pub fn apply_transform(pc: &mut PointCloud, transform: &Matrix4<f64>) {
    pc.apply_transform(transform);
}

pub fn evaluate_alignment(
    source: &PointCloud,
    target: &PointCloud,
    transform: &Matrix4<f64>,
) -> AlignmentAccuracyReport {
    let target_kdtree = KdTree::from_point_cloud(target);
    let avg_spacing_target = compute_average_point_spacing(target, 500);
    let inlier_threshold = avg_spacing_target * 3.0;
    let overlap_threshold = inlier_threshold;
    let threshold_sq = inlier_threshold * inlier_threshold;

    let mut all_distances: Vec<f64> = Vec::new();
    let mut sum_sq = 0.0;
    let mut max_error = 0.0;
    let mut inlier_count = 0usize;
    let mut overlap_count = 0usize;

    for sp in source.iter() {
        let p_homog = transform * sp.position.to_homogeneous();
        let p = Point3::from_homogeneous(p_homog).unwrap_or(sp.position);

        if let Some((_, dist)) = target_kdtree.nearest_neighbor(&p) {
            let dsq = dist * dist;
            all_distances.push(dist);
            sum_sq += dsq;

            if dist > max_error {
                max_error = dist;
            }
            if dsq <= threshold_sq {
                inlier_count += 1;
            }
            if dist <= overlap_threshold {
                overlap_count += 1;
            }
        }
    }

    let total_pairs = all_distances.len();
    let rmse = if total_pairs > 0 {
        (sum_sq / total_pairs as f64).sqrt()
    } else {
        0.0
    };

    all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_distance = if total_pairs > 0 {
        all_distances.iter().sum::<f64>() / total_pairs as f64
    } else {
        0.0
    };

    let median_distance = if total_pairs > 0 {
        all_distances[total_pairs / 2]
    } else {
        0.0
    };

    let inlier_ratio = if total_pairs > 0 {
        inlier_count as f64 / total_pairs as f64
    } else {
        0.0
    };

    let overlap_rate = if source.len() > 0 {
        overlap_count as f64 / source.len() as f64
    } else {
        0.0
    };

    let (rotation_angle_deg, translation_distance) = extract_rotation_translation(transform);

    AlignmentAccuracyReport {
        rmse,
        max_error,
        inlier_ratio,
        overlap_rate,
        rotation_angle_deg,
        translation_distance,
        mean_distance,
        median_distance,
        source_points: source.len(),
        target_points: target.len(),
        corresponding_pairs: total_pairs,
        inlier_threshold,
        overlap_threshold,
    }
}

pub fn extract_rotation_translation(transform: &Matrix4<f64>) -> (f64, f64) {
    let rotation = transform.fixed_view::<3, 3>(0, 0);
    let translation = transform.fixed_view::<3, 1>(0, 3);

    let trace = rotation[(0, 0)] + rotation[(1, 1)] + rotation[(2, 2)];
    let cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle_rad = cos_angle.acos();
    let angle_deg = angle_rad * 180.0 / std::f64::consts::PI;

    let translation_dist = translation.norm();

    (angle_deg, translation_dist)
}

fn kdtree_point(tree: &KdTree, idx: usize) -> Point3<f64> {
    tree.points[idx]
}

pub fn save_transform_matrix(transform: &Matrix4<f64>, path: &std::path::Path) -> Result<()> {
    let mut content = String::new();
    for r in 0..4 {
        let row: Vec<String> = (0..4)
            .map(|c| format!("{:.15e}", transform[(r, c)]))
            .collect();
        content.push_str(&row.join(" "));
        content.push('\n');
    }
    std::fs::write(path, content)
        .map_err(|e| PointCloudError::IoError(e))?;
    Ok(())
}

pub fn load_transform_matrix(path: &std::path::Path) -> Result<Matrix4<f64>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PointCloudError::IoError(e))?;
    let mut m = Matrix4::identity();
    for (r, line) in content.lines().enumerate() {
        if r >= 4 { break; }
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        for (c, p) in parts.iter().enumerate() {
            if c >= 4 { break; }
            if let Ok(v) = p.parse::<f64>() {
                m[(r, c)] = v;
            }
        }
    }
    Ok(m)
}
