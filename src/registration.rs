use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Point3D, KdTree, Transform, AABB};
use nalgebra::{Point3, Vector3, Matrix3, Matrix4, Vector6, DMatrix};
use rand::seq::SliceRandom;

#[derive(Debug, Clone, Copy)]
pub struct RegistrationParams {
    pub fpfh_radius: f64,
    pub ransac_max_iterations: usize,
    pub ransac_inlier_threshold: f64,
    pub icp_max_iterations: usize,
    pub icp_convergence_threshold: f64,
    pub icp_max_correspondence_distance: f64,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        RegistrationParams {
            fpfh_radius: 0.1,
            ransac_max_iterations: 100000,
            ransac_inlier_threshold: 0.05,
            icp_max_iterations: 100,
            icp_convergence_threshold: 1e-8,
            icp_max_correspondence_distance: 1.0,
        }
    }
}

pub struct RegistrationResult {
    pub transform: Matrix4<f64>,
    pub rmse: f64,
    pub inlier_count: usize,
    pub iterations: usize,
    pub converged: bool,
}

pub fn register_point_clouds(
    source: &PointCloud,
    target: &PointCloud,
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    if !source.has_normals() || !target.has_normals() {
        return Err(PointCloudError::NormalsNotComputed);
    }

    log::info!("计算源点云FPFH特征...");
    let source_fpfh = compute_fpfh(source, params.fpfh_radius)?;
    log::info!("计算目标点云FPFH特征...");
    let target_fpfh = compute_fpfh(target, params.fpfh_radius)?;

    log::info!("执行RANSAC粗配准...");
    let coarse = ransac_registration(
        source, target, &source_fpfh, &target_fpfh, params
    )?;

    log::info!("执行Point-to-Plane ICP精配准...");
    let icp = point_to_plane_icp(source, target, &coarse.transform, params)?;

    Ok(icp)
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
    let v = u.cross(&diff).normalize();
    let w = u.cross(&v);

    let alpha = v.dot(&nj).clamp(-1.0, 1.0);
    let phi = u.dot(&diff).atan2(diff.norm());
    let theta = w.dot(&nj).clamp(-1.0, 1.0);

    (alpha, phi, theta)
}

fn ransac_registration(
    source: &PointCloud,
    target: &PointCloud,
    source_fpfh: &[[f64; FPFH_DIM]],
    target_fpfh: &[[f64; FPFH_DIM]],
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    let mut rng = rand::thread_rng();
    let n_source = source.len();
    let n_target = target.len();
    let threshold_sq = params.ransac_inlier_threshold * params.ransac_inlier_threshold;

    let target_kdtree = KdTree::from_point_cloud(target);

    let source_indices: Vec<usize> = (0..n_source).collect();
    let mut best_inliers = Vec::new();
    let mut best_transform = Matrix4::identity();
    let mut best_rmse = f64::INFINITY;

    let sample_count = 4usize;
    if n_source < sample_count || n_target < sample_count {
        return Err(PointCloudError::RegistrationFailed("点云点数太少".to_string()));
    }

    let feature_matches = find_feature_matches(source_fpfh, target_fpfh, 500);

    for _ in 0..params.ransac_max_iterations {
        let sample_source: Vec<usize> = (0..sample_count)
            .map(|_| {
                feature_matches.choose(&mut rng).map(|(s, _)| *s).unwrap_or_else(|| rand::random::<usize>() % n_source)
            })
            .collect();

        let sample_target: Vec<usize> = sample_source.iter()
            .map(|&s| {
                find_closest_feature(source_fpfh, target_fpfh, s).unwrap_or(rand::random::<usize>() % n_target)
            })
            .collect();

        let transform = estimate_rigid_transform(
            &source.points.iter().enumerate()
                .filter(|(i, _)| sample_source.contains(i))
                .map(|(_, p)| p.position)
                .collect::<Vec<_>>(),
            &target.points.iter().enumerate()
                .filter(|(i, _)| sample_target.contains(i))
                .map(|(_, p)| p.position)
                .collect::<Vec<_>>(),
        );

        if transform.is_none() { continue; }
        let transform = transform.unwrap();

        let (inliers, rmse) = count_inliers(source, target, &target_kdtree, &transform, threshold_sq);

        if inliers.len() > best_inliers.len() || (inliers.len() == best_inliers.len() && rmse < best_rmse) {
            best_inliers = inliers;
            best_transform = transform;
            best_rmse = rmse;
        }
    }

    if best_inliers.is_empty() {
        best_transform = Matrix4::identity();
        best_rmse = 0.0;
    }

    let final_iterations = if !best_inliers.is_empty() {
        let src_pts: Vec<_> = best_inliers.iter().map(|&(s, _)| source[s].position).collect();
        let tgt_pts: Vec<_> = best_inliers.iter().map(|&(_, t)| target[t].position).collect();
        if let Some(t) = estimate_rigid_transform(&src_pts, &tgt_pts) {
            best_transform = t;
        }
        params.ransac_max_iterations
    } else {
        0
    };

    Ok(RegistrationResult {
        transform: best_transform,
        rmse: best_rmse,
        inlier_count: best_inliers.len(),
        iterations: final_iterations,
        converged: !best_inliers.is_empty(),
    })
}

fn find_feature_matches(
    source_fpfh: &[[f64; FPFH_DIM]],
    target_fpfh: &[[f64; FPFH_DIM]],
    max_matches: usize,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let step = (source_fpfh.len() / max_matches.max(1)).max(1);
    for i in (0..source_fpfh.len()).step_by(step) {
        if matches.len() >= max_matches { break; }
        let mut best_j = 0;
        let mut best_d = f64::INFINITY;
        for j in 0..target_fpfh.len() {
            let d = fpfh_distance(&source_fpfh[i], &target_fpfh[j]);
            if d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        matches.push((i, best_j));
    }
    matches
}

fn find_closest_feature(
    source_fpfh: &[[f64; FPFH_DIM]],
    target_fpfh: &[[f64; FPFH_DIM]],
    idx: usize,
) -> Option<usize> {
    if idx >= source_fpfh.len() { return None; }
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for j in 0..target_fpfh.len() {
        let d = fpfh_distance(&source_fpfh[idx], &target_fpfh[j]);
        if d < best_d {
            best_d = d;
            best = j;
        }
    }
    Some(best)
}

fn fpfh_distance(a: &[f64; FPFH_DIM], b: &[f64; FPFH_DIM]) -> f64 {
    let mut d = 0.0;
    for k in 0..FPFH_DIM {
        let diff = a[k] - b[k];
        d += diff * diff;
    }
    d
}

fn count_inliers(
    source: &PointCloud,
    _target: &PointCloud,
    target_kdtree: &KdTree,
    transform: &Matrix4<f64>,
    threshold_sq: f64,
) -> (Vec<(usize, usize)>, f64) {
    let mut inliers = Vec::new();
    let mut sum_sq = 0.0;

    for (i, sp) in source.iter().enumerate() {
        let p_homog = transform * sp.position.to_homogeneous();
        let p = Point3::from_homogeneous(p_homog).unwrap_or(sp.position);

        if let Some((j, dist)) = target_kdtree.nearest_neighbor(&p) {
            let dsq = dist * dist;
            if dsq <= threshold_sq {
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
    let mut v = svd.v_t.unwrap().transpose();
    let u = svd.u.unwrap();
    let mut d = (v * u.transpose()).determinant();
    let signum = if d < 0.0 { -1.0 } else { 1.0 };
    d = d.signum();

    let mut corr = Matrix3::identity();
    corr[(2, 2)] = signum;
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
) -> Result<RegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }
    if !target.has_normals() {
        return Err(PointCloudError::NormalsNotComputed);
    }

    let target_kdtree = KdTree::from_point_cloud(target);
    let mut current_transform = *initial_transform;
    let mut prev_error = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut final_rmse = 0.0;

    for iter in 0..params.icp_max_iterations {
        iterations = iter + 1;
        let mut correspondences = Vec::new();
        let mut sum_sq = 0.0;

        for sp in source.iter() {
            let p_homog = current_transform * sp.position.to_homogeneous();
            let p = Point3::from_homogeneous(p_homog).unwrap_or(sp.position);

            if let Some((ti, dist)) = target_kdtree.nearest_neighbor(&p) {
                if dist <= params.icp_max_correspondence_distance {
                    correspondences.push((p, target[ti].position, target[ti].normal.unwrap()));
                    sum_sq += dist * dist;
                }
            }
        }

        if correspondences.len() < 6 {
            break;
        }

        let rmse = (sum_sq / correspondences.len() as f64).sqrt();
        final_rmse = rmse;
        let delta = (prev_error - rmse).abs();
        if delta < params.icp_convergence_threshold && iter > 0 {
            converged = true;
            break;
        }
        prev_error = rmse;

        let delta_transform = solve_point_to_plane(&correspondences);
        current_transform = delta_transform * current_transform;
    }

    Ok(RegistrationResult {
        transform: current_transform,
        rmse: final_rmse,
        inlier_count: source.len(),
        iterations,
        converged,
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

    let _tx = x[(3, 0)];
    let _ty = x[(4, 0)];
    let _tz = x[(5, 0)];

    let rx_skew = Matrix3::new(
        0.0, -rx, rx,
        rx, 0.0, -rx,
        -rx, rx, 0.0,
    );
    let _ = rx_skew;

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
