use crate::error::Result;
use crate::types::{PointCloud, Point3D, KdTree, AABB};
use crate::utils::{mean, std_dev};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct StatisticalFilterParams {
    pub k: usize,
    pub std_ratio: f64,
}

impl Default for StatisticalFilterParams {
    fn default() -> Self {
        StatisticalFilterParams {
            k: 30,
            std_ratio: 1.5,
        }
    }
}

pub struct StatisticalFilterResult {
    pub kept_points: PointCloud,
    pub removed_count: usize,
    pub removed_ratio: f64,
}

pub fn statistical_outlier_removal(
    pc: &PointCloud,
    params: &StatisticalFilterParams,
) -> Result<StatisticalFilterResult> {
    if pc.is_empty() {
        return Ok(StatisticalFilterResult {
            kept_points: PointCloud::new(),
            removed_count: 0,
            removed_ratio: 0.0,
        });
    }

    let kdtree = KdTree::from_point_cloud(pc);

    let mean_distances: Vec<f64> = pc
        .points
        .par_iter()
        .map(|p| {
            let neighbors = kdtree.knn(&p.position, params.k + 1);
            if neighbors.len() < 2 {
                return f64::INFINITY;
            }
            let sum: f64 = neighbors.iter().skip(1).map(|(_, d)| *d).sum();
            sum / (neighbors.len() - 1) as f64
        })
        .collect();

    let m = mean(&mean_distances);
    let sigma = std_dev(&mean_distances);
    let threshold = m + params.std_ratio * sigma;

    let mut kept = Vec::new();
    let mut removed = 0usize;
    for (i, p) in pc.iter().enumerate() {
        if mean_distances[i] <= threshold {
            kept.push(p.clone());
        } else {
            removed += 1;
        }
    }

    let total = pc.len();
    let ratio = if total > 0 { removed as f64 / total as f64 } else { 0.0 };

    Ok(StatisticalFilterResult {
        kept_points: PointCloud::from_points(kept),
        removed_count: removed,
        removed_ratio: ratio,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelDownsampleParams {
    pub voxel_size: f64,
}

impl Default for VoxelDownsampleParams {
    fn default() -> Self {
        VoxelDownsampleParams { voxel_size: 0.05 }
    }
}

pub struct VoxelDownsampleResult {
    pub downsampled: PointCloud,
    pub original_count: usize,
    pub compressed_ratio: f64,
}

pub fn voxel_downsample(
    pc: &PointCloud,
    params: &VoxelDownsampleParams,
) -> Result<VoxelDownsampleResult> {
    if pc.is_empty() {
        return Ok(VoxelDownsampleResult {
            downsampled: PointCloud::new(),
            original_count: 0,
            compressed_ratio: 1.0,
        });
    }

    let voxel_size = params.voxel_size;
    let mut grid: HashMap<(i64, i64, i64), Vec<Point3D>> = HashMap::new();

    for p in pc.iter() {
        let key = (
            (p.position.x / voxel_size).floor() as i64,
            (p.position.y / voxel_size).floor() as i64,
            (p.position.z / voxel_size).floor() as i64,
        );
        grid.entry(key).or_insert_with(Vec::new).push(p.clone());
    }

    let mut result = Vec::with_capacity(grid.len());
    for (_, bucket) in grid {
        if bucket.is_empty() { continue; }
        let n = bucket.len() as f64;
        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;
        let mut has_normal = true;
        let mut has_color = true;
        let mut nx = 0.0f64;
        let mut ny = 0.0f64;
        let mut nz = 0.0f64;
        let mut r_sum = 0.0f64;
        let mut g_sum = 0.0f64;
        let mut b_sum = 0.0f64;

        for p in &bucket {
            cx += p.position.x;
            cy += p.position.y;
            cz += p.position.z;
            if let Some(n) = p.normal {
                nx += n.x;
                ny += n.y;
                nz += n.z;
            } else {
                has_normal = false;
            }
            if let Some(c) = p.color {
                r_sum += c.r as f64;
                g_sum += c.g as f64;
                b_sum += c.b as f64;
            } else {
                has_color = false;
            }
        }

        let mut centroid = Point3D::new(cx / n, cy / n, cz / n);
        if has_normal {
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len > 1e-15 {
                centroid.normal = Some(nalgebra::Vector3::new(nx / len, ny / len, nz / len));
            }
        }
        if has_color {
            centroid.color = Some(crate::types::Color::new(
                (r_sum / n).min(255.0) as u8,
                (g_sum / n).min(255.0) as u8,
                (b_sum / n).min(255.0) as u8,
            ));
        }
        result.push(centroid);
    }

    let original = pc.len();
    let after = result.len();
    let ratio = if original > 0 { after as f64 / original as f64 } else { 1.0 };

    Ok(VoxelDownsampleResult {
        downsampled: PointCloud::from_points(result),
        original_count: original,
        compressed_ratio: ratio,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct RadiusFilterParams {
    pub radius: f64,
    pub min_neighbors: usize,
}

impl Default for RadiusFilterParams {
    fn default() -> Self {
        RadiusFilterParams {
            radius: 0.1,
            min_neighbors: 5,
        }
    }
}

pub struct RadiusFilterResult {
    pub kept_points: PointCloud,
    pub removed_count: usize,
}

pub fn radius_outlier_removal(
    pc: &PointCloud,
    params: &RadiusFilterParams,
) -> Result<RadiusFilterResult> {
    if pc.is_empty() {
        return Ok(RadiusFilterResult {
            kept_points: PointCloud::new(),
            removed_count: 0,
        });
    }

    let kdtree = KdTree::from_point_cloud(pc);

    let mask: Vec<bool> = pc
        .points
        .par_iter()
        .map(|p| {
            let neighbors = kdtree.radius_search(&p.position, params.radius);
            neighbors.len().saturating_sub(1) >= params.min_neighbors
        })
        .collect();

    let mut kept = Vec::new();
    let mut removed = 0usize;
    for (i, p) in pc.iter().enumerate() {
        if mask[i] {
            kept.push(p.clone());
        } else {
            removed += 1;
        }
    }

    Ok(RadiusFilterResult {
        kept_points: PointCloud::from_points(kept),
        removed_count: removed,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct GroundFilterParams {
    pub initial_window: f64,
    pub max_window: f64,
    pub cell_size: f64,
    pub slope_threshold: f64,
    pub height_threshold: f64,
}

impl Default for GroundFilterParams {
    fn default() -> Self {
        GroundFilterParams {
            initial_window: 0.5,
            max_window: 5.0,
            cell_size: 1.0,
            slope_threshold: 0.5,
            height_threshold: 0.15,
        }
    }
}

pub struct GroundFilterResult {
    pub ground: PointCloud,
    pub non_ground: PointCloud,
}

pub fn remove_ground(
    pc: &PointCloud,
    params: &GroundFilterParams,
) -> Result<GroundFilterResult> {
    if pc.is_empty() {
        return Ok(GroundFilterResult {
            ground: PointCloud::new(),
            non_ground: PointCloud::new(),
        });
    }

    let aabb = AABB::from_points(&pc.points).unwrap();
    let cell_size = params.cell_size;
    let nx = ((aabb.max.x - aabb.min.x) / cell_size).ceil() as usize + 1;
    let ny = ((aabb.max.y - aabb.min.y) / cell_size).ceil() as usize + 1;

    let mut grid_min: Vec<Vec<f64>> = vec![vec![f64::INFINITY; ny]; nx];
    let mut grid_idx: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); ny]; nx];

    for (i, p) in pc.iter().enumerate() {
        let gx = ((p.position.x - aabb.min.x) / cell_size) as usize;
        let gy = ((p.position.y - aabb.min.y) / cell_size) as usize;
        let gx = gx.min(nx - 1);
        let gy = gy.min(ny - 1);
        grid_idx[gx][gy].push(i);
        if p.position.z < grid_min[gx][gy] {
            grid_min[gx][gy] = p.position.z;
        }
    }

    let mut is_ground = vec![false; pc.len()];
    let mut windows = Vec::new();
    let mut w = params.initial_window;
    while w <= params.max_window {
        windows.push(w);
        w *= 1.5;
    }

    let mut current_min = grid_min.clone();
    let mut seed_mask = vec![vec![false; ny]; nx];

    let last_window = windows.last().copied().unwrap_or(params.max_window);

    for &window in &windows {
        let w_cells = (window / cell_size).ceil() as isize;
        if w_cells < 1 { continue; }

        let eroded = morphological_open(&current_min, nx, ny, w_cells);
        let ht = params.height_threshold + params.slope_threshold * cell_size * w_cells as f64;

        for gx in 0..nx {
            for gy in 0..ny {
                if current_min[gx][gy] - eroded[gx][gy] < ht {
                    seed_mask[gx][gy] = true;
                    current_min[gx][gy] = current_min[gx][gy].min(eroded[gx][gy]);
                }
            }
        }
    }

    let w_cells = (last_window / cell_size).ceil() as isize;
    let final_eroded = morphological_open(&current_min, nx, ny, w_cells.max(1));

    for gx in 0..nx {
        for gy in 0..ny {
            if !seed_mask[gx][gy] { continue; }
            let base_z = final_eroded[gx][gy];
            for &pidx in &grid_idx[gx][gy] {
                if (pc[pidx].position.z - base_z).abs() <= params.height_threshold {
                    is_ground[pidx] = true;
                }
            }
        }
    }

    let mut ground = Vec::new();
    let mut non_ground = Vec::new();
    for (i, p) in pc.iter().enumerate() {
        if is_ground[i] {
            ground.push(p.clone());
        } else {
            non_ground.push(p.clone());
        }
    }

    Ok(GroundFilterResult {
        ground: PointCloud::from_points(ground),
        non_ground: PointCloud::from_points(non_ground),
    })
}

fn morphological_open(
    grid: &Vec<Vec<f64>>,
    nx: usize,
    ny: usize,
    window: isize,
) -> Vec<Vec<f64>> {
    let eroded = erode(grid, nx, ny, window);
    dilate(&eroded, nx, ny, window)
}

fn erode(grid: &Vec<Vec<f64>>, nx: usize, ny: usize, w: isize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![f64::INFINITY; ny]; nx];
    for gx in 0..nx as isize {
        for gy in 0..ny as isize {
            let mut min_val = f64::INFINITY;
            let x_start = (gx - w).max(0) as usize;
            let x_end = (gx + w + 1).min(nx as isize) as usize;
            let y_start = (gy - w).max(0) as usize;
            let y_end = (gy + w + 1).min(ny as isize) as usize;
            for xx in x_start..x_end {
                for yy in y_start..y_end {
                    if grid[xx][yy] < min_val {
                        min_val = grid[xx][yy];
                    }
                }
            }
            result[gx as usize][gy as usize] = min_val;
        }
    }
    result
}

fn dilate(grid: &Vec<Vec<f64>>, nx: usize, ny: usize, w: isize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![f64::NEG_INFINITY; ny]; nx];
    for gx in 0..nx as isize {
        for gy in 0..ny as isize {
            let mut max_val = f64::NEG_INFINITY;
            let x_start = (gx - w).max(0) as usize;
            let x_end = (gx + w + 1).min(nx as isize) as usize;
            let y_start = (gy - w).max(0) as usize;
            let y_end = (gy + w + 1).min(ny as isize) as usize;
            for xx in x_start..x_end {
                for yy in y_start..y_end {
                    if grid[xx][yy].is_finite() && grid[xx][yy] > max_val {
                        max_val = grid[xx][yy];
                    }
                }
            }
            if max_val.is_finite() {
                result[gx as usize][gy as usize] = max_val;
            } else {
                result[gx as usize][gy as usize] = grid[gx as usize][gy as usize];
            }
        }
    }
    result
}
