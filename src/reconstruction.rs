use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Point3D, Mesh, Vertex, TriangleFace, AABB, KdTree};
use nalgebra::{Point3, Vector3, Matrix3};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub enum ReconstructionAlgorithm {
    Poisson,
    BallPivoting,
    MarchingCubes,
}

#[derive(Debug, Clone, Copy)]
pub struct PoissonParams {
    pub depth: u32,
    pub min_depth: u32,
    pub solver_divide: u32,
    pub samples_per_node: f64,
    pub point_weight: f64,
}

impl Default for PoissonParams {
    fn default() -> Self {
        PoissonParams {
            depth: 8,
            min_depth: 5,
            solver_divide: 8,
            samples_per_node: 1.0,
            point_weight: 4.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BallPivotingParams {
    pub ball_radius: f64,
    pub clustering: f64,
    pub angle_threshold: f64,
    pub delete_clusters: bool,
}

impl Default for BallPivotingParams {
    fn default() -> Self {
        BallPivotingParams {
            ball_radius: 0.01,
            clustering: 20.0,
            angle_threshold: std::f64::consts::PI / 3.0,
            delete_clusters: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarchingCubesParams {
    pub resolution: u32,
    pub iso_value: f64,
    pub min_distance: f64,
}

impl Default for MarchingCubesParams {
    fn default() -> Self {
        MarchingCubesParams {
            resolution: 64,
            iso_value: 0.0,
            min_distance: 0.001,
        }
    }
}

pub fn reconstruct_surface(
    pc: &PointCloud,
    algorithm: ReconstructionAlgorithm,
    poisson_params: &PoissonParams,
    ball_params: &BallPivotingParams,
    mc_params: &MarchingCubesParams,
) -> Result<Mesh> {
    if pc.is_empty() {
        return Err(PointCloudError::EmptyPointCloud);
    }

    match algorithm {
        ReconstructionAlgorithm::Poisson => {
            if !pc.has_normals() {
                return Err(PointCloudError::NormalsNotComputed);
            }
            poisson_reconstruction(pc, poisson_params)
        }
        ReconstructionAlgorithm::BallPivoting => {
            if !pc.has_normals() {
                return Err(PointCloudError::NormalsNotComputed);
            }
            ball_pivoting(pc, ball_params)
        }
        ReconstructionAlgorithm::MarchingCubes => {
            marching_cubes(pc, mc_params)
        }
    }
}

fn interpolate(p1: &Point3<f64>, p2: &Point3<f64>, v1: f64, v2: f64) -> Point3<f64> {
    if (v2 - v1).abs() < 1e-15 {
        return *p1;
    }
    let t = (-v1) / (v2 - v1);
    Point3::new(
        p1.x + t * (p2.x - p1.x),
        p1.y + t * (p2.y - p1.y),
        p1.z + t * (p2.z - p1.z),
    )
}

const EDGE_CONNECTIONS: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

fn get_cube_vertices(origin: &Point3<f64>, cell_size: f64) -> [Point3<f64>; 8] {
    let c = cell_size;
    [
        Point3::new(origin.x,     origin.y,     origin.z),
        Point3::new(origin.x + c, origin.y,     origin.z),
        Point3::new(origin.x + c, origin.y + c, origin.z),
        Point3::new(origin.x,     origin.y + c, origin.z),
        Point3::new(origin.x,     origin.y,     origin.z + c),
        Point3::new(origin.x + c, origin.y,     origin.z + c),
        Point3::new(origin.x + c, origin.y + c, origin.z + c),
        Point3::new(origin.x,     origin.y + c, origin.z + c),
    ]
}

const MARCHING_CUBES_EDGE_TABLE: [u16; 256] = [
    0x0000, 0x0109, 0x0203, 0x030a, 0x0406, 0x050f, 0x0605, 0x070c,
    0x080c, 0x0905, 0x0a0f, 0x0b06, 0x0c0a, 0x0d03, 0x0e09, 0x0f00,
    0x0190, 0x0099, 0x0393, 0x029a, 0x0596, 0x049f, 0x0795, 0x069c,
    0x099c, 0x0895, 0x0b9f, 0x0a96, 0x0d9a, 0x0c93, 0x0f99, 0x0e90,
    0x0230, 0x0339, 0x0033, 0x013a, 0x0636, 0x073f, 0x0435, 0x053c,
    0x0a3c, 0x0b35, 0x083f, 0x0936, 0x0e3a, 0x0f33, 0x0c39, 0x0d30,
    0x03a0, 0x02a9, 0x01a3, 0x00aa, 0x07a6, 0x06af, 0x05a5, 0x04ac,
    0x0bac, 0x0aa5, 0x09af, 0x08a6, 0x0faa, 0x0ea3, 0x0da9, 0x0ca0,
    0x0460, 0x0569, 0x0663, 0x076a, 0x0066, 0x016f, 0x0265, 0x036c,
    0x0c6c, 0x0d65, 0x0e6f, 0x0f66, 0x086a, 0x0963, 0x0a69, 0x0b60,
    0x05f0, 0x04f9, 0x07f3, 0x06fa, 0x01f6, 0x00ff, 0x03f5, 0x02fc,
    0x0dfc, 0x0cf5, 0x0fff, 0x0ef6, 0x09fa, 0x08f3, 0x0bf9, 0x0af0,
    0x0650, 0x0759, 0x0453, 0x055a, 0x0256, 0x035f, 0x0055, 0x015c,
    0x0e5c, 0x0f55, 0x0c5f, 0x0d56, 0x0a5a, 0x0b53, 0x0859, 0x0950,
    0x07c0, 0x06c9, 0x05c3, 0x04ca, 0x03c6, 0x02cf, 0x01c5, 0x00cc,
    0x0fcc, 0x0ec5, 0x0dcf, 0x0cc6, 0x0bca, 0x0ac3, 0x09c9, 0x08c0,
    0x08c0, 0x09c9, 0x0ac3, 0x0bca, 0x0cc6, 0x0dcf, 0x0ec5, 0x0fcc,
    0x00cc, 0x01c5, 0x02cf, 0x03c6, 0x04ca, 0x05c3, 0x06c9, 0x07c0,
    0x0950, 0x0859, 0x0b53, 0x0a5a, 0x0d56, 0x0c5f, 0x0f55, 0x0e5c,
    0x015c, 0x0055, 0x035f, 0x0256, 0x055a, 0x0453, 0x0759, 0x0650,
    0x0af0, 0x0bf9, 0x08f3, 0x09fa, 0x0ef6, 0x0fff, 0x0cf5, 0x0dfc,
    0x02fc, 0x03f5, 0x00ff, 0x01f6, 0x06fa, 0x07f3, 0x04f9, 0x05f0,
    0x0b60, 0x0a69, 0x0963, 0x086a, 0x0f66, 0x0e6f, 0x0d65, 0x0c6c,
    0x036c, 0x0265, 0x016f, 0x0066, 0x076a, 0x0663, 0x0569, 0x0460,
    0x0ca0, 0x0da9, 0x0ea3, 0x0faa, 0x08a6, 0x09af, 0x0aa5, 0x0bac,
    0x04ac, 0x05a5, 0x06af, 0x07a6, 0x00aa, 0x01a3, 0x02a9, 0x03a0,
    0x0d30, 0x0c39, 0x0f33, 0x0e3a, 0x0936, 0x083f, 0x0b35, 0x0a3c,
    0x053c, 0x0435, 0x073f, 0x0636, 0x013a, 0x0033, 0x0339, 0x0230,
    0x0e90, 0x0f99, 0x0c93, 0x0d9a, 0x0a96, 0x0b9f, 0x0895, 0x099c,
    0x069c, 0x0795, 0x049f, 0x0596, 0x029a, 0x0393, 0x0099, 0x0190,
    0x0f00, 0x0e09, 0x0d03, 0x0c0a, 0x0b06, 0x0a0f, 0x0905, 0x080c,
    0x070c, 0x0605, 0x050f, 0x0406, 0x030a, 0x0203, 0x0109, 0x0000,
];

const MARCHING_CUBES_TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1],
    [ 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1],
    [ 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1],
    [ 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1],
    [10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1],
    [ 5, 0,11, 5, 4, 0,11,10, 0, 3,11, 0,-1,-1,-1,-1],
    [ 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1],
    [ 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1],
    [ 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1],
    [11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1],
    [ 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1],
    [11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1],
    [11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1],
    [ 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1],
    [ 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1],
    [ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1],
    [ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1],
    [ 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1],
    [ 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1],
    [ 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1],
    [ 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1],
    [ 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1],
    [10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1],
    [ 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1],
    [ 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1],
    [ 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1],
    [ 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1],
    [11, 6, 3,11, 3, 0, 6, 4, 3, 0, 4, 3,-1,-1,-1,-1],
    [ 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1],
    [10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1],
    [10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1],
    [ 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1],
    [ 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1],
    [11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1],
    [ 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1],
    [ 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1],
    [ 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1],
    [ 7, 2, 3, 6, 2, 7, 4, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 0, 8, 7, 6, 0, 6, 2, 0, 6, 9, 2,-1,-1,-1,-1],
    [ 3, 6, 2, 3, 7, 6, 1, 9, 0, 9, 7, 0, 9, 6, 7,-1],
    [ 7, 6, 2, 7, 2, 8, 2, 1, 8, 6, 9, 1,-1,-1,-1,-1],
    [10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1],
    [ 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1],
    [ 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1],
    [ 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1],
    [ 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1],
    [10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1],
    [ 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1],
    [ 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1],
    [10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1],
    [10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1],
    [ 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1],
    [ 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1],
    [ 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1],
    [ 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1],
    [ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1],
    [ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1],
    [ 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1],
    [ 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1],
    [ 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1],
    [ 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1],
    [ 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1],
    [ 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1],
    [ 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1],
    [ 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1],
    [11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1],
    [ 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1],
    [ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1],
    [ 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1],
    [ 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1],
    [10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1],
    [ 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1],
    [11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1],
    [ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1],
    [ 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1],
    [ 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1],
    [ 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1],
    [ 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1],
    [10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1],
    [ 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1],
    [ 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1],
    [ 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1],
    [ 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1],
    [ 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1],
    [ 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1],
    [ 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1],
    [ 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1],
    [ 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1],
    [ 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1],
    [ 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1],
    [11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1],
    [ 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1],
    [ 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1],
    [ 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1],
    [ 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1],
    [ 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1],
    [ 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1],
    [ 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
];

fn process_marching_cubes_cell(
    values: &[f64; 8],
    origin: &Point3<f64>,
    cell_size: f64,
    cube_index: usize,
    mesh: &mut Mesh,
    vertex_map: &mut HashMap<(usize, usize, usize, usize), usize>,
) {
    let edge_mask = MARCHING_CUBES_EDGE_TABLE[cube_index];
    if edge_mask == 0 { return; }

    let cube_vertices = get_cube_vertices(origin, cell_size);
    let mut edge_vertices = [Point3::origin(); 12];

    for edge in 0..12usize {
        if (edge_mask & (1 << edge)) != 0 {
            let v0 = EDGE_CONNECTIONS[edge][0];
            let v1 = EDGE_CONNECTIONS[edge][1];
            edge_vertices[edge] = interpolate(
                &cube_vertices[v0], &cube_vertices[v1],
                values[v0], values[v1]
            );
        }
    }

    let tri_table = &MARCHING_CUBES_TRI_TABLE[cube_index];
    for t in (0..16).step_by(3) {
        if tri_table[t] < 0 { break; }
        let e0 = tri_table[t] as usize;
        let e1 = tri_table[t + 1] as usize;
        let e2 = tri_table[t + 2] as usize;

        let idx0 = mesh.vertex_count();
        let idx1 = mesh.vertex_count() + 1;
        let idx2 = mesh.vertex_count() + 2;

        mesh.add_vertex(Vertex::from_point3(edge_vertices[e0]));
        mesh.add_vertex(Vertex::from_point3(edge_vertices[e1]));
        mesh.add_vertex(Vertex::from_point3(edge_vertices[e2]));
        mesh.add_face(TriangleFace::new(idx0, idx1, idx2));
    }
}

pub fn poisson_reconstruction(pc: &PointCloud, params: &PoissonParams) -> Result<Mesh> {
    log::info!("开始Poisson重建，深度={}", params.depth);

    let aabb = AABB::from_points(&pc.points).unwrap();
    let size = aabb.size();
    let max_size = size.x.max(size.y).max(size.z);
    let grid_size = 2usize.pow(params.depth);
    let cell_size = max_size / grid_size as f64 * 1.1;

    let padding = cell_size * 2.0;
    let adjusted_min = Point3::new(
        aabb.min.x - padding,
        aabb.min.y - padding,
        aabb.min.z - padding,
    );

    let total_cells = grid_size + 1;
    let n_nodes = total_cells * total_cells * total_cells;
    log::info!("体素网格尺寸: {} x {} x {} = {} 节点", total_cells, total_cells, total_cells, n_nodes);

    let mut indicator = vec![0.5f64; n_nodes];

    for p in pc.iter() {
        if let Some(n) = p.normal {
            for ox in 0i32..2 {
                for oy in 0i32..2 {
                    for oz in 0i32..2 {
                        let cx = ((p.position.x - adjusted_min.x) / cell_size).floor() as i32 + ox;
                        let cy = ((p.position.y - adjusted_min.y) / cell_size).floor() as i32 + oy;
                        let cz = ((p.position.z - adjusted_min.z) / cell_size).floor() as i32 + oz;

                        if cx < 0 || cy < 0 || cz < 0 || cx >= total_cells as i32 || cy >= total_cells as i32 || cz >= total_cells as i32 {
                            continue;
                        }

                        let cell_center = Point3::new(
                            adjusted_min.x + (cx as f64 + 0.5) * cell_size,
                            adjusted_min.y + (cy as f64 + 0.5) * cell_size,
                            adjusted_min.z + (cz as f64 + 0.5) * cell_size,
                        );
                        let diff = cell_center - p.position;
                        let dot = n.dot(&diff);
                        let idx = (cx as usize) * total_cells * total_cells + (cy as usize) * total_cells + (cz as usize);
                        indicator[idx] += if dot < 0.0 {
                            params.point_weight
                        } else {
                            -params.point_weight
                        } / cell_size.powi(3);
                    }
                }
            }
        }
    }

    let iterations = params.depth.saturating_sub(params.min_depth);
    for _iter in 0..iterations {
        indicator = smooth_field(&indicator, total_cells, 3);
    }

    let min_val = indicator.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = indicator.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mid = (min_val + max_val) / 2.0;
    log::info!("指示函数范围: [{:.3}, {:.3}], 中值: {:.3}", min_val, max_val, mid);

    let mut mesh = Mesh::new();
    let mut vertex_map: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
    let grid_res = total_cells.saturating_sub(1);
    if grid_res == 0 {
        return Ok(mesh);
    }

    for cx in 0..grid_res {
        for cy in 0..grid_res {
            for cz in 0..grid_res {
                let idx = |x, y, z| x * total_cells * total_cells + y * total_cells + z;
                let vals = [
                    indicator[idx(cx, cy, cz)] - mid,
                    indicator[idx(cx+1, cy, cz)] - mid,
                    indicator[idx(cx+1, cy+1, cz)] - mid,
                    indicator[idx(cx, cy+1, cz)] - mid,
                    indicator[idx(cx, cy, cz+1)] - mid,
                    indicator[idx(cx+1, cy, cz+1)] - mid,
                    indicator[idx(cx+1, cy+1, cz+1)] - mid,
                    indicator[idx(cx, cy+1, cz+1)] - mid,
                ];
                let vals = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]];

                let mut cube_index = 0usize;
                for (i, &v) in vals.iter().enumerate() {
                    if v < 0.0 { cube_index |= 1 << i; }
                }
                if cube_index == 0 || cube_index == 255 { continue; }

                let origin = Point3::new(
                    adjusted_min.x + cx as f64 * cell_size,
                    adjusted_min.y + cy as f64 * cell_size,
                    adjusted_min.z + cz as f64 * cell_size,
                );
                process_marching_cubes_cell(
                    &vals, &origin, cell_size, cube_index,
                    &mut mesh, &mut vertex_map,
                );
            }
        }
    }

    mesh.compute_vertex_normals();
    log::info!("Poisson重建完成: {} 顶点, {} 面片", mesh.vertex_count(), mesh.face_count());
    Ok(mesh)
}

fn smooth_field(field: &[f64], size: usize, iterations: usize) -> Vec<f64> {
    let mut current = field.to_vec();
    for _ in 0..iterations {
        let mut next = current.clone();
        for x in 1..size.saturating_sub(1) {
            for y in 1..size.saturating_sub(1) {
                for z in 1..size.saturating_sub(1) {
                    let idx = x * size * size + y * size + z;
                    let sum =
                        current[(x-1)*size*size + y*size + z] +
                        current[(x+1)*size*size + y*size + z] +
                        current[x*size*size + (y-1)*size + z] +
                        current[x*size*size + (y+1)*size + z] +
                        current[x*size*size + y*size + (z-1)] +
                        current[x*size*size + y*size + (z+1)];
                    next[idx] = (current[idx] + sum / 6.0) / 2.0;
                }
            }
        }
        current = next;
    }
    current
}

fn sort_tri(a: usize, b: usize, c: usize) -> (usize, usize, usize) {
    let mut v = [a, b, c];
    v.sort();
    (v[0], v[1], v[2])
}

fn circumscribed_center(
    p1_idx: usize,
    p2_idx: usize,
    p3_idx: usize,
    pc: &PointCloud,
    ball_radius: f64,
) -> Option<Point3<f64>> {
    let p1 = &pc[p1_idx].position;
    let p2 = &pc[p2_idx].position;
    let p3 = &pc[p3_idx].position;
    let v1 = p2 - p1;
    let v2 = p3 - p1;
    let normal = v1.cross(&v2);
    let area2 = normal.norm();
    if area2 < 1e-15 { return None; }
    let normal = normal / area2;

    let d1 = v1.norm_squared();
    let d2 = v2.norm_squared();
    let d12 = v1.dot(&v2);
    let denom = 2.0 * (d1 * d2 - d12 * d12);
    if denom.abs() < 1e-20 { return None; }
    let alpha = (d2 * (d1 - d12)) / denom;
    let beta = (d1 * (d2 - d12)) / denom;

    let circum = p1 + v1 * alpha + v2 * beta;
    let radius = (circum - p1).norm();

    if radius > ball_radius * 2.0 { return None; }
    let r_sq = ball_radius * ball_radius;
    let depth_sq = (r_sq - radius * radius).max(0.0);
    let depth = depth_sq.sqrt();

    let n1 = pc[p1_idx].normal.unwrap_or(Vector3::z());
    let direction = if normal.dot(&n1) > 0.0 { -normal } else { normal };
    Some(circum + direction * depth)
}

fn check_empty_ball(
    center: &Point3<f64>,
    r_sq: f64,
    kdtree: &KdTree,
    exclude: &[usize],
    pc: &PointCloud,
) -> bool {
    let r = r_sq.sqrt();
    let neighbors = kdtree.radius_search(center, r * 1.01);
    for (idx, _dist) in &neighbors {
        if exclude.contains(idx) { continue; }
        let actual = (center.coords - pc[*idx].position.coords).norm_squared();
        if actual < r_sq * 0.999 {
            return false;
        }
    }
    true
}

pub fn ball_pivoting(pc: &PointCloud, params: &BallPivotingParams) -> Result<Mesh> {
    log::info!("开始Ball Pivoting重建，球半径={}", params.ball_radius);

    let kdtree = KdTree::from_point_cloud(pc);
    let n = pc.len();
    let mut mesh = Mesh::with_capacity(n, n * 2);
    for p in pc.iter() {
        let mut v = Vertex::from_point3(p.position);
        if let Some(c) = p.color { v = v.with_color(c.r, c.g, c.b); }
        if let Some(nn) = p.normal { v = v.with_normal(nn.x, nn.y, nn.z); }
        mesh.add_vertex(v);
    }

    let search_radius = params.ball_radius * 4.0;
    let r_sq = params.ball_radius * params.ball_radius;

    let mut edges: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    let mut active_edges: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
    let mut visited: std::collections::HashSet<(usize, usize, usize)> = std::collections::HashSet::new();

    'seed: for seed_i in 0..n {
        let neighbors = kdtree.radius_search(&pc[seed_i].position, search_radius);
        for &(j, _) in &neighbors {
            if j == seed_i { continue; }
            for &(k, _) in &neighbors {
                if k == seed_i || k == j { continue; }
                let tri_key = sort_tri(seed_i, j, k);
                if visited.contains(&tri_key) { continue; }
                if let Some(circ) = circumscribed_center(seed_i, j, k, pc, params.ball_radius) {
                    if check_empty_ball(&circ, r_sq, &kdtree, &[seed_i, j, k], pc) {
                        mesh.add_face(TriangleFace::new(seed_i, j, k));
                        visited.insert(tri_key);
                        let edge_key1 = if seed_i < j { (seed_i, j) } else { (j, seed_i) };
                        edges.insert(edge_key1, edges.get(&edge_key1).unwrap_or(&0) + 1);
                        active_edges.push_back(edge_key1);
                        let edge_key2 = if j < k { (j, k) } else { (k, j) };
                        edges.insert(edge_key2, edges.get(&edge_key2).unwrap_or(&0) + 1);
                        active_edges.push_back(edge_key2);
                        let edge_key3 = if k < seed_i { (k, seed_i) } else { (seed_i, k) };
                        edges.insert(edge_key3, edges.get(&edge_key3).unwrap_or(&0) + 1);
                        active_edges.push_back(edge_key3);
                        break 'seed;
                    }
                }
            }
        }
    }

    let max_iter = active_edges.len() * 10;
    let mut iter = 0;
    while let Some(edge) = active_edges.pop_front() {
        iter += 1;
        if iter > max_iter { break; }

        let count = *edges.get(&edge).unwrap_or(&0);
        if count >= 2 { continue; }

        let (a, b) = edge;
        let neighbors_a = kdtree.radius_search(&pc[a].position, search_radius);
        let neighbors_b = kdtree.radius_search(&pc[b].position, search_radius);
        let mut candidates: Vec<usize> = neighbors_a.iter().chain(&neighbors_b)
            .map(|(i, _)| *i)
            .filter(|&i| i != a && i != b)
            .collect();
        candidates.sort();
        candidates.dedup();

        for &c in &candidates {
            let tri_key = sort_tri(a, b, c);
            if visited.contains(&tri_key) { continue; }
            if let Some(circ) = circumscribed_center(a, b, c, pc, params.ball_radius) {
                if check_empty_ball(&circ, r_sq, &kdtree, &[a, b, c], pc) {
                    mesh.add_face(TriangleFace::new(a, b, c));
                    visited.insert(tri_key);
                    let ek1 = if a < b { (a, b) } else { (b, a) };
                    edges.insert(ek1, edges.get(&ek1).unwrap_or(&0) + 1);
                    let ek2 = if b < c { (b, c) } else { (c, b) };
                    let cnt2 = *edges.get(&ek2).unwrap_or(&0);
                    edges.insert(ek2, cnt2 + 1);
                    if cnt2 < 2 { active_edges.push_back(ek2); }
                    let ek3 = if c < a { (c, a) } else { (a, c) };
                    let cnt3 = *edges.get(&ek3).unwrap_or(&0);
                    edges.insert(ek3, cnt3 + 1);
                    if cnt3 < 2 { active_edges.push_back(ek3); }
                    break;
                }
            }
        }
    }

    mesh.compute_vertex_normals();
    log::info!("Ball Pivoting完成: {} 面片", mesh.face_count());
    Ok(mesh)
}

pub fn marching_cubes(pc: &PointCloud, params: &MarchingCubesParams) -> Result<Mesh> {
    log::info!("开始Marching Cubes重建，分辨率={}", params.resolution);

    let aabb = AABB::from_points(&pc.points).unwrap();
    let size = aabb.size();
    let diag = size.norm();
    let padding = diag * 0.05;
    let adjusted_min = Point3::new(
        aabb.min.x - padding,
        aabb.min.y - padding,
        aabb.min.z - padding,
    );
    let adjusted_max = Point3::new(
        aabb.max.x + padding,
        aabb.max.y + padding,
        aabb.max.z + padding,
    );
    let real_size = adjusted_max - adjusted_min;
    let cell_size = real_size.x.max(real_size.y).max(real_size.z) / params.resolution as f64;
    let total = real_size / cell_size;
    let nx = (total.x as usize).max(2) + 1;
    let ny = (total.y as usize).max(2) + 1;
    let nz = (total.z as usize).max(2) + 1;
    let n_cells = nx * ny * nz;
    log::info!("距离场: {}x{}x{} = {} 采样点", nx, ny, nz, n_cells);

    let kdtree = KdTree::from_point_cloud(pc);
    let mut distance_field = vec![0.0f64; n_cells];

    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    for cx in 0..nx {
        for cy in 0..ny {
            for cz in 0..nz {
                let sample = Point3::new(
                    adjusted_min.x + cx as f64 * cell_size,
                    adjusted_min.y + cy as f64 * cell_size,
                    adjusted_min.z + cz as f64 * cell_size,
                );
                let i = idx(cx, cy, cz);
                if let Some((pi, dist)) = kdtree.nearest_neighbor(&sample) {
                    let p = &pc[pi];
                    let signed = if let Some(n) = p.normal {
                        let diff = sample - p.position;
                        if n.dot(&diff) >= 0.0 { dist } else { -dist }
                    } else { dist };
                    distance_field[i] = signed;
                } else {
                    distance_field[i] = diag;
                }
            }
        }
    }

    let mut mesh = Mesh::new();
    let mut vertex_map = HashMap::new();

    for cx in 0..nx.saturating_sub(1) {
        for cy in 0..ny.saturating_sub(1) {
            for cz in 0..nz.saturating_sub(1) {
                let vals = [
                    distance_field[idx(cx, cy, cz)] - params.iso_value,
                    distance_field[idx(cx+1, cy, cz)] - params.iso_value,
                    distance_field[idx(cx+1, cy+1, cz)] - params.iso_value,
                    distance_field[idx(cx, cy+1, cz)] - params.iso_value,
                    distance_field[idx(cx, cy, cz+1)] - params.iso_value,
                    distance_field[idx(cx+1, cy, cz+1)] - params.iso_value,
                    distance_field[idx(cx+1, cy+1, cz+1)] - params.iso_value,
                    distance_field[idx(cx, cy+1, cz+1)] - params.iso_value,
                ];
                let vals = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]];

                let mut cube_index = 0usize;
                for (i, &v) in vals.iter().enumerate() {
                    if v < 0.0 { cube_index |= 1 << i; }
                }
                if cube_index == 0 || cube_index == 255 { continue; }
                let origin = Point3::new(
                    adjusted_min.x + cx as f64 * cell_size,
                    adjusted_min.y + cy as f64 * cell_size,
                    adjusted_min.z + cz as f64 * cell_size,
                );
                process_marching_cubes_cell(
                    &vals, &origin, cell_size, cube_index,
                    &mut mesh, &mut vertex_map,
                );
            }
        }
    }

    mesh.compute_vertex_normals();
    log::info!("Marching Cubes完成: {} 顶点, {} 面片", mesh.vertex_count(), mesh.face_count());
    Ok(mesh)
}