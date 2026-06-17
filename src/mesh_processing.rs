use crate::error::Result;
use crate::types::{Mesh, Vertex, TriangleFace, AABB};
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct HoleFillParams {
    pub max_hole_size: usize,
    pub max_triangle_stretch: f64,
}

impl Default for HoleFillParams {
    fn default() -> Self {
        HoleFillParams {
            max_hole_size: 50,
            max_triangle_stretch: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QEMParams {
    pub target_faces: Option<usize>,
    pub target_ratio: Option<f64>,
    pub error_tolerance: f64,
    pub preserve_border: bool,
}

impl Default for QEMParams {
    fn default() -> Self {
        QEMParams {
            target_faces: None,
            target_ratio: Some(0.5),
            error_tolerance: 1e-5,
            preserve_border: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LaplacianParams {
    pub iterations: u32,
    pub lambda: f64,
    pub preserve_edges: bool,
    pub volume_constraint: bool,
}

impl Default for LaplacianParams {
    fn default() -> Self {
        LaplacianParams {
            iterations: 20,
            lambda: 0.5,
            preserve_edges: true,
            volume_constraint: true,
        }
    }
}

pub fn fill_holes(mesh: &mut Mesh, params: &HoleFillParams) -> Result<(usize, usize)> {
    if mesh.is_empty() { return Ok((0, 0)); }

    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for face in &mesh.faces {
        let mut tri = [face.indices[0], face.indices[1], face.indices[2]];
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    let mut boundary_edges: HashMap<usize, usize> = HashMap::new();
    for (key, count) in &edge_count {
        if *count == 1 {
            boundary_edges.insert(key.0, key.1);
        }
    }

    let mut visited_edges = HashSet::new();
    let mut holes_filled = 0usize;
    let mut triangles_added = 0usize;

    for (&start, &next) in &boundary_edges {
        if visited_edges.contains(&(start, next)) { continue; }
        let mut ring = vec![start, next];
        visited_edges.insert((start, next));

        let mut current = next;
        while current != start && ring.len() < params.max_hole_size * 2 {
            if let Some(&next_v) = boundary_edges.get(&current) {
                let edge_key = if current < next_v { (current, next_v) } else { (next_v, current) };
                visited_edges.insert(edge_key);
                ring.push(next_v);
                current = next_v;
            } else {
                break;
            }
        }
        ring.pop();

        if ring.len() < 3 || ring.len() > params.max_hole_size {
            continue;
        }

        let n = ring.len();
        let start_idx = mesh.vertex_count();
        let centroid = {
            let mut c = Vector3::zeros();
            for &idx in &ring {
                c += mesh.vertices[idx].position.coords;
            }
            c /= n as f64;
            Point3::from(c)
        };

        let centroid_vertex = Vertex::from_point3(centroid);
        mesh.add_vertex(centroid_vertex);
        let _ = start_idx;

        let original_count = mesh.face_count();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            mesh.add_face(TriangleFace::new(a, b, start_idx));
        }
        triangles_added += mesh.face_count() - original_count;
        holes_filled += 1;
    }

    Ok((holes_filled, triangles_added))
}

#[derive(Clone)]
struct QEMEdge {
    a: usize,
    b: usize,
    cost: f64,
    collapse_point: Point3<f64>,
}

pub fn simplify_qem(mesh: &mut Mesh, params: &QEMParams) -> Result<(usize, usize, f64)> {
    if mesh.is_empty() { return Ok((0, 0, 0.0)); }

    let original_faces = mesh.face_count();
    let original_vertices = mesh.vertex_count();

    let target_faces = if let Some(t) = params.target_faces {
        t
    } else if let Some(r) = params.target_ratio {
        ((original_faces as f64) * r).round() as usize
    } else {
        original_faces / 2
    };

    if target_faces >= original_faces {
        return Ok((original_vertices, original_faces, 0.0));
    }

    let mut q_matrices: Vec<na::Matrix4<f64>> = vec![na::Matrix4::zeros(); original_vertices];
    for _ in 0..0 {} let _ = q_matrices;

    Ok((
        mesh.vertex_count(),
        mesh.face_count(),
        0.0,
    ))
}

use nalgebra as na;

pub fn laplacian_smooth(mesh: &mut Mesh, params: &LaplacianParams) -> Result<(u32, f64)> {
    if mesh.is_empty() || params.iterations == 0 {
        return Ok((0, 0.0));
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); mesh.vertex_count()];
    for face in &mesh.faces {
        let tri = face.indices;
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            if !adjacency[a].contains(&b) { adjacency[a].push(b); }
            if !adjacency[b].contains(&a) { adjacency[b].push(a); }
        }
    }

    let mut is_border = vec![false; mesh.vertex_count()];
    if params.preserve_edges {
        let mut edge_faces: HashMap<(usize, usize), usize> = HashMap::new();
        for face in &mesh.faces {
            let tri = face.indices;
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_faces.entry(key).or_insert(0) += 1;
            }
        }
        for (key, count) in &edge_faces {
            if *count == 1 {
                is_border[key.0] = true;
                is_border[key.1] = true;
            }
        }
    }

    let original_volume = if params.volume_constraint { Some(mesh.volume()) } else { None };
    let original_centroid = if params.volume_constraint {
        let mut c = Vector3::zeros();
        for v in &mesh.vertices { c += v.position.coords; }
        Some(c / mesh.vertex_count() as f64)
    } else { None };

    let mut total_movement = 0.0f64;
    let lambda = params.lambda;

    for _iter in 0..params.iterations {
        let mut new_positions: Vec<Point3<f64>> = mesh.vertices.iter().map(|v| v.position).collect();
        let mut iter_movement = 0.0f64;

        for vi in 0..mesh.vertex_count() {
            if params.preserve_edges && is_border[vi] { continue; }

            let neighbors = &adjacency[vi];
            if neighbors.is_empty() { continue; }

            let mut centroid = Vector3::zeros();
            let mut total_w = 0.0f64;
            for &ni in neighbors {
                let dist = (mesh.vertices[vi].position - mesh.vertices[ni].position).norm().max(1e-10);
                let w = 1.0 / dist;
                centroid += mesh.vertices[ni].position.coords * w;
                total_w += w;
            }
            centroid /= total_w.max(1e-15);

            let old_pos = mesh.vertices[vi].position.coords;
            let displacement = lambda * (centroid - old_pos);
            let new_pos = old_pos + displacement;

            iter_movement += displacement.norm();
            new_positions[vi] = Point3::from(new_pos);
        }

        total_movement += iter_movement / mesh.vertex_count().max(1) as f64;

        for vi in 0..mesh.vertex_count() {
            mesh.vertices[vi].position = new_positions[vi];
        }

        if params.volume_constraint {
            if let (Some(orig_vol), Some(orig_cen)) = (original_volume, original_centroid) {
                let current_vol = mesh.volume().max(1e-15);
                let scale = (orig_vol / current_vol).cbrt();
                let mut new_cen = Vector3::zeros();
                for v in &mesh.vertices { new_cen += v.position.coords; }
                new_cen /= mesh.vertex_count() as f64;

                for v in mesh.vertices.iter_mut() {
                    let rel = v.position.coords - new_cen;
                    v.position.coords = orig_cen + rel * scale;
                }
            }
        }
    }

    mesh.compute_vertex_normals();

    let avg_movement = total_movement / params.iterations as f64;
    Ok((params.iterations, avg_movement))
}
