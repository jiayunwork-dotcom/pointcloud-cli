use nalgebra::{Point3, Vector3, Matrix3, Matrix4, OPoint, Const};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::ops::Index;
use std::ops::IndexMut;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b }
    }

    pub fn white() -> Self {
        Color { r: 255, g: 255, b: 255 }
    }

    pub fn gray() -> Self {
        Color { r: 128, g: 128, b: 128 }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3D {
    pub position: Point3<f64>,
    pub color: Option<Color>,
    pub normal: Option<Vector3<f64>>,
    pub intensity: Option<f64>,
    pub curvature: Option<f64>,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D {
            position: Point3::new(x, y, z),
            color: None,
            normal: None,
            intensity: None,
            curvature: None,
        }
    }

    pub fn with_color(x: f64, y: f64, z: f64, r: u8, g: u8, b: u8) -> Self {
        Point3D {
            position: Point3::new(x, y, z),
            color: Some(Color::new(r, g, b)),
            normal: None,
            intensity: None,
            curvature: None,
        }
    }

    pub fn with_normal(x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64) -> Self {
        Point3D {
            position: Point3::new(x, y, z),
            color: None,
            normal: Some(Vector3::new(nx, ny, nz)),
            intensity: None,
            curvature: None,
        }
    }

    pub fn x(&self) -> f64 { self.position.x }
    pub fn y(&self) -> f64 { self.position.y }
    pub fn z(&self) -> f64 { self.position.z }

    pub fn distance_to(&self, other: &Point3D) -> f64 {
        (self.position - other.position).norm()
    }

    pub fn squared_distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.position.x - other.position.x;
        let dy = self.position.y - other.position.y;
        let dz = self.position.z - other.position.z;
        dx * dx + dy * dy + dz * dz
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AABB {
    pub min: Point3<f64>,
    pub max: Point3<f64>,
}

impl AABB {
    pub fn new(min: Point3<f64>, max: Point3<f64>) -> Self {
        AABB { min, max }
    }

    pub fn from_points(points: &[Point3D]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in points {
            min.x = min.x.min(p.position.x);
            min.y = min.y.min(p.position.y);
            min.z = min.z.min(p.position.z);
            max.x = max.x.max(p.position.x);
            max.y = max.y.max(p.position.y);
            max.z = max.z.max(p.position.z);
        }
        Some(AABB { min, max })
    }

    pub fn size(&self) -> Vector3<f64> {
        self.max - self.min
    }

    pub fn center(&self) -> Point3<f64> {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn diagonal(&self) -> f64 {
        let s = self.size();
        (s.x * s.x + s.y * s.y + s.z * s.z).sqrt()
    }

    pub fn contains(&self, p: &Point3<f64>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    pub fn volume(&self) -> f64 {
        let s = self.size();
        s.x * s.y * s.z
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudSummary {
    pub total_points: usize,
    pub bounding_box: AABB,
    pub centroid: Point3<f64>,
    pub point_density: f64,
    pub has_color: bool,
    pub has_normals: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud {
    pub points: Vec<Point3D>,
}

impl PointCloud {
    pub fn new() -> Self {
        PointCloud { points: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        PointCloud { points: Vec::with_capacity(capacity) }
    }

    pub fn from_points(points: Vec<Point3D>) -> Self {
        PointCloud { points }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn push(&mut self, point: Point3D) {
        self.points.push(point);
    }

    pub fn extend(&mut self, other: PointCloud) {
        self.points.extend(other.points);
    }

    pub fn get(&self, index: usize) -> Option<&Point3D> {
        self.points.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<Point3D> {
        self.points.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Point3D> {
        self.points.iter_mut()
    }

    pub fn summary(&self) -> Option<PointCloudSummary> {
        if self.points.is_empty() {
            return None;
        }
        let aabb = AABB::from_points(&self.points).unwrap();
        let mut centroid = Point3::new(0.0, 0.0, 0.0);
        for p in &self.points {
            centroid.x += p.position.x;
            centroid.y += p.position.y;
            centroid.z += p.position.z;
        }
        let n = self.points.len() as f64;
        centroid.x /= n;
        centroid.y /= n;
        centroid.z /= n;

        let volume = aabb.volume().max(1e-10);
        let density = self.points.len() as f64 / volume;

        let has_color = self.points.iter().any(|p| p.color.is_some());
        let has_normals = self.points.iter().any(|p| p.normal.is_some());

        Some(PointCloudSummary {
            total_points: self.points.len(),
            bounding_box: aabb,
            centroid,
            point_density: density,
            has_color,
            has_normals,
        })
    }

    pub fn has_normals(&self) -> bool {
        self.points.iter().all(|p| p.normal.is_some())
    }

    pub fn has_colors(&self) -> bool {
        self.points.iter().all(|p| p.color.is_some())
    }

    pub fn apply_transform(&mut self, transform: &Matrix4<f64>) {
        for p in &mut self.points {
            let homog = nalgebra::Point3::from_homogeneous(
                transform * p.position.to_homogeneous()
            ).unwrap_or(p.position);
            p.position = homog;
            if let Some(ref mut n) = p.normal {
                let n_transformed = transform.fixed_view::<3, 3>(0, 0) * *n;
                *n = n_transformed.normalize();
            }
        }
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<usize> for PointCloud {
    type Output = Point3D;
    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl IndexMut<usize> for PointCloud {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.points[index]
    }
}

pub struct PointCloudIntoIter {
    inner: std::vec::IntoIter<Point3D>,
}

impl Iterator for PointCloudIntoIter {
    type Item = Point3D;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl IntoIterator for PointCloud {
    type Item = Point3D;
    type IntoIter = PointCloudIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        PointCloudIntoIter { inner: self.points.into_iter() }
    }
}

impl<'a> IntoIterator for &'a PointCloud {
    type Item = &'a Point3D;
    type IntoIter = std::slice::Iter<'a, Point3D>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

impl<'a> IntoIterator for &'a mut PointCloud {
    type Item = &'a mut Point3D;
    type IntoIter = std::slice::IterMut<'a, Point3D>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter_mut()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vertex {
    pub position: Point3<f64>,
    pub normal: Option<Vector3<f64>>,
    pub tex_coord: Option<(f32, f32)>,
    pub color: Option<Color>,
}

impl Vertex {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vertex {
            position: Point3::new(x, y, z),
            normal: None,
            tex_coord: None,
            color: None,
        }
    }

    pub fn from_point3(p: Point3<f64>) -> Self {
        Vertex {
            position: p,
            normal: None,
            tex_coord: None,
            color: None,
        }
    }

    pub fn with_normal(mut self, nx: f64, ny: f64, nz: f64) -> Self {
        self.normal = Some(Vector3::new(nx, ny, nz));
        self
    }

    pub fn with_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.color = Some(Color::new(r, g, b));
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TriangleFace {
    pub indices: [usize; 3],
}

impl TriangleFace {
    pub fn new(i0: usize, i1: usize, i2: usize) -> Self {
        TriangleFace { indices: [i0, i1, i2] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<TriangleFace>,
}

impl Mesh {
    pub fn new() -> Self {
        Mesh {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn with_capacity(n_vertices: usize, n_faces: usize) -> Self {
        Mesh {
            vertices: Vec::with_capacity(n_vertices),
            faces: Vec::with_capacity(n_faces),
        }
    }

    pub fn add_vertex(&mut self, v: Vertex) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(v);
        idx
    }

    pub fn add_face(&mut self, f: TriangleFace) {
        self.faces.push(f);
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    pub fn compute_face_normals(&self, face_index: usize) -> Option<Vector3<f64>> {
        let face = &self.faces[face_index];
        let v0 = &self.vertices[face.indices[0]].position;
        let v1 = &self.vertices[face.indices[1]].position;
        let v2 = &self.vertices[face.indices[2]].position;
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let n = e1.cross(&e2);
        let len = n.norm();
        if len < 1e-15 {
            None
        } else {
            Some(n / len)
        }
    }

    pub fn compute_vertex_normals(&mut self) {
        let mut normals: Vec<Vector3<f64>> = vec![Vector3::zeros(); self.vertices.len()];
        let mut counts: Vec<f64> = vec![0.0; self.vertices.len()];

        for face in &self.faces {
            if let Some(n) = self.compute_face_normal_from_indices(face.indices) {
                for &idx in &face.indices {
                    normals[idx] += n;
                    counts[idx] += 1.0;
                }
            }
        }

        for i in 0..self.vertices.len() {
            if counts[i] > 0.0 {
                let n = normals[i] / counts[i];
                let n_norm = n.norm();
                if n_norm > 1e-15 {
                    self.vertices[i].normal = Some(n / n_norm);
                }
            }
        }
    }

    fn compute_face_normal_from_indices(&self, indices: [usize; 3]) -> Option<Vector3<f64>> {
        let v0 = &self.vertices[indices[0]].position;
        let v1 = &self.vertices[indices[1]].position;
        let v2 = &self.vertices[indices[2]].position;
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let n = e1.cross(&e2);
        let len = n.norm();
        if len < 1e-15 {
            None
        } else {
            Some(n / len)
        }
    }

    pub fn aabb(&self) -> Option<AABB> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for v in &self.vertices {
            min.x = min.x.min(v.position.x);
            min.y = min.y.min(v.position.y);
            min.z = min.z.min(v.position.z);
            max.x = max.x.max(v.position.x);
            max.y = max.y.max(v.position.y);
            max.z = max.z.max(v.position.z);
        }
        Some(AABB { min, max })
    }

    pub fn volume(&self) -> f64 {
        let mut vol = 0.0;
        for face in &self.faces {
            let v0 = self.vertices[face.indices[0]].position;
            let v1 = self.vertices[face.indices[1]].position;
            let v2 = self.vertices[face.indices[2]].position;
            let cross = (v1.coords).cross(&v2.coords);
            vol += v0.coords.dot(&cross);
        }
        vol.abs() / 6.0
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl Transform {
    pub fn identity() -> Self {
        Transform {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn to_matrix4(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rotation);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn from_matrix4(m: &Matrix4<f64>) -> Self {
        Transform {
            rotation: m.fixed_view::<3, 3>(0, 0).clone_owned(),
            translation: m.fixed_view::<3, 1>(0, 3).clone_owned(),
        }
    }

    pub fn inverse(&self) -> Self {
        let rot_inv = self.rotation.transpose();
        Transform {
            rotation: rot_inv,
            translation: -(rot_inv * self.translation),
        }
    }

    pub fn apply(&self, point: &Point3<f64>) -> Point3<f64> {
        self.rotation * point + self.translation
    }
}

#[derive(Debug, Clone)]
pub struct KdNode {
    pub point_idx: usize,
    pub axis: usize,
    pub left: Option<Arc<KdNode>>,
    pub right: Option<Arc<KdNode>>,
}

pub struct KdTree {
    root: Option<Arc<KdNode>>,
    pub points: Vec<Point3<f64>>,
}

impl KdTree {
    pub fn from_point_cloud(pc: &PointCloud) -> Self {
        let points: Vec<Point3<f64>> = pc.iter().map(|p| p.position).collect();
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_recursive(&points, &mut indices, 0);
        KdTree { root, points }
    }

    pub fn from_points(points: Vec<Point3<f64>>) -> Self {
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_recursive(&points, &mut indices, 0);
        KdTree { root, points }
    }

    fn build_recursive(points: &[Point3<f64>], indices: &mut [usize], depth: usize) -> Option<Arc<KdNode>> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % 3;
        let mid = indices.len() / 2;

        indices.sort_by(|&a, &b| {
            let va = match axis {
                0 => points[a].x,
                1 => points[a].y,
                _ => points[a].z,
            };
            let vb = match axis {
                0 => points[b].x,
                1 => points[b].y,
                _ => points[b].z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let point_idx = indices[mid];
        let (left_part, rest) = indices.split_at_mut(mid);
        let (_, right_part) = rest.split_at_mut(1);

        let left = Self::build_recursive(points, &mut left_part.to_vec(), depth + 1);
        let right = Self::build_recursive(points, &mut right_part.to_vec(), depth + 1);

        Some(Arc::new(KdNode {
            point_idx,
            axis,
            left,
            right,
        }))
    }

    pub fn nearest_neighbor(&self, query: &Point3<f64>) -> Option<(usize, f64)> {
        let mut best: Vec<(usize, f64)> = Vec::with_capacity(1);
        if let Some(ref root) = self.root {
            Self::knn_recursive(root, &self.points, query, 0, &mut best, 1);
        }
        best.pop()
    }

    pub fn knn(&self, query: &Point3<f64>, k: usize) -> Vec<(usize, f64)> {
        let mut result: Vec<(usize, f64)> = Vec::with_capacity(k);
        if let Some(ref root) = self.root {
            Self::knn_recursive(root, &self.points, query, 0, &mut result, k);
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn radius_search(&self, query: &Point3<f64>, radius: f64) -> Vec<(usize, f64)> {
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            Self::radius_recursive(root, &self.points, query, radius * radius, 0, &mut result);
        }
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    fn knn_recursive(
        node: &Arc<KdNode>,
        points: &[Point3<f64>],
        query: &Point3<f64>,
        depth: usize,
        result: &mut Vec<(usize, f64)>,
        k: usize,
    ) {
        let axis = node.axis;
        let p = &points[node.point_idx];
        let dist = (p.coords - query.coords).norm_squared();

        if result.len() < k || dist < result.last().map(|x| x.1).unwrap_or(f64::INFINITY) {
            if result.len() >= k {
                result.pop();
            }
            result.push((node.point_idx, dist.sqrt()));
            result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        let query_val = match axis {
            0 => query.x,
            1 => query.y,
            _ => query.z,
        };
        let node_val = match axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        };

        let (near, far) = if query_val < node_val {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(n) = near {
            Self::knn_recursive(n, points, query, depth + 1, result, k);
        }

        let plane_dist = (query_val - node_val).powi(2);
        if let Some(f) = far {
            if result.len() < k || plane_dist < result.last().map(|x| x.1).unwrap_or(f64::INFINITY) {
                Self::knn_recursive(f, points, query, depth + 1, result, k);
            }
        }
    }

    fn radius_recursive(
        node: &Arc<KdNode>,
        points: &[Point3<f64>],
        query: &Point3<f64>,
        radius_sq: f64,
        depth: usize,
        result: &mut Vec<(usize, f64)>,
    ) {
        let axis = node.axis;
        let p = &points[node.point_idx];
        let dist_sq = (p.coords - query.coords).norm_squared();

        if dist_sq <= radius_sq {
            result.push((node.point_idx, dist_sq.sqrt()));
        }

        let query_val = match axis {
            0 => query.x,
            1 => query.y,
            _ => query.z,
        };
        let node_val = match axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        };

        let (near, far) = if query_val < node_val {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(n) = near {
            Self::radius_recursive(n, points, query, radius_sq, depth + 1, result);
        }

        let plane_dist_sq = (query_val - node_val).powi(2);
        if let Some(f) = far {
            if plane_dist_sq <= radius_sq {
                Self::radius_recursive(f, points, query, radius_sq, depth + 1, result);
            }
        }
    }
}
