use crate::error::{Result, PointCloudError};
use crate::types::{PointCloud, Point3D, Mesh, Vertex, TriangleFace, Color};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write, Read};
use std::path::{Path, PathBuf};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointCloudFormat {
    PLY,
    PCD,
    LAS,
    XYZ,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    PLY,
    OBJ,
    STL,
}

#[derive(Clone)]
struct PLYProperty {
    name: String,
    dtype: String,
}

struct PLYHeader {
    is_binary: bool,
    little_endian: bool,
    vertex_count: usize,
    properties: Vec<PLYProperty>,
    header_bytes: usize,
}

struct PCDHeader {
    fields: Vec<String>,
    sizes: Vec<usize>,
    types: Vec<char>,
    counts: Vec<usize>,
    width: usize,
    height: usize,
    points: usize,
    data_type: String,
}

struct LASHeader {
    version_major: u8,
    version_minor: u8,
    point_data_format: u8,
    point_data_start: u32,
    point_count: u64,
    scale: [f64; 3],
    offset: [f64; 3],
    point_record_length: u16,
}

pub struct PointCloudReader {
    pub large_file_threshold: usize,
}

impl Default for PointCloudReader {
    fn default() -> Self {
        PointCloudReader {
            large_file_threshold: 5_000_000,
        }
    }
}

impl PointCloudReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read(&self, path: &Path) -> Result<PointCloud> {
        let format = crate::utils::detect_file_format(path)?;
        match format {
            PointCloudFormat::PLY => self.read_ply(path),
            PointCloudFormat::PCD => self.read_pcd(path),
            PointCloudFormat::LAS => self.read_las(path),
            PointCloudFormat::XYZ => self.read_xyz(path),
        }
    }

    pub fn read_streaming<F>(&self, path: &Path, callback: F) -> Result<usize>
    where
        F: FnMut(Point3D) -> Result<()>,
    {
        let format = crate::utils::detect_file_format(path)?;
        match format {
            PointCloudFormat::PLY => self.read_ply_streaming(path, callback),
            PointCloudFormat::PCD => self.read_pcd_streaming(path, callback),
            PointCloudFormat::LAS => self.read_las_streaming(path, callback),
            PointCloudFormat::XYZ => self.read_xyz_streaming(path, callback),
        }
    }

    fn read_ply(&self, path: &Path) -> Result<PointCloud> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let estimated_points = metadata.len() as usize / 32;

        if estimated_points > self.large_file_threshold {
            log::warn!("大文件检测，采用流式读取: {} (估计{}点)", path.display(), estimated_points);
            let mut pc = PointCloud::with_capacity(estimated_points);
            self.read_ply_streaming(path, |p| {
                pc.push(p);
                Ok(())
            })?;
            Ok(pc)
        } else {
            self.read_ply_internal(path)
        }
    }

    fn read_ply_internal(&self, path: &Path) -> Result<PointCloud> {
        use std::io::Seek;
        let mut file = File::open(path)?;
        let mut reader = BufReader::new(&mut file);

        let header = self.parse_ply_header(&mut reader)?;
        let header_bytes = header.header_bytes;

        file.seek(std::io::SeekFrom::Start(header_bytes as u64))?;
        reader = BufReader::new(&mut file);

        let mut points = Vec::with_capacity(header.vertex_count);

        if header.is_binary {
            self.read_ply_binary(&mut reader, &header, &mut points)?;
        } else {
            self.read_ply_ascii(&mut reader, &header, &mut points)?;
        }

        Ok(PointCloud::from_points(points))
    }

    fn parse_ply_header<R: BufRead>(&self, reader: &mut R) -> Result<PLYHeader> {
        let mut line = String::new();
        let mut header_bytes = 0usize;
        let mut is_binary = false;
        let mut little_endian = true;
        let mut vertex_count = 0usize;
        let mut properties: Vec<PLYProperty> = Vec::new();
        let mut in_vertex = false;

        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            header_bytes += n;
            let trimmed = line.trim();

            if trimmed == "ply" {
                continue;
            }
            if trimmed.starts_with("format") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    match parts[1] {
                        "ascii" => is_binary = false,
                        "binary_little_endian" => { is_binary = true; little_endian = true; }
                        "binary_big_endian" => { is_binary = true; little_endian = false; }
                        _ => {}
                    }
                }
            } else if trimmed.starts_with("element vertex") {
                in_vertex = true;
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    vertex_count = parts[2].parse().unwrap_or(0);
                }
            } else if trimmed.starts_with("element") {
                in_vertex = false;
            } else if trimmed.starts_with("property") && in_vertex {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    properties.push(PLYProperty {
                        name: parts[2].to_string(),
                        dtype: parts[1].to_string(),
                    });
                }
            } else if trimmed == "end_header" {
                break;
            }
        }

        Ok(PLYHeader {
            is_binary,
            little_endian,
            vertex_count,
            properties,
            header_bytes,
        })
    }

    fn read_ply_binary<R: Read>(
        &self,
        reader: &mut R,
        header: &PLYHeader,
        points: &mut Vec<Point3D>,
    ) -> Result<()> {
        for _ in 0..header.vertex_count {
            let mut x = 0.0f64;
            let mut y = 0.0f64;
            let mut z = 0.0f64;
            let mut nx = None;
            let mut ny = None;
            let mut nz = None;
            let mut r = None;
            let mut g = None;
            let mut b = None;

            for prop in &header.properties {
                match prop.name.as_str() {
                    "x" => x = self.read_f64_binary(reader, &prop.dtype, header.little_endian)?,
                    "y" => y = self.read_f64_binary(reader, &prop.dtype, header.little_endian)?,
                    "z" => z = self.read_f64_binary(reader, &prop.dtype, header.little_endian)?,
                    "nx" => nx = Some(self.read_f64_binary(reader, &prop.dtype, header.little_endian)?),
                    "ny" => ny = Some(self.read_f64_binary(reader, &prop.dtype, header.little_endian)?),
                    "nz" => nz = Some(self.read_f64_binary(reader, &prop.dtype, header.little_endian)?),
                    "red" | "r" => r = Some(self.read_u8_binary(reader, &prop.dtype, header.little_endian)?),
                    "green" | "g" => g = Some(self.read_u8_binary(reader, &prop.dtype, header.little_endian)?),
                    "blue" | "b" => b = Some(self.read_u8_binary(reader, &prop.dtype, header.little_endian)?),
                    _ => { self.skip_binary(reader, &prop.dtype, header.little_endian)?; }
                }
            }

            let mut point = Point3D::new(x, y, z);
            if let (Some(nxv), Some(nyv), Some(nzv)) = (nx, ny, nz) {
                point.normal = Some(nalgebra::Vector3::new(nxv, nyv, nzv));
            }
            if let (Some(rv), Some(gv), Some(bv)) = (r, g, b) {
                point.color = Some(Color::new(rv, gv, bv));
            }
            points.push(point);
        }
        Ok(())
    }

    fn read_ply_ascii<R: BufRead>(
        &self,
        reader: &mut R,
        header: &PLYHeader,
        points: &mut Vec<Point3D>,
    ) -> Result<()> {
        for _ in 0..header.vertex_count {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();

            let mut x = 0.0f64;
            let mut y = 0.0f64;
            let mut z = 0.0f64;
            let mut nx = None;
            let mut ny = None;
            let mut nz = None;
            let mut r = None;
            let mut g = None;
            let mut b = None;

            for (i, prop) in header.properties.iter().enumerate() {
                if i >= parts.len() { break; }
                let val = parts[i];
                match prop.name.as_str() {
                    "x" => x = val.parse().unwrap_or(0.0),
                    "y" => y = val.parse().unwrap_or(0.0),
                    "z" => z = val.parse().unwrap_or(0.0),
                    "nx" => nx = Some(val.parse().unwrap_or(0.0)),
                    "ny" => ny = Some(val.parse().unwrap_or(0.0)),
                    "nz" => nz = Some(val.parse().unwrap_or(0.0)),
                    "red" | "r" => r = Some(val.parse().unwrap_or(255)),
                    "green" | "g" => g = Some(val.parse().unwrap_or(255)),
                    "blue" | "b" => b = Some(val.parse().unwrap_or(255)),
                    _ => {}
                }
            }

            let mut point = Point3D::new(x, y, z);
            if let (Some(nxv), Some(nyv), Some(nzv)) = (nx, ny, nz) {
                point.normal = Some(nalgebra::Vector3::new(nxv, nyv, nzv));
            }
            if let (Some(rv), Some(gv), Some(bv)) = (r, g, b) {
                point.color = Some(Color::new(rv, gv, bv));
            }
            points.push(point);
        }
        Ok(())
    }

    fn read_f64_binary<R: Read>(&self, r: &mut R, dtype: &str, le: bool) -> Result<f64> {
        match dtype {
            "float" | "float32" => {
                if le { Ok(r.read_f32::<LittleEndian>()? as f64) } else { Ok(r.read_f32::<byteorder::BigEndian>()? as f64) }
            }
            "double" | "float64" => {
                if le { Ok(r.read_f64::<LittleEndian>()?) } else { Ok(r.read_f64::<byteorder::BigEndian>()?) }
            }
            _ => Ok(0.0),
        }
    }

    fn read_u8_binary<R: Read>(&self, r: &mut R, dtype: &str, _le: bool) -> Result<u8> {
        match dtype {
            "uchar" | "uint8" => Ok(r.read_u8()?),
            "float" | "float32" | "double" => {
                let v = r.read_f32::<LittleEndian>()?;
                Ok((v * 255.0).min(255.0).max(0.0) as u8)
            }
            _ => Ok(r.read_u8()?),
        }
    }

    fn skip_binary<R: Read>(&self, r: &mut R, dtype: &str, _le: bool) -> Result<()> {
        let n = match dtype {
            "char" | "uchar" | "int8" | "uint8" => 1usize,
            "short" | "ushort" | "int16" | "uint16" => 2,
            "int" | "uint" | "int32" | "uint32" | "float" | "float32" => 4,
            "double" | "float64" | "int64" | "uint64" => 8,
            _ => 1,
        };
        let mut buf = vec![0u8; n];
        r.read_exact(&mut buf)?;
        Ok(())
    }

    fn read_ply_streaming<F>(&self, path: &Path, mut cb: F) -> Result<usize>
    where F: FnMut(Point3D) -> Result<()> {
        let pc = self.read_ply_internal(path)?;
        let n = pc.len();
        for p in pc {
            cb(p)?;
        }
        Ok(n)
    }

    fn read_pcd(&self, path: &Path) -> Result<PointCloud> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let header = self.parse_pcd_header(&mut reader)?;
        let mut points = Vec::with_capacity(header.points as usize);

        if header.data_type == "binary" || header.data_type == "binary_compressed" {
            self.read_pcd_binary(&mut reader, &header, &mut points)?;
        } else {
            self.read_pcd_ascii(&mut reader, &header, &mut points)?;
        }

        Ok(PointCloud::from_points(points))
    }

    fn parse_pcd_header<R: BufRead>(&self, reader: &mut R) -> Result<PCDHeader> {
        let mut line = String::new();
        let mut fields: Vec<String> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        let mut types: Vec<char> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let mut width = 0;
        let mut height = 0;
        let mut points = 0;
        let mut data_type = "ascii".to_string();

        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            if trimmed.starts_with("DATA") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    data_type = parts[1].to_string();
                }
                break;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            match parts.get(0) {
                Some(&"FIELDS") => { fields = parts[1..].iter().map(|s| s.to_string()).collect(); }
                Some(&"SIZE") => { sizes = parts[1..].iter().filter_map(|s| s.parse().ok()).collect(); }
                Some(&"TYPE") => { types = parts[1..].iter().filter_map(|s| s.chars().next()).collect(); }
                Some(&"COUNT") => { counts = parts[1..].iter().filter_map(|s| s.parse().ok()).collect(); }
                Some(&"WIDTH") => { width = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0); }
                Some(&"HEIGHT") => { height = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0); }
                Some(&"POINTS") => { points = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0); }
                _ => {}
            }
        }

        if counts.is_empty() {
            counts = vec![1; fields.len()];
        }

        Ok(PCDHeader {
            fields, sizes, types, counts, width, height, points, data_type
        })
    }

    fn read_pcd_ascii<R: BufRead>(&self, reader: &mut R, header: &PCDHeader, points: &mut Vec<Point3D>) -> Result<()> {
        for _ in 0..header.points {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 { break; }
            let vals: Vec<f64> = line.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            let mut x = 0.0; let mut y = 0.0; let mut z = 0.0;
            let mut nx = None; let mut ny = None; let mut nz = None;
            let mut r = None; let mut g = None; let mut b = None;
            let mut vi = 0;

            for (i, field) in header.fields.iter().enumerate() {
                for _ in 0..*header.counts.get(i).unwrap_or(&1) {
                    if vi >= vals.len() { break; }
                    match field.as_str() {
                        "x" => x = vals[vi],
                        "y" => y = vals[vi],
                        "z" => z = vals[vi],
                        "normal_x" => nx = Some(vals[vi]),
                        "normal_y" => ny = Some(vals[vi]),
                        "normal_z" => nz = Some(vals[vi]),
                        "rgb" | "rgba" => {
                            let rgb = vals[vi] as u32;
                            r = Some(((rgb >> 16) & 0xFF) as u8);
                            g = Some(((rgb >> 8) & 0xFF) as u8);
                            b = Some((rgb & 0xFF) as u8);
                        }
                        _ => {}
                    }
                    vi += 1;
                }
            }

            let mut point = Point3D::new(x, y, z);
            if let (Some(nxv), Some(nyv), Some(nzv)) = (nx, ny, nz) {
                point.normal = Some(nalgebra::Vector3::new(nxv, nyv, nzv));
            }
            if let (Some(rv), Some(gv), Some(bv)) = (r, g, b) {
                point.color = Some(Color::new(rv, gv, bv));
            }
            points.push(point);
        }
        Ok(())
    }

    fn read_pcd_binary<R: Read>(&self, reader: &mut R, header: &PCDHeader, points: &mut Vec<Point3D>) -> Result<()> {
        for _ in 0..header.points {
            let mut x = 0.0f32; let mut y = 0.0f32; let mut z = 0.0f32;
            let mut nx = None; let mut ny = None; let mut nz = None;
            let mut r = None; let mut g = None; let mut b = None;

            for (i, field) in header.fields.iter().enumerate() {
                for _ in 0..*header.counts.get(i).unwrap_or(&1) {
                    match field.as_str() {
                        "x" => x = reader.read_f32::<LittleEndian>()?,
                        "y" => y = reader.read_f32::<LittleEndian>()?,
                        "z" => z = reader.read_f32::<LittleEndian>()?,
                        "normal_x" => nx = Some(reader.read_f32::<LittleEndian>()? as f64),
                        "normal_y" => ny = Some(reader.read_f32::<LittleEndian>()? as f64),
                        "normal_z" => nz = Some(reader.read_f32::<LittleEndian>()? as f64),
                        "rgb" | "rgba" => {
                            let rgb_val = reader.read_f32::<LittleEndian>()?;
                            let rgb = rgb_val.to_bits();
                            r = Some(((rgb >> 16) & 0xFF) as u8);
                            g = Some(((rgb >> 8) & 0xFF) as u8);
                            b = Some((rgb & 0xFF) as u8);
                        }
                        _ => {
                            let size = *header.sizes.get(i).unwrap_or(&4);
                            let mut buf = vec![0u8; size];
                            reader.read_exact(&mut buf)?;
                        }
                    }
                }
            }

            let mut point = Point3D::new(x as f64, y as f64, z as f64);
            if let (Some(nxv), Some(nyv), Some(nzv)) = (nx, ny, nz) {
                point.normal = Some(nalgebra::Vector3::new(nxv, nyv, nzv));
            }
            if let (Some(rv), Some(gv), Some(bv)) = (r, g, b) {
                point.color = Some(Color::new(rv, gv, bv));
            }
            points.push(point);
        }
        Ok(())
    }

    fn read_pcd_streaming<F>(&self, path: &Path, mut cb: F) -> Result<usize>
    where F: FnMut(Point3D) -> Result<()> {
        let pc = self.read_pcd(path)?;
        let n = pc.len();
        for p in pc {
            cb(p)?;
        }
        Ok(n)
    }

    fn read_las(&self, path: &Path) -> Result<PointCloud> {
        use std::io::Seek;
        let mut file = File::open(path)?;
        let mut reader = BufReader::new(&mut file);
        let header = self.parse_las_header(&mut reader)?;
        drop(reader);
        file.seek(std::io::SeekFrom::Start(header.point_data_start as u64))?;
        let mut reader = BufReader::new(&mut file);
        let mut points = Vec::with_capacity(header.point_count as usize);
        self.read_las_points(&mut reader, &header, &mut points)?;
        Ok(PointCloud::from_points(points))
    }

    fn parse_las_header<R: Read>(&self, reader: &mut R) -> Result<LASHeader> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"LASF" {
            return Err(PointCloudError::ParseError("无效的LAS文件头".to_string()));
        }

        let _file_source_id = reader.read_u16::<LittleEndian>()?;
        let mut reserved_bytes = [0u8; 2];
        reader.read_exact(&mut reserved_bytes)?;
        let mut project_guid = [0u8; 16];
        reader.read_exact(&mut project_guid)?;

        let version_major = reader.read_u8()?;
        let version_minor = reader.read_u8()?;

        let mut system_id = [0u8; 32];
        reader.read_exact(&mut system_id)?;
        let mut gen_software = [0u8; 32];
        reader.read_exact(&mut gen_software)?;

        let _creation_day = reader.read_u16::<LittleEndian>()?;
        let _creation_year = reader.read_u16::<LittleEndian>()?;
        let _header_size = reader.read_u16::<LittleEndian>()?;
        let point_data_start = reader.read_u32::<LittleEndian>()?;

        let _num_vlr = reader.read_u32::<LittleEndian>()?;
        let point_data_format = reader.read_u8()?;
        let point_record_length = reader.read_u16::<LittleEndian>()?;

        let legacy_count = reader.read_u32::<LittleEndian>()?;
        let mut legacy_n_returns = [0u32; 5];
        for i in 0..5 {
            legacy_n_returns[i] = reader.read_u32::<LittleEndian>()?;
        }

        let x_scale = reader.read_f64::<LittleEndian>()?;
        let y_scale = reader.read_f64::<LittleEndian>()?;
        let z_scale = reader.read_f64::<LittleEndian>()?;
        let x_offset = reader.read_f64::<LittleEndian>()?;
        let y_offset = reader.read_f64::<LittleEndian>()?;
        let z_offset = reader.read_f64::<LittleEndian>()?;
        let _max_x = reader.read_f64::<LittleEndian>()?;
        let _min_x = reader.read_f64::<LittleEndian>()?;
        let _max_y = reader.read_f64::<LittleEndian>()?;
        let _min_y = reader.read_f64::<LittleEndian>()?;
        let _max_z = reader.read_f64::<LittleEndian>()?;
        let _min_z = reader.read_f64::<LittleEndian>()?;

        let mut point_count = legacy_count as u64;
        if version_major >= 1 && version_minor >= 3 {
            let mut waveform_data_start = [0u8; 8];
            reader.read_exact(&mut waveform_data_start)?;
            let _first_extra_vlr = reader.read_u64::<LittleEndian>()?;
            let _num_extra_vlr = reader.read_u32::<LittleEndian>()?;
            point_count = reader.read_u64::<LittleEndian>()?;
        }

        Ok(LASHeader {
            version_major,
            version_minor,
            point_data_format,
            point_data_start,
            point_count,
            scale: [x_scale, y_scale, z_scale],
            offset: [x_offset, y_offset, z_offset],
            point_record_length,
        })
    }

    fn read_las_points<R: Read>(&self, reader: &mut R, header: &LASHeader, points: &mut Vec<Point3D>) -> Result<()> {
        for _ in 0..header.point_count {
            let xi = reader.read_i32::<LittleEndian>()?;
            let yi = reader.read_i32::<LittleEndian>()?;
            let zi = reader.read_i32::<LittleEndian>()?;

            let x = (xi as f64) * header.scale[0] + header.offset[0];
            let y = (yi as f64) * header.scale[1] + header.offset[1];
            let z = (zi as f64) * header.scale[2] + header.offset[2];

            let mut point = Point3D::new(x, y, z);

            match header.point_data_format {
                0 => {
                    let intensity = reader.read_u16::<LittleEndian>()?;
                    let _return_byte = reader.read_u8()?;
                    let _classification = reader.read_u8()?;
                    let _scan_angle = reader.read_i8()?;
                    let _user_data = reader.read_u8()?;
                    let _point_src_id = reader.read_u16::<LittleEndian>()?;
                    point.intensity = Some(intensity as f64 / 65535.0);
                }
                2 => {
                    let intensity = reader.read_u16::<LittleEndian>()?;
                    let _return_byte = reader.read_u8()?;
                    let _classification = reader.read_u8()?;
                    let _scan_angle = reader.read_i8()?;
                    let _user_data = reader.read_u8()?;
                    let _point_src_id = reader.read_u16::<LittleEndian>()?;
                    let r = reader.read_u16::<LittleEndian>()?;
                    let g = reader.read_u16::<LittleEndian>()?;
                    let b = reader.read_u16::<LittleEndian>()?;
                    point.intensity = Some(intensity as f64 / 65535.0);
                    point.color = Some(Color::new(
                        ((r as f64 / 65535.0) * 255.0) as u8,
                        ((g as f64 / 65535.0) * 255.0) as u8,
                        ((b as f64 / 65535.0) * 255.0) as u8,
                    ));
                }
                3 => {
                    let intensity = reader.read_u16::<LittleEndian>()?;
                    let _return_byte = reader.read_u8()?;
                    let _classification = reader.read_u8()?;
                    let _scan_angle = reader.read_i8()?;
                    let _user_data = reader.read_u8()?;
                    let _point_src_id = reader.read_u16::<LittleEndian>()?;
                    let _gps_time = reader.read_f64::<LittleEndian>()?;
                    let r = reader.read_u16::<LittleEndian>()?;
                    let g = reader.read_u16::<LittleEndian>()?;
                    let b = reader.read_u16::<LittleEndian>()?;
                    point.intensity = Some(intensity as f64 / 65535.0);
                    point.color = Some(Color::new(
                        ((r as f64 / 65535.0) * 255.0) as u8,
                        ((g as f64 / 65535.0) * 255.0) as u8,
                        ((b as f64 / 65535.0) * 255.0) as u8,
                    ));
                }
                _ => {
                    let skip = header.point_record_length as i64 - 26;
                    if skip > 0 {
                        let mut buf = vec![0u8; skip as usize];
                        reader.read_exact(&mut buf)?;
                    }
                }
            }
            points.push(point);
        }
        Ok(())
    }

    fn read_las_streaming<F>(&self, path: &Path, mut cb: F) -> Result<usize>
    where F: FnMut(Point3D) -> Result<()> {
        let pc = self.read_las(path)?;
        let n = pc.len();
        for p in pc {
            cb(p)?;
        }
        Ok(n)
    }

    fn read_xyz(&self, path: &Path) -> Result<PointCloud> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let estimated = (metadata.len() as usize / 50).max(1024);
        let reader = BufReader::new(file);
        let mut points = Vec::with_capacity(estimated);

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 3 { continue; }

            let x: f64 = match parts[0].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let y: f64 = match parts[1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let z: f64 = match parts[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let mut point = Point3D::new(x, y, z);

            if parts.len() >= 6 {
                let a: f64 = parts[3].parse().unwrap_or(0.0);
                let bb: f64 = parts[4].parse().unwrap_or(0.0);
                let c: f64 = parts[5].parse().unwrap_or(0.0);
                let are_normals = a.abs() <= 1.0 && bb.abs() <= 1.0 && c.abs() <= 1.0
                    && ((a * a + bb * bb + c * c).sqrt() - 1.0).abs() < 0.2;
                if are_normals {
                    point.normal = Some(nalgebra::Vector3::new(a, bb, c));
                } else {
                    let r = if a <= 1.0 { (a * 255.0) as u8 } else { a as u8 };
                    let g = if bb <= 1.0 { (bb * 255.0) as u8 } else { bb as u8 };
                    let b_val = if c <= 1.0 { (c * 255.0) as u8 } else { c as u8 };
                    point.color = Some(Color::new(r, g, b_val));
                }
            }
            points.push(point);
        }
        Ok(PointCloud::from_points(points))
    }

    fn read_xyz_streaming<F>(&self, path: &Path, mut cb: F) -> Result<usize>
    where F: FnMut(Point3D) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 3 { continue; }
            let x: f64 = match parts[0].parse() { Ok(v) => v, Err(_) => continue };
            let y: f64 = match parts[1].parse() { Ok(v) => v, Err(_) => continue };
            let z: f64 = match parts[2].parse() { Ok(v) => v, Err(_) => continue };
            let mut point = Point3D::new(x, y, z);

            if parts.len() >= 6 {
                let a: f64 = parts[3].parse().unwrap_or(0.0);
                let bb: f64 = parts[4].parse().unwrap_or(0.0);
                let c: f64 = parts[5].parse().unwrap_or(0.0);
                let are_normals = a.abs() <= 1.0 && bb.abs() <= 1.0 && c.abs() <= 1.0
                    && ((a * a + bb * bb + c * c).sqrt() - 1.0).abs() < 0.2;
                if are_normals {
                    point.normal = Some(nalgebra::Vector3::new(a, bb, c));
                } else {
                    let r = if a <= 1.0 { (a * 255.0) as u8 } else { a as u8 };
                    let g = if bb <= 1.0 { (bb * 255.0) as u8 } else { bb as u8 };
                    let b_val = if c <= 1.0 { (c * 255.0) as u8 } else { c as u8 };
                    point.color = Some(Color::new(r, g, b_val));
                }
            }
            cb(point)?;
            count += 1;
        }
        Ok(count)
    }
}

pub struct MeshWriter;

impl MeshWriter {
    pub fn new() -> Self { MeshWriter }

    pub fn write(&self, mesh: &Mesh, path: &Path) -> Result<()> {
        let format = crate::utils::detect_mesh_format(path)?;
        match format {
            MeshFormat::PLY => self.write_ply(mesh, path),
            MeshFormat::OBJ => self.write_obj(mesh, path),
            MeshFormat::STL => self.write_stl(mesh, path),
        }
    }

    pub fn write_ply(&self, mesh: &Mesh, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "ply")?;
        writeln!(writer, "format binary_little_endian 1.0")?;
        writeln!(writer, "comment Generated by pointcloud-cli")?;
        writeln!(writer, "element vertex {}", mesh.vertex_count())?;
        writeln!(writer, "property float64 x")?;
        writeln!(writer, "property float64 y")?;
        writeln!(writer, "property float64 z")?;

        let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());
        let has_colors = mesh.vertices.iter().any(|v| v.color.is_some());
        if has_normals {
            writeln!(writer, "property float64 nx")?;
            writeln!(writer, "property float64 ny")?;
            writeln!(writer, "property float64 nz")?;
        }
        if has_colors {
            writeln!(writer, "property uchar red")?;
            writeln!(writer, "property uchar green")?;
            writeln!(writer, "property uchar blue")?;
        }
        writeln!(writer, "element face {}", mesh.face_count())?;
        writeln!(writer, "property list uchar int vertex_indices")?;
        writeln!(writer, "end_header")?;

        for v in &mesh.vertices {
            writer.write_f64::<LittleEndian>(v.position.x)?;
            writer.write_f64::<LittleEndian>(v.position.y)?;
            writer.write_f64::<LittleEndian>(v.position.z)?;
            if has_normals {
                let n = v.normal.unwrap_or(nalgebra::Vector3::new(0.0, 0.0, 1.0));
                writer.write_f64::<LittleEndian>(n.x)?;
                writer.write_f64::<LittleEndian>(n.y)?;
                writer.write_f64::<LittleEndian>(n.z)?;
            }
            if has_colors {
                let c = v.color.unwrap_or(Color::white());
                writer.write_u8(c.r)?;
                writer.write_u8(c.g)?;
                writer.write_u8(c.b)?;
            }
        }

        for f in &mesh.faces {
            writer.write_u8(3)?;
            writer.write_i32::<LittleEndian>(f.indices[0] as i32)?;
            writer.write_i32::<LittleEndian>(f.indices[1] as i32)?;
            writer.write_i32::<LittleEndian>(f.indices[2] as i32)?;
        }

        writer.flush()?;
        Ok(())
    }

    pub fn write_obj(&self, mesh: &Mesh, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# Generated by pointcloud-cli")?;
        writeln!(writer, "o mesh")?;

        for v in &mesh.vertices {
            if let Some(c) = v.color {
                writeln!(writer, "v {:.6} {:.6} {:.6} {:.4} {:.4} {:.4}",
                    v.position.x, v.position.y, v.position.z,
                    c.r as f64 / 255.0, c.g as f64 / 255.0, c.b as f64 / 255.0
                )?;
            } else {
                writeln!(writer, "v {:.6} {:.6} {:.6}", v.position.x, v.position.y, v.position.z)?;
            }
        }

        for v in &mesh.vertices {
            if let Some(n) = v.normal {
                writeln!(writer, "vn {:.6} {:.6} {:.6}", n.x, n.y, n.z)?;
            }
        }

        let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());
        for f in &mesh.faces {
            if has_normals {
                writeln!(writer, "f {}//{} {}//{} {}//{}",
                    f.indices[0] + 1, f.indices[0] + 1,
                    f.indices[1] + 1, f.indices[1] + 1,
                    f.indices[2] + 1, f.indices[2] + 1
                )?;
            } else {
                writeln!(writer, "f {} {} {}",
                    f.indices[0] + 1, f.indices[1] + 1, f.indices[2] + 1
                )?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_stl(&self, mesh: &Mesh, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "solid mesh")?;

        for f in &mesh.faces {
            let n = {
                let v0 = &mesh.vertices[f.indices[0]].position;
                let v1 = &mesh.vertices[f.indices[1]].position;
                let v2 = &mesh.vertices[f.indices[2]].position;
                let e1 = v1 - v0;
                let e2 = v2 - v0;
                let n = e1.cross(&e2);
                let len = n.norm();
                if len < 1e-15 {
                    nalgebra::Vector3::new(0.0, 0.0, 1.0)
                } else {
                    n / len
                }
            };
            writeln!(writer, "  facet normal {:.6} {:.6} {:.6}", n.x, n.y, n.z)?;
            writeln!(writer, "    outer loop")?;
            for i in 0..3 {
                let v = &mesh.vertices[f.indices[i]].position;
                writeln!(writer, "      vertex {:.6} {:.6} {:.6}", v.x, v.y, v.z)?;
            }
            writeln!(writer, "    endloop")?;
            writeln!(writer, "  endfacet")?;
        }
        writeln!(writer, "endsolid mesh")?;
        writer.flush()?;
        Ok(())
    }
}

pub fn write_point_cloud_ply(pc: &PointCloud, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "comment Generated by pointcloud-cli")?;
    writeln!(writer, "element vertex {}", pc.len())?;
    writeln!(writer, "property float64 x")?;
    writeln!(writer, "property float64 y")?;
    writeln!(writer, "property float64 z")?;
    if pc.has_normals() {
        writeln!(writer, "property float64 nx")?;
        writeln!(writer, "property float64 ny")?;
        writeln!(writer, "property float64 nz")?;
    }
    if pc.has_colors() {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
    }
    writeln!(writer, "end_header")?;

    let has_normals = pc.has_normals();
    let has_colors = pc.has_colors();
    for p in &pc.points {
        writer.write_f64::<LittleEndian>(p.position.x)?;
        writer.write_f64::<LittleEndian>(p.position.y)?;
        writer.write_f64::<LittleEndian>(p.position.z)?;
        if has_normals {
            let n = p.normal.unwrap_or(nalgebra::Vector3::new(0.0, 0.0, 1.0));
            writer.write_f64::<LittleEndian>(n.x)?;
            writer.write_f64::<LittleEndian>(n.y)?;
            writer.write_f64::<LittleEndian>(n.z)?;
        }
        if has_colors {
            let c = p.color.unwrap_or(Color::white());
            writer.write_u8(c.r)?;
            writer.write_u8(c.g)?;
            writer.write_u8(c.b)?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub fn find_point_cloud_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    if !dir.is_dir() {
        return Err(PointCloudError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotADirectory,
            format!("不是目录: {}", dir.display())
        )));
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Ok(_) = crate::utils::detect_file_format(&path) {
                result.push(path);
            }
        }
    }
    result.sort();
    Ok(result)
}

pub struct MeshReader;

impl MeshReader {
    pub fn new() -> Self { MeshReader }

    pub fn read(&self, path: &Path) -> Result<Mesh> {
        let format = crate::utils::detect_mesh_format(path)?;
        match format {
            MeshFormat::PLY => self.read_ply(path),
            MeshFormat::OBJ => self.read_obj(path),
            MeshFormat::STL => self.read_stl(path),
        }
    }

    pub fn read_ply(&self, path: &Path) -> Result<Mesh> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut line = String::new();
        let mut format_str = "ascii".to_string();
        let mut vertex_count = 0usize;
        let mut face_count = 0usize;
        let mut vertex_properties: Vec<String> = Vec::new();
        let mut in_vertex = false;
        let mut in_face = false;

        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 { break; }
            let trimmed = line.trim();

            if trimmed == "ply" { continue; }
            if trimmed.starts_with("format") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    format_str = parts[1].to_string();
                }
            } else if trimmed.starts_with("element vertex") {
                in_vertex = true;
                in_face = false;
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    vertex_count = parts[2].parse().unwrap_or(0);
                }
            } else if trimmed.starts_with("element face") {
                in_vertex = false;
                in_face = true;
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    face_count = parts[2].parse().unwrap_or(0);
                }
            } else if trimmed.starts_with("element") {
                in_vertex = false;
                in_face = false;
            } else if trimmed.starts_with("property") && in_vertex {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    vertex_properties.push(parts[2].to_string());
                }
            } else if trimmed == "end_header" {
                break;
            }
        }

        let has_x = vertex_properties.iter().any(|p| p == "x");
        let has_y = vertex_properties.iter().any(|p| p == "y");
        let has_z = vertex_properties.iter().any(|p| p == "z");
        let has_nx = vertex_properties.iter().any(|p| p == "nx");
        let has_ny = vertex_properties.iter().any(|p| p == "ny");
        let has_nz = vertex_properties.iter().any(|p| p == "nz");
        let has_red = vertex_properties.iter().any(|p| p == "red");
        let has_green = vertex_properties.iter().any(|p| p == "green");
        let has_blue = vertex_properties.iter().any(|p| p == "blue");

        if !has_x || !has_y || !has_z {
            return Err(PointCloudError::ParseError("PLY文件缺少x/y/z字段".into()));
        }

        let mut vertices: Vec<Vertex> = Vec::with_capacity(vertex_count);
        let mut faces: Vec<TriangleFace> = Vec::with_capacity(face_count);

        match format_str.as_str() {
            "ascii" => {
                for _ in 0..vertex_count {
                    let mut line = String::new();
                    reader.read_line(&mut line)?;
                    let parts: Vec<&str> = line.trim().split_whitespace().collect();
                    let mut pi = 0;

                    let x: f64 = parts.get(pi).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    pi += if has_x { 1 } else { 0 };
                    let y: f64 = parts.get(pi).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    pi += if has_y { 1 } else { 0 };
                    let z: f64 = parts.get(pi).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    pi += if has_z { 1 } else { 0 };

                    let normal = if has_nx && has_ny && has_nz {
                        let nx: f64 = parts.get(pi).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                        let ny: f64 = parts.get(pi + 1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                        let nz: f64 = parts.get(pi + 2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                        Some(nalgebra::Vector3::new(nx, ny, nz))
                    } else { None };

                    let color = if has_red && has_green && has_blue {
                        let cidx = pi + if has_nx && has_ny && has_nz { 3 } else { 0 };
                        let r: u8 = parts.get(cidx).and_then(|s| s.parse().ok()).unwrap_or(255);
                        let g: u8 = parts.get(cidx + 1).and_then(|s| s.parse().ok()).unwrap_or(255);
                        let b: u8 = parts.get(cidx + 2).and_then(|s| s.parse().ok()).unwrap_or(255);
                        Some(Color { r, g, b })
                    } else { None };

                    vertices.push(Vertex {
                        position: nalgebra::Point3::new(x, y, z),
                        normal,
                        tex_coord: None,
                        color,
                    });
                }

                for _ in 0..face_count {
                    let mut line = String::new();
                    reader.read_line(&mut line)?;
                    let parts: Vec<&str> = line.trim().split_whitespace().collect();
                    if parts.len() < 4 { continue; }
                    let count: usize = parts[0].parse().unwrap_or(0);
                    if count == 3 {
                        let i0: usize = parts[1].parse().unwrap_or(0);
                        let i1: usize = parts[2].parse().unwrap_or(0);
                        let i2: usize = parts[3].parse().unwrap_or(0);
                        faces.push(TriangleFace { indices: [i0, i1, i2] });
                    }
                }
            }
            "binary_little_endian" | "binary_big_endian" => {
                use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
                let is_little = format_str == "binary_little_endian";

                for _ in 0..vertex_count {
                    let x = if is_little { reader.read_f64::<LittleEndian>()? } else { reader.read_f64::<BigEndian>()? };
                    let y = if is_little { reader.read_f64::<LittleEndian>()? } else { reader.read_f64::<BigEndian>()? };
                    let z = if is_little { reader.read_f64::<LittleEndian>()? } else { reader.read_f64::<BigEndian>()? };

                    let normal = if has_nx && has_ny && has_nz {
                        let nx = if is_little { reader.read_f64::<LittleEndian>()? } else { reader.read_f64::<BigEndian>()? };
                        let ny = if is_little { reader.read_f64::<LittleEndian>()? } else { reader.read_f64::<BigEndian>()? };
                        let nz = if is_little { reader.read_f64::<LittleEndian>()? } else { reader.read_f64::<BigEndian>()? };
                        Some(nalgebra::Vector3::new(nx, ny, nz))
                    } else { None };

                    let color = if has_red && has_green && has_blue {
                        let r = reader.read_u8()?;
                        let g = reader.read_u8()?;
                        let b = reader.read_u8()?;
                        Some(Color { r, g, b })
                    } else { None };

                    vertices.push(Vertex {
                        position: nalgebra::Point3::new(x, y, z),
                        normal,
                        tex_coord: None,
                        color,
                    });
                }

                for _ in 0..face_count {
                    let count = reader.read_u8()? as usize;
                    if count == 3 {
                        let i0 = if is_little { reader.read_i32::<LittleEndian>()? } else { reader.read_i32::<BigEndian>()? } as usize;
                        let i1 = if is_little { reader.read_i32::<LittleEndian>()? } else { reader.read_i32::<BigEndian>()? } as usize;
                        let i2 = if is_little { reader.read_i32::<LittleEndian>()? } else { reader.read_i32::<BigEndian>()? } as usize;
                        faces.push(TriangleFace { indices: [i0, i1, i2] });
                    } else {
                        for _ in 0..count {
                            if is_little { reader.read_i32::<LittleEndian>()?; } else { reader.read_i32::<BigEndian>()?; }
                        }
                    }
                }
            }
            _ => return Err(PointCloudError::ParseError(format!("不支持的PLY格式: {}", format_str))),
        }

        Ok(Mesh { vertices, faces })
    }

    pub fn read_obj(&self, path: &Path) -> Result<Mesh> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let lines = std::io::BufRead::lines(reader);

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut normals: Vec<nalgebra::Vector3<f64>> = Vec::new();
        let mut faces: Vec<TriangleFace> = Vec::new();
        let mut face_normals: Vec<[usize; 3]> = Vec::new();

        for line_result in lines {
            let line = line_result?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() { continue; }

            match parts[0] {
                "v" => {
                    if parts.len() >= 4 {
                        let x: f64 = parts[1].parse().unwrap_or(0.0);
                        let y: f64 = parts[2].parse().unwrap_or(0.0);
                        let z: f64 = parts[3].parse().unwrap_or(0.0);
                        let color = if parts.len() >= 7 {
                            let r = (parts[4].parse::<f64>().unwrap_or(1.0) * 255.0) as u8;
                            let g = (parts[5].parse::<f64>().unwrap_or(1.0) * 255.0) as u8;
                            let b = (parts[6].parse::<f64>().unwrap_or(1.0) * 255.0) as u8;
                            Some(Color { r, g, b })
                        } else { None };
                        vertices.push(Vertex {
                            position: nalgebra::Point3::new(x, y, z),
                            normal: None,
                            tex_coord: None,
                            color,
                        });
                    }
                }
                "vn" => {
                    if parts.len() >= 4 {
                        let nx: f64 = parts[1].parse().unwrap_or(0.0);
                        let ny: f64 = parts[2].parse().unwrap_or(0.0);
                        let nz: f64 = parts[3].parse().unwrap_or(0.0);
                        normals.push(nalgebra::Vector3::new(nx, ny, nz));
                    }
                }
                "f" => {
                    let mut vert_indices = Vec::new();
                    let mut norm_indices = Vec::new();
                    for part in &parts[1..] {
                        let indices: Vec<&str> = part.split('/').collect();
                        if let Some(v_idx) = indices.first() {
                            if let Ok(idx) = v_idx.parse::<i32>() {
                                let idx_i32 = if idx > 0 { idx - 1 } else { vertices.len() as i32 + idx };
                                vert_indices.push(idx_i32 as usize);
                            }
                        }
                        if indices.len() >= 3 {
                            if let Ok(n_idx) = indices[2].parse::<i32>() {
                                let nidx_i32 = if n_idx > 0 { n_idx - 1 } else { normals.len() as i32 + n_idx };
                                norm_indices.push(nidx_i32 as usize);
                            }
                        }
                    }
                    if vert_indices.len() >= 3 {
                        faces.push(TriangleFace { indices: [vert_indices[0], vert_indices[1], vert_indices[2]] });
                        if norm_indices.len() >= 3 {
                            face_normals.push([norm_indices[0], norm_indices[1], norm_indices[2]]);
                        }
                    }
                }
                _ => {}
            }
        }

        if !normals.is_empty() && !face_normals.is_empty() {
            let mut vertex_normal_indices: Vec<Option<usize>> = vec![None; vertices.len()];
            for (i, face) in faces.iter().enumerate() {
                if i < face_normals.len() {
                    for j in 0..3 {
                        let vi = face.indices[j];
                        let ni = face_normals[i][j];
                        if vertex_normal_indices[vi].is_none() {
                            vertex_normal_indices[vi] = Some(ni);
                        }
                    }
                }
            }
            for (i, v) in vertices.iter_mut().enumerate() {
                if let Some(ni) = vertex_normal_indices[i] {
                    if ni < normals.len() {
                        v.normal = Some(normals[ni]);
                    }
                }
            }
        }

        Ok(Mesh { vertices, faces })
    }

    pub fn read_stl(&self, path: &Path) -> Result<Mesh> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut header_bytes = [0u8; 80];
        if reader.read(&mut header_bytes)? < 80 {
            return Err(PointCloudError::ParseError("STL文件太短".into()));
        }

        let header_str = String::from_utf8_lossy(&header_bytes);
        let is_ascii = header_str.starts_with("solid");

        if is_ascii {
            self.read_stl_ascii(path)
        } else {
            self.read_stl_binary(&mut reader)
        }
    }

    fn read_stl_ascii(&self, path: &Path) -> Result<Mesh> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let lines = std::io::BufRead::lines(reader);

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut faces: Vec<TriangleFace> = Vec::new();
        let mut current_normal: Option<nalgebra::Vector3<f64>> = None;
        let mut face_vertices: Vec<usize> = Vec::new();

        for line_result in lines {
            let line = line_result?.trim().to_lowercase();
            if line.is_empty() { continue; }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() { continue; }

            match parts[0] {
                "facet" => {
                    if parts.len() >= 5 && parts[1] == "normal" {
                        let nx: f64 = parts[2].parse().unwrap_or(0.0);
                        let ny: f64 = parts[3].parse().unwrap_or(0.0);
                        let nz: f64 = parts[4].parse().unwrap_or(0.0);
                        current_normal = Some(nalgebra::Vector3::new(nx, ny, nz));
                    }
                    face_vertices.clear();
                }
                "vertex" => {
                    if parts.len() >= 4 {
                        let x: f64 = parts[1].parse().unwrap_or(0.0);
                        let y: f64 = parts[2].parse().unwrap_or(0.0);
                        let z: f64 = parts[3].parse().unwrap_or(0.0);
                        let v = Vertex {
                            position: nalgebra::Point3::new(x, y, z),
                            normal: current_normal,
                            tex_coord: None,
                            color: None,
                        };
                        vertices.push(v);
                        face_vertices.push(vertices.len() - 1);
                    }
                }
                "endfacet" => {
                    if face_vertices.len() >= 3 {
                        faces.push(TriangleFace { indices: [face_vertices[0], face_vertices[1], face_vertices[2]] });
                    }
                    current_normal = None;
                }
                _ => {}
            }
        }

        Ok(Mesh { vertices, faces })
    }

    fn read_stl_binary<R: std::io::Read>(&self, reader: &mut R) -> Result<Mesh> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let num_triangles = reader.read_u32::<LittleEndian>()? as usize;
        let mut vertices: Vec<Vertex> = Vec::with_capacity(num_triangles * 3);
        let mut faces: Vec<TriangleFace> = Vec::with_capacity(num_triangles);

        for _ in 0..num_triangles {
            let nx = reader.read_f32::<LittleEndian>()? as f64;
            let ny = reader.read_f32::<LittleEndian>()? as f64;
            let nz = reader.read_f32::<LittleEndian>()? as f64;
            let normal = nalgebra::Vector3::new(nx, ny, nz);

            let mut idx = [0usize; 3];
            for j in 0..3 {
                let x = reader.read_f32::<LittleEndian>()? as f64;
                let y = reader.read_f32::<LittleEndian>()? as f64;
                let z = reader.read_f32::<LittleEndian>()? as f64;
                vertices.push(Vertex {
                    position: nalgebra::Point3::new(x, y, z),
                    normal: Some(normal),
                    tex_coord: None,
                    color: None,
                });
                idx[j] = vertices.len() - 1;
            }
            faces.push(TriangleFace { indices: idx });

            let _attr = reader.read_u16::<LittleEndian>()?;
        }

        Ok(Mesh { vertices, faces })
    }
}
