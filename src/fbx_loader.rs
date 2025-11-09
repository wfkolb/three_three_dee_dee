use std::fs::File;
use std::io::BufReader;

/// Represents a 3D vertex with position coordinates
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vertex {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Represents a mesh loaded from an FBX file
#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Manager for loading and parsing FBX files
pub struct FbxLoader {
    pub meshes: Vec<Mesh>,
    pub version: fbx::Version,
}

impl FbxLoader {
    /// Load an FBX file from the given path
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Load the FBX file using the fbx crate
        let fbx_file = fbx::File::read_from(reader)?;

        println!("FBX Version: {:?}", fbx_file.version);
        println!("Number of root nodes: {}", fbx_file.children.len());

        let mut meshes = Vec::new();

        // Parse the FBX structure to extract mesh data
        Self::parse_nodes(&fbx_file.children, &mut meshes);

        Ok(Self {
            meshes,
            version: fbx_file.version,
        })
    }

    /// Recursively parse FBX nodes to find and extract mesh data
    fn parse_nodes(nodes: &[fbx::Node], meshes: &mut Vec<Mesh>) {
        for node in nodes {
            // Look for geometry nodes that contain mesh data
            if node.name == "Geometry" {
                if let Some(mesh) = Self::extract_mesh(node) {
                    meshes.push(mesh);
                }
            }

            // Recursively parse child nodes
            Self::parse_nodes(&node.children, meshes);
        }
    }

    /// Extract mesh data (vertices and indices) from a geometry node
    fn extract_mesh(node: &fbx::Node) -> Option<Mesh> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut name = String::from("Unnamed");

        // Try to get the mesh name from properties
        if node.properties.len() >= 2 {
            if let fbx::Property::String(mesh_name) = &node.properties[1] {
                name = mesh_name.clone();
            }
        }

        // Look through child nodes for vertex and index data
        for child in &node.children {
            match child.name.as_str() {
                "Vertices" => {
                    if let Some(vertex_data) = Self::extract_vertices(child) {
                        vertices = vertex_data;
                    }
                }
                "PolygonVertexIndex" => {
                    if let Some(index_data) = Self::extract_indices(child) {
                        indices = index_data;
                    }
                }
                _ => {}
            }
        }

        if !vertices.is_empty() {
            println!(
                "Found mesh '{}': {} vertices, {} indices",
                name,
                vertices.len(),
                indices.len()
            );
            Some(Mesh {
                name,
                vertices,
                indices,
            })
        } else {
            None
        }
    }

    /// Extract vertex positions from a Vertices node
    fn extract_vertices(node: &fbx::Node) -> Option<Vec<Vertex>> {
        if node.properties.is_empty() {
            return None;
        }

        // Vertices are stored as a flat array of f64 values (x, y, z, x, y, z, ...)
        match &node.properties[0] {
            fbx::Property::F64Array(coords) => {
                let mut vertices = Vec::new();
                for chunk in coords.chunks(3) {
                    if chunk.len() == 3 {
                        // FBX uses Z-up right-handed coordinate system
                        // Convert to Y-up left-handed (wgpu/WebGPU convention)
                        // FBX: X right, Y forward, Z up
                        // wgpu: X right, Y up, Z forward
                        let fbx_x = chunk[0] as f32;
                        let fbx_y = chunk[1] as f32;
                        let fbx_z = chunk[2] as f32;

                        // Transform: X stays X, Z becomes Y (up), -Y becomes Z (forward)
                        vertices.push(Vertex::new(
                            fbx_x,      // X stays the same
                            fbx_z,      // Z becomes Y (up)
                            -fbx_y,     // -Y becomes Z (forward, negated for left-handed)
                        ));
                    }
                }
                Some(vertices)
            }
            _ => None,
        }
    }

    /// Extract polygon vertex indices from a PolygonVertexIndex node
    fn extract_indices(node: &fbx::Node) -> Option<Vec<u32>> {
        if node.properties.is_empty() {
            return None;
        }

        // Indices are stored as i32 values
        // In FBX, negative indices indicate the last vertex of a polygon
        match &node.properties[0] {
            fbx::Property::I32Array(indices) => {
                let mut result = Vec::new();
                for &idx in indices {
                    // FBX uses negative indices to mark polygon boundaries
                    // Convert negative indices to positive
                    let index = if idx < 0 { -idx - 1 } else { idx };
                    result.push(index as u32);
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get all vertices from all meshes as a flat vector
    pub fn get_all_vertices(&self) -> Vec<Vertex> {
        self.meshes
            .iter()
            .flat_map(|mesh| mesh.vertices.iter().copied())
            .collect()
    }

    /// Get the total number of vertices across all meshes
    pub fn total_vertex_count(&self) -> usize {
        self.meshes.iter().map(|mesh| mesh.vertices.len()).sum()
    }

    /// Get the total number of indices across all meshes
    pub fn total_index_count(&self) -> usize {
        self.meshes.iter().map(|mesh| mesh.indices.len()).sum()
    }

    /// Calculate the bounding box center of all meshes
    pub fn get_center(&self) -> [f32; 3] {
        let vertices = self.get_all_vertices();
        if vertices.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let mut min_x = vertices[0].x;
        let mut max_x = vertices[0].x;
        let mut min_y = vertices[0].y;
        let mut max_y = vertices[0].y;
        let mut min_z = vertices[0].z;
        let mut max_z = vertices[0].z;

        for v in &vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
            min_z = min_z.min(v.z);
            max_z = max_z.max(v.z);
        }

        [
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        ]
    }

    /// Get the maximum dimension (size) of the model for camera distance calculation
    pub fn get_max_dimension(&self) -> f32 {
        let vertices = self.get_all_vertices();
        if vertices.is_empty() {
            return 1.0;
        }

        let mut min_x = vertices[0].x;
        let mut max_x = vertices[0].x;
        let mut min_y = vertices[0].y;
        let mut max_y = vertices[0].y;
        let mut min_z = vertices[0].z;
        let mut max_z = vertices[0].z;

        for v in &vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
            min_z = min_z.min(v.z);
            max_z = max_z.max(v.z);
        }

        let size_x = max_x - min_x;
        let size_y = max_y - min_y;
        let size_z = max_z - min_z;

        size_x.max(size_y).max(size_z)
    }
}
