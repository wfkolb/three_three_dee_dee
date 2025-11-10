/// Simple reference mesh generator for testing without external file dependencies

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}

impl Vertex {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, nx: 0.0, ny: 0.0, nz: 0.0 }
    }

    pub fn with_normal(x: f32, y: f32, z: f32, nx: f32, ny: f32, nz: f32) -> Self {
        Self { x, y, z, nx, ny, nz }
    }
}

pub struct ReferenceMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ReferenceMesh {
    /// Create a simple cube mesh centered at origin with size 2.0 (from -1.0 to 1.0)
    /// Each face has its own vertices with proper normals
    pub fn cube() -> Self {
        // Create vertices with proper normals for each face
        // We need 24 vertices (4 per face, 6 faces) to have correct per-face normals
        let vertices = vec![
            // Front face (normal: 0, 0, 1)
            Vertex::with_normal(-1.0, -1.0, 1.0, 0.0, 0.0, 1.0),  // 0
            Vertex::with_normal(1.0, -1.0, 1.0, 0.0, 0.0, 1.0),   // 1
            Vertex::with_normal(1.0, 1.0, 1.0, 0.0, 0.0, 1.0),    // 2
            Vertex::with_normal(-1.0, 1.0, 1.0, 0.0, 0.0, 1.0),   // 3

            // Back face (normal: 0, 0, -1)
            Vertex::with_normal(1.0, -1.0, -1.0, 0.0, 0.0, -1.0), // 4
            Vertex::with_normal(-1.0, -1.0, -1.0, 0.0, 0.0, -1.0),// 5
            Vertex::with_normal(-1.0, 1.0, -1.0, 0.0, 0.0, -1.0), // 6
            Vertex::with_normal(1.0, 1.0, -1.0, 0.0, 0.0, -1.0),  // 7

            // Top face (normal: 0, 1, 0)
            Vertex::with_normal(-1.0, 1.0, 1.0, 0.0, 1.0, 0.0),   // 8
            Vertex::with_normal(1.0, 1.0, 1.0, 0.0, 1.0, 0.0),    // 9
            Vertex::with_normal(1.0, 1.0, -1.0, 0.0, 1.0, 0.0),   // 10
            Vertex::with_normal(-1.0, 1.0, -1.0, 0.0, 1.0, 0.0),  // 11

            // Bottom face (normal: 0, -1, 0)
            Vertex::with_normal(-1.0, -1.0, -1.0, 0.0, -1.0, 0.0),// 12
            Vertex::with_normal(1.0, -1.0, -1.0, 0.0, -1.0, 0.0), // 13
            Vertex::with_normal(1.0, -1.0, 1.0, 0.0, -1.0, 0.0),  // 14
            Vertex::with_normal(-1.0, -1.0, 1.0, 0.0, -1.0, 0.0), // 15

            // Right face (normal: 1, 0, 0)
            Vertex::with_normal(1.0, -1.0, 1.0, 1.0, 0.0, 0.0),   // 16
            Vertex::with_normal(1.0, -1.0, -1.0, 1.0, 0.0, 0.0),  // 17
            Vertex::with_normal(1.0, 1.0, -1.0, 1.0, 0.0, 0.0),   // 18
            Vertex::with_normal(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),    // 19

            // Left face (normal: -1, 0, 0)
            Vertex::with_normal(-1.0, -1.0, -1.0, -1.0, 0.0, 0.0),// 20
            Vertex::with_normal(-1.0, -1.0, 1.0, -1.0, 0.0, 0.0), // 21
            Vertex::with_normal(-1.0, 1.0, 1.0, -1.0, 0.0, 0.0),  // 22
            Vertex::with_normal(-1.0, 1.0, -1.0, -1.0, 0.0, 0.0), // 23
        ];

        // Define indices for 12 triangles (2 per face, 6 faces)
        #[rustfmt::skip]
        let indices = vec![
            // Front face
            0, 1, 2,  0, 2, 3,
            // Back face
            4, 5, 6,  4, 6, 7,
            // Top face
            8, 9, 10,  8, 10, 11,
            // Bottom face
            12, 13, 14,  12, 14, 15,
            // Right face
            16, 17, 18,  16, 18, 19,
            // Left face
            20, 21, 22,  20, 22, 23,
        ];

        Self { vertices, indices }
    }

    /// Calculate the bounding box center
    pub fn get_center(&self) -> [f32; 3] {
        if self.vertices.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let mut min_x = self.vertices[0].x;
        let mut max_x = self.vertices[0].x;
        let mut min_y = self.vertices[0].y;
        let mut max_y = self.vertices[0].y;
        let mut min_z = self.vertices[0].z;
        let mut max_z = self.vertices[0].z;

        for v in &self.vertices {
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

    /// Get the maximum dimension (size) of the mesh
    pub fn get_max_dimension(&self) -> f32 {
        if self.vertices.is_empty() {
            return 1.0;
        }

        let mut min_x = self.vertices[0].x;
        let mut max_x = self.vertices[0].x;
        let mut min_y = self.vertices[0].y;
        let mut max_y = self.vertices[0].y;
        let mut min_z = self.vertices[0].z;
        let mut max_z = self.vertices[0].z;

        for v in &self.vertices {
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
