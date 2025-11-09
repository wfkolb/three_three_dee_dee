/// Simple reference mesh generator for testing without external file dependencies

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

pub struct ReferenceMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ReferenceMesh {
    /// Create a simple cube mesh centered at origin with size 2.0 (from -1.0 to 1.0)
    pub fn cube() -> Self {
        // Define 8 vertices of a cube
        let vertices = vec![
            // Front face
            Vertex::new(-1.0, -1.0, 1.0),  // 0
            Vertex::new(1.0, -1.0, 1.0),   // 1
            Vertex::new(1.0, 1.0, 1.0),    // 2
            Vertex::new(-1.0, 1.0, 1.0),   // 3
            // Back face
            Vertex::new(-1.0, -1.0, -1.0), // 4
            Vertex::new(1.0, -1.0, -1.0),  // 5
            Vertex::new(1.0, 1.0, -1.0),   // 6
            Vertex::new(-1.0, 1.0, -1.0),  // 7
        ];

        // Define indices for 12 triangles (2 per face, 6 faces)
        #[rustfmt::skip]
        let indices = vec![
            // Front face
            0, 1, 2,  0, 2, 3,
            // Back face
            5, 4, 7,  5, 7, 6,
            // Top face
            3, 2, 6,  3, 6, 7,
            // Bottom face
            4, 5, 1,  4, 1, 0,
            // Right face
            1, 5, 6,  1, 6, 2,
            // Left face
            4, 0, 3,  4, 3, 7,
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
