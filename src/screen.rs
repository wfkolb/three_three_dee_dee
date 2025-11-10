use crate::fbx_loader::FbxLoader;
use crate::reference_mesh::{ReferenceMesh, Vertex};
use crate::renderer::{Camera, Renderer, Viewport};

/// Screen manages multiple renderers and coordinates rendering
pub struct Screen {
    renderers: Vec<Renderer>,
    frame_number: u32,
}

/// Mesh source that can be either FBX or reference mesh
pub enum MeshSource<'a> {
    Fbx(&'a FbxLoader),
    Reference(&'a ReferenceMesh),
}

impl Screen {
    pub fn new() -> Self {
        Self {
            renderers: Vec::new(),
            frame_number: 0,
        }
    }

    /// Add a renderer with specific viewport and camera settings
    pub fn add_renderer(
        &mut self,
        camera: Camera,
        viewport: Viewport,
        frame_offset: u32,
        model_center: [f32; 3],
        camera_distance: f32,
    ) {
        let renderer = Renderer::new(camera, viewport, frame_offset, model_center, camera_distance);
        self.renderers.push(renderer);
    }

    /// Initialize all renderers with vertex data from mesh source
    pub fn initialize_renderers(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        mesh_source: MeshSource,
    ) {
        // Get vertices and indices based on the source
        let (vertices, indices): (Vec<Vertex>, Vec<u32>) = match mesh_source {
            MeshSource::Reference(reference_mesh) => {
                (reference_mesh.vertices.clone(), reference_mesh.indices.clone())
            }
            MeshSource::Fbx(fbx_loader) => {
                // Get all vertices from FBX
                let verts = fbx_loader.get_all_vertices();

                // Build indices from all meshes
                let mut all_indices = Vec::new();
                let mut vertex_offset = 0u32;

                for mesh in &fbx_loader.meshes {
                    if mesh.indices.is_empty() {
                        // Create sequential indices if none exist
                        for i in 0..mesh.vertices.len() as u32 {
                            all_indices.push(vertex_offset + i);
                        }
                    } else {
                        // Use existing indices with offset
                        for &idx in &mesh.indices {
                            all_indices.push(vertex_offset + idx);
                        }
                    }
                    vertex_offset += mesh.vertices.len() as u32;
                }

                (verts, all_indices)
            }
        };

        let vertices_slice = &vertices;
        let indices_slice = &indices;

        println!("\nScreen: Initializing {} renderers", self.renderers.len());
        println!("  Vertices to send: {}", vertices.len());
        println!("  Indices to send: {}", indices.len());

        // Calculate bounding box
        if !vertices_slice.is_empty() {
            let mut min_x = vertices_slice[0].x;
            let mut max_x = vertices_slice[0].x;
            let mut min_y = vertices_slice[0].y;
            let mut max_y = vertices_slice[0].y;
            let mut min_z = vertices_slice[0].z;
            let mut max_z = vertices_slice[0].z;

            for v in vertices_slice {
                min_x = min_x.min(v.x);
                max_x = max_x.max(v.x);
                min_y = min_y.min(v.y);
                max_y = max_y.max(v.y);
                min_z = min_z.min(v.z);
                max_z = max_z.max(v.z);
            }

            println!("  Model bounds:");
            println!("    X: {} to {}", min_x, max_x);
            println!("    Y: {} to {}", min_y, max_y);
            println!("    Z: {} to {}", min_z, max_z);
            println!("    Center: ({}, {}, {})",
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                (min_z + max_z) / 2.0);
        }

        // Initialize each renderer with the same vertex data
        for (i, renderer) in self.renderers.iter_mut().enumerate() {
            println!("  Initializing renderer {}", i);
            renderer.initialize(device, config, vertices_slice, indices_slice);
        }
    }

    /// Render all viewports
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        queue: &wgpu::Queue,
    ) {
        // Update all cameras
        for renderer in &mut self.renderers {
            renderer.update_camera(queue);
        }

        // Render all viewports
        for renderer in &self.renderers {
            renderer.render(encoder, view, self.frame_number);
        }

        self.frame_number += 1;
    }

    /// Get number of active renderers
    pub fn renderer_count(&self) -> usize {
        self.renderers.len()
    }

    /// Adjust light strength for all renderers
    pub fn adjust_light(&mut self, delta: f32, queue: &wgpu::Queue) {
        for renderer in &mut self.renderers {
            renderer.adjust_light(delta, queue);
        }
    }

    /// Get camera orientation from the first renderer (if any)
    pub fn get_camera_orientation(&self) -> Option<(f32, f32, f32)> {
        self.renderers.first().map(|r| {
            (r.camera.yaw, r.camera.pitch, r.camera.roll)
        })
    }

    /// Get camera distance from the first renderer (if any)
    pub fn get_camera_distance(&self) -> Option<f32> {
        self.renderers.first().map(|r| r.camera_distance)
    }

    /// Adjust camera rotation by delta angles
    pub fn adjust_camera_rotation(&mut self, azimuth_delta: f32, elevation_delta: f32, queue: &wgpu::Queue) {
        for renderer in &mut self.renderers {
            renderer.adjust_camera_rotation(azimuth_delta, elevation_delta, queue);
        }
    }

    /// Adjust camera distance (zoom)
    pub fn adjust_camera_distance(&mut self, distance_delta: f32, queue: &wgpu::Queue) {
        for renderer in &mut self.renderers {
            renderer.adjust_camera_distance(distance_delta, queue);
        }
    }
}
