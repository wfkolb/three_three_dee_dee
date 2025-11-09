use crate::fbx_loader::FbxLoader;
use crate::renderer::{Camera, Renderer, Viewport};

/// Screen manages multiple renderers and coordinates rendering
pub struct Screen {
    renderers: Vec<Renderer>,
    frame_number: u32,
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

    /// Initialize all renderers with vertex data from FBX
    pub fn initialize_renderers(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        fbx_loader: &FbxLoader,
    ) {
        // Get all vertices from all meshes
        let vertices = fbx_loader.get_all_vertices();

        // Get all indices - for simplicity, create sequential indices if none exist
        let indices: Vec<u32> = if fbx_loader.meshes.is_empty() {
            Vec::new()
        } else {
            let mut all_indices = Vec::new();
            let mut vertex_offset = 0u32;

            for mesh in &fbx_loader.meshes {
                if mesh.indices.is_empty() {
                    // If no indices, create them sequentially
                    for i in 0..mesh.vertices.len() as u32 {
                        all_indices.push(vertex_offset + i);
                    }
                    
                } else {
                    // Use existing indices with offset
                    for &idx in &mesh.indices {
                        all_indices.push(vertex_offset + idx);
                    }
                    println!("pushed vetexes")
                }
                vertex_offset += mesh.vertices.len() as u32;
            }
            all_indices
        };

        println!("\nScreen: Initializing {} renderers", self.renderers.len());
        println!("  Vertices to send: {}", vertices.len());
        println!("  Indices to send: {}", indices.len());

        // Calculate bounding box
        if !vertices.is_empty() {
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
            renderer.initialize(device, config, &vertices, &indices);
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
}
