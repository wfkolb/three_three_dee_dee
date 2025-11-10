use crate::fbx_loader::FbxLoader;
use crate::reference_mesh::{ReferenceMesh, Vertex};
use crate::renderer::{Camera, Renderer, Viewport};

/// Screen manages multiple renderers and coordinates rendering
pub struct Screen {
    renderers: Vec<Renderer>,
    model_center: [f32; 3],
    camera_distance: f32,
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
            model_center: [0.0, 0.0, 0.0],
            camera_distance: 5.0,
        }
    }

    /// Calculate layout configuration for a given number of viewports
    /// Returns a vector where each element is the number of viewports in that row
    fn calculate_layout(count: usize) -> Vec<usize> {
        match count {
            0 => vec![],
            1 => vec![1],
            2 => vec![1, 1],
            3 => vec![1, 2],
            4 => vec![2, 2],
            5 => vec![2, 3],
            6 => vec![3, 3],
            7 => vec![1, 3, 3],
            8 => vec![2, 3, 3],
            9 => vec![3, 3, 3],
            _ => {
                // For counts > 9, create a square-ish grid
                let sqrt = (count as f32).sqrt().ceil() as usize;
                let mut layout = vec![sqrt; count / sqrt];
                let remainder = count % sqrt;
                if remainder > 0 {
                    layout.push(remainder);
                }
                layout
            }
        }
    }

    /// Set model parameters for camera positioning
    pub fn set_model_params(&mut self, model_center: [f32; 3], camera_distance: f32) {
        self.model_center = model_center;
        self.camera_distance = camera_distance;
    }

    /// Recalculate viewport sizes and positions based on window dimensions
    pub fn recalculate_viewports(&mut self, window_width: u32, window_height: u32, queue: &wgpu::Queue) {
        let count = self.renderers.len();
        if count == 0 {
            return;
        }

        let layout = Self::calculate_layout(count);
        let num_rows = layout.len();

        let row_height = window_height / num_rows as u32;
        let mut viewport_index = 0;

        for (row_idx, &viewports_in_row) in layout.iter().enumerate() {
            let viewport_width = window_width / viewports_in_row as u32;
            let y = row_idx as u32 * row_height;

            for col_idx in 0..viewports_in_row {
                if viewport_index >= count {
                    break;
                }

                let x = col_idx as u32 * viewport_width;
                let new_viewport = Viewport {
                    x,
                    y,
                    width: viewport_width,
                    height: row_height,
                };

                self.renderers[viewport_index].update_viewport(new_viewport, queue);
                viewport_index += 1;
            }
        }
    }

    /// Add a renderer with specific viewport and camera settings
    pub fn add_renderer(
        &mut self,
        camera: Camera,
        viewport: Viewport,
        time_offset: f32,
        model_center: [f32; 3],
        camera_distance: f32,
    ) {
        let renderer = Renderer::new(camera, viewport, time_offset, model_center, camera_distance);
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
        elapsed_time: f32,
    ) {
        // Update all cameras
        for renderer in &mut self.renderers {
            renderer.update_camera(queue);
        }

        // Render all viewports
        for renderer in &self.renderers {
            renderer.render(encoder, view, elapsed_time);
        }
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

    /// Add a new viewport (creates a new renderer with shared camera state)
    pub fn add_viewport(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        vertices: &[Vertex],
        indices: &[u32],
        window_width: u32,
        window_height: u32,
        queue: &wgpu::Queue,
        time_offset: f32,
    ) {
        // Get camera state from first renderer if available
        let (camera, viewport) = if let Some(first) = self.renderers.first() {
            (first.camera.clone(), Viewport { x: 0, y: 0, width: 100, height: 100 })
        } else {
            // Create default camera and viewport
            let cam_pos = [
                self.model_center[0],
                self.model_center[1],
                self.model_center[2] + self.camera_distance,
            ];
            let cam_dir = [
                self.model_center[0] - cam_pos[0],
                self.model_center[1] - cam_pos[1],
                self.model_center[2] - cam_pos[2],
            ];
            (
                Camera::new(cam_pos, cam_dir, 0.1, 100.0),
                Viewport { x: 0, y: 0, width: 100, height: 100 }
            )
        };

        // Create new renderer
        let mut renderer = Renderer::new(camera, viewport, time_offset, self.model_center, self.camera_distance);
        renderer.initialize(device, config, vertices, indices);
        self.renderers.push(renderer);

        // Recalculate all viewport positions
        self.recalculate_viewports(window_width, window_height, queue);

        println!("Added viewport with time offset: {:.2}s. Total viewports: {}", time_offset, self.renderers.len());
    }

    /// Remove a viewport
    pub fn remove_viewport(&mut self, window_width: u32, window_height: u32, queue: &wgpu::Queue) {
        if self.renderers.len() > 1 {
            self.renderers.pop();
            self.recalculate_viewports(window_width, window_height, queue);
            println!("Removed viewport. Total viewports: {}", self.renderers.len());
        } else {
            println!("Cannot remove last viewport");
        }
    }
}
