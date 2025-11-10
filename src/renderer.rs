use crate::reference_mesh::Vertex;
use wgpu::util::DeviceExt;

/// Viewport definition for screen segmentation
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Camera direction and viewing parameters
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub near_plane: f32,
    pub far_plane: f32,
    pub yaw: f32,    // Rotation around Y axis (radians)
    pub pitch: f32,  // Rotation around X axis (radians)
    pub roll: f32,   // Rotation around Z axis (radians)
}

impl Camera {
    pub fn new(position: [f32; 3], direction: [f32; 3], near: f32, far: f32) -> Self {
        Self {
            position,
            direction,
            near_plane: near,
            far_plane: far,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }
}

/// Vertex layout for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl GpuVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Individual renderer that renders to a viewport
pub struct Renderer {
    pub camera: Camera,
    pub viewport: Viewport,
    pub time_offset: f32, // Time offset in seconds
    model_center: [f32; 3],
    pub camera_distance: f32,
    rotation_speed: f32,
    current_angle: f32,
    light_strength: f32,

    // GPU resources
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    edge_index_buffer: Option<wgpu::Buffer>,
    num_edge_indices: u32,
    render_pipeline: Option<wgpu::RenderPipeline>,
    wireframe_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
}

impl Renderer {
    pub fn new(camera: Camera, viewport: Viewport, time_offset: f32, model_center: [f32; 3], camera_distance: f32) -> Self {
        println!("Renderer::new - Initial camera_distance: {}", camera_distance);
        println!("Renderer::new - Initial current_angle: {} rad ({} deg)", std::f32::consts::PI / 2.0, 90.0);
        println!("Renderer::new - Initial pitch: {} rad ({} deg)", camera.pitch, camera.pitch.to_degrees());

        Self {
            camera,
            viewport,
            time_offset,
            model_center,
            camera_distance,
            rotation_speed: 0.00, // radians per frame (slower, smoother rotation)
            current_angle: std::f32::consts::PI / 2.0, // Start at 90° to match initial +Z position
            light_strength: 0.7,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            edge_index_buffer: None,
            num_edge_indices: 0,
            render_pipeline: None,
            wireframe_pipeline: None,
            uniform_buffer: None,
            bind_group: None,
            depth_texture: None,
            depth_view: None,
        }
    }

    /// Update camera position for rotation
    pub fn update_camera(&mut self, queue: &wgpu::Queue) {
        self.current_angle += self.rotation_speed;

        // Calculate camera position using spherical coordinates (supports both auto and manual rotation)
        let horizontal_distance = self.camera_distance * self.camera.pitch.cos();
        let x = self.model_center[0] + horizontal_distance * self.current_angle.cos();
        let y = self.model_center[1] + self.camera_distance * self.camera.pitch.sin();
        let z = self.model_center[2] + horizontal_distance * self.current_angle.sin();

        self.camera.position = [x, y, z];
        self.camera.direction = [
            self.model_center[0] - x,
            self.model_center[1] - y,
            self.model_center[2] - z,
        ];

        // Update camera orientation
        self.camera.yaw = self.current_angle;
        self.camera.roll = 0.0;

        // Update uniforms with new camera position
        let uniforms = self.create_uniforms();
        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    /// Adjust light strength (clamped between 0.0 and 2.0)
    pub fn adjust_light(&mut self, delta: f32, queue: &wgpu::Queue) {
        self.light_strength = (self.light_strength + delta).clamp(0.0, 2.0);
        let uniforms = self.create_uniforms();
        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    /// Adjust camera rotation manually (azimuth and elevation)
    pub fn adjust_camera_rotation(&mut self, azimuth_delta: f32, elevation_delta: f32, queue: &wgpu::Queue) {
        // Update angles
        self.current_angle += azimuth_delta;
        self.camera.pitch += elevation_delta;

        // Clamp pitch to prevent flipping (keep between -89 and 89 degrees)
        self.camera.pitch = self.camera.pitch.clamp(-1.55, 1.55);

        // Calculate camera position using spherical coordinates
        let horizontal_distance = self.camera_distance * self.camera.pitch.cos();
        let x = self.model_center[0] + horizontal_distance * self.current_angle.cos();
        let y = self.model_center[1] + self.camera_distance * self.camera.pitch.sin();
        let z = self.model_center[2] + horizontal_distance * self.current_angle.sin();

        self.camera.position = [x, y, z];
        self.camera.direction = [
            self.model_center[0] - x,
            self.model_center[1] - y,
            self.model_center[2] - z,
        ];

        // Update camera orientation
        self.camera.yaw = self.current_angle;
        self.camera.roll = 0.0;

        // Update uniforms with new camera position
        let uniforms = self.create_uniforms();
        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    /// Adjust camera distance (zoom in/out)
    pub fn adjust_camera_distance(&mut self, distance_delta: f32, queue: &wgpu::Queue) {
        // Adjust distance with min/max constraints
        self.camera_distance = (self.camera_distance + distance_delta).clamp(-5000.0, 5000.0);

        // Recalculate camera position with new distance
        let horizontal_distance = self.camera_distance * self.camera.pitch.cos();
        let x = self.model_center[0] + horizontal_distance * self.current_angle.cos();
        let y = self.model_center[1] + self.camera_distance * self.camera.pitch.sin();
        let z = self.model_center[2] + horizontal_distance * self.current_angle.sin();

        self.camera.position = [x, y, z];
        self.camera.direction = [
            self.model_center[0] - x,
            self.model_center[1] - y,
            self.model_center[2] - z,
        ];

        // Update uniforms
        let uniforms = self.create_uniforms();
        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    /// Update viewport dimensions and recreate resources that depend on it
    pub fn update_viewport(&mut self, new_viewport: Viewport, queue: &wgpu::Queue) {
        self.viewport = new_viewport;

        // Update uniforms (projection matrix depends on aspect ratio)
        let uniforms = self.create_uniforms();
        queue.write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    /// Initialize GPU resources
    pub fn initialize(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        vertices: &[Vertex],
        indices: &[u32],
    ) {
        println!("    Renderer.initialize: Received {} vertices, {} indices", vertices.len(), indices.len());

        // Convert vertices to GPU format
        let gpu_vertices: Vec<GpuVertex> = vertices
            .iter()
            .map(|v| GpuVertex {
                position: [v.x, v.y, v.z],
                normal: [v.nx, v.ny, v.nz],
            })
            .collect();

        // Create vertex buffer
        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&gpu_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );

        // Create index buffer
        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        self.num_indices = indices.len() as u32;

        // Create edge indices for wireframe (draw edges as lines)
        // For a cube with 24 vertices (4 per face), we need to extract unique edges
        let edge_indices = Self::generate_edge_indices(indices);
        self.edge_index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Edge Index Buffer"),
                contents: bytemuck::cast_slice(&edge_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        self.num_edge_indices = edge_indices.len() as u32;

        println!("    Renderer: Created buffers with {} indices, {} edge indices", self.num_indices, self.num_edge_indices);

        // Create uniform buffer for camera matrices
        let uniforms = self.create_uniforms();
        self.uniform_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        ));

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("bind_group_layout"),
        });

        // Create bind group
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_ref().unwrap().as_entire_binding(),
            }],
            label: Some("bind_group"),
        }));

        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_texture = Some(depth_texture);
        self.depth_view = Some(depth_view);

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        self.render_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[GpuVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,  // Disable culling to see both sides
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            }),
        );

        // Create wireframe pipeline for drawing edges
        self.wireframe_pipeline = Some(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Wireframe Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[GpuVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_wireframe"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,  // Don't write depth for wireframe, just test
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            }),
        );
    }

    /// Create uniform data for camera matrices
    fn create_uniforms(&self) -> Uniforms {
        // Create view matrix from camera position and direction
        let eye = self.camera.position;
        let target = [
            eye[0] + self.camera.direction[0],
            eye[1] + self.camera.direction[1],
            eye[2] + self.camera.direction[2],
        ];
        let up = [0.0, 1.0, 0.0];

        let view = Self::look_at(eye, target, up);

        // Create perspective projection matrix
        let aspect = self.viewport.width as f32 / self.viewport.height as f32;
        let fov_y = 45.0_f32.to_radians();
        let projection = Self::perspective(fov_y, aspect, self.camera.near_plane, self.camera.far_plane);

        // Build view-projection matrix (projection * view for standard transform order)
        // Matrices are already in WebGPU column-major format
        let view_proj = Self::multiply_matrices_column_major(projection, view);

        // Debug output on first frame
        static FIRST_CALL: std::sync::Once = std::sync::Once::new();
        FIRST_CALL.call_once(|| {
            println!("\n=== Camera Debug Info ===");
            println!("Model center: [{}, {}, {}]", self.model_center[0], self.model_center[1], self.model_center[2]);
            println!("Camera distance: {}", self.camera_distance);
            println!("Current angle (azimuth): {} rad ({} deg)", self.current_angle, self.current_angle.to_degrees());
            println!("Pitch (elevation): {} rad ({} deg)", self.camera.pitch, self.camera.pitch.to_degrees());
            println!("Eye: [{}, {}, {}]", eye[0], eye[1], eye[2]);
            println!("Target: [{}, {}, {}]", target[0], target[1], target[2]);
            println!("FOV: {} degrees, Aspect: {}", fov_y.to_degrees(), aspect);
            println!("Near: {}, Far: {}", self.camera.near_plane, self.camera.far_plane);
            println!("View matrix:");
            for row in &view {
                println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
            }
            println!("Projection matrix:");
            for row in &projection {
                println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
            }
            println!("View-Projection matrix:");
            for row in &view_proj {
                println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
            }
        });

        Uniforms {
            view_proj,
            light_strength: self.light_strength,
            camera_pos: self.camera.position,
        }
    }

    /// Look-at view matrix (WebGPU column-major format)
    /// Camera at 'eye' looking at 'target' with 'up' direction
    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        // Forward vector (from eye to target)
        let f = Self::normalize([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ]);
        // Right vector
        let s = Self::normalize(Self::cross(f, up));
        // Up vector
        let u = Self::cross(s, f);

        // WebGPU column-major view matrix
        // Each inner array is a column: [col0, col1, col2, col3]
        [
            [s[0], s[1], s[2], 0.0],                                    // Right vector (column 0)
            [u[0], u[1], u[2], 0.0],                                    // Up vector (column 1)
            [-f[0], -f[1], -f[2], 0.0],                                 // Forward vector negated (column 2)
            [-Self::dot(s, eye), -Self::dot(u, eye), Self::dot(f, eye), 1.0],  // Translation (column 3)
        ]
    }

    /// Perspective projection matrix for WebGPU (0 to 1 depth range, column-major)
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let f = 1.0 / (fov_y / 2.0).tan();

        // WebGPU uses 0 to 1 depth range (not -1 to 1 like OpenGL)
        // Column-major format: matrix[column][row]
        // Standard perspective formula for 0-to-1 depth (Vulkan/WebGPU/Metal style)
        let a = far / (near - far);
        let b = (far * near) / (near - far);

        [
            [f / aspect, 0.0, 0.0, 0.0],     // Column 0: X scaling
            [0.0, f, 0.0, 0.0],              // Column 1: Y scaling
            [0.0, 0.0, a, -1.0],             // Column 2: Z transformation + perspective divide trigger
            [0.0, 0.0, b, 0.0],              // Column 3: Z translation
        ]
    }

    /// Generate edge indices from triangle indices for wireframe rendering
    fn generate_edge_indices(indices: &[u32]) -> Vec<u32> {
        let mut edge_indices = Vec::new();

        // For each triangle, add its three edges
        for chunk in indices.chunks(3) {
            if chunk.len() == 3 {
                // Add three edges of the triangle (as line pairs)
                edge_indices.push(chunk[0]);
                edge_indices.push(chunk[1]);

                edge_indices.push(chunk[1]);
                edge_indices.push(chunk[2]);

                edge_indices.push(chunk[2]);
                edge_indices.push(chunk[0]);
            }
        }

        edge_indices
    }

    // Helper math functions
    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            // Return a default up vector for zero-length vectors
            [0.0, 1.0, 0.0]
        }
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Multiply two column-major matrices (for WebGPU format)
    /// In column-major: matrix[col][row], so a[i] is column i
    fn multiply_matrices_column_major(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        // For each output column
        for col in 0..4 {
            // For each output row
            for row in 0..4 {
                // Dot product of row from A with column from B
                for k in 0..4 {
                    result[col][row] += a[k][row] * b[col][k];
                }
            }
        }
        result
    }

    fn transpose(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ]
    }

    /// Render to the specified viewport
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        elapsed_time: f32,
    ) {
        // Apply time offset - this viewport's effective time
        let _viewport_time = elapsed_time + self.time_offset;

        // Note: Currently animations are handled by camera updates, not per-frame in render
        // The time offset is applied during camera updates instead

        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear, preserve other viewports
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        };

        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

        // Set viewport
        render_pass.set_viewport(
            self.viewport.x as f32,
            self.viewport.y as f32,
            self.viewport.width as f32,
            self.viewport.height as f32,
            0.0,
            1.0,
        );

        // Draw solid mesh
        render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.as_ref().unwrap().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

        // Draw wireframe edges on top
        render_pass.set_pipeline(self.wireframe_pipeline.as_ref().unwrap());
        // Bind group and vertex buffer are already set
        render_pass.set_index_buffer(
            self.edge_index_buffer.as_ref().unwrap().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.num_edge_indices, 0, 0..1);
    }
}

/// Uniform data sent to GPU
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    light_strength: f32,
    camera_pos: [f32; 3], // Camera position (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.0001;

    fn assert_vec3_near(a: [f32; 3], b: [f32; 3], epsilon: f32) {
        assert!(
            (a[0] - b[0]).abs() < epsilon &&
            (a[1] - b[1]).abs() < epsilon &&
            (a[2] - b[2]).abs() < epsilon,
            "Vectors not equal: [{}, {}, {}] vs [{}, {}, {}]",
            a[0], a[1], a[2], b[0], b[1], b[2]
        );
    }

    fn assert_matrix_near(a: [[f32; 4]; 4], b: [[f32; 4]; 4], epsilon: f32) {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a[i][j] - b[i][j]).abs() < epsilon,
                    "Matrix element [{},{}] not equal: {} vs {} (diff: {})",
                    i, j, a[i][j], b[i][j], (a[i][j] - b[i][j]).abs()
                );
            }
        }
    }

    #[test]
    fn test_normalize() {
        let v = [3.0, 4.0, 0.0];
        let normalized = Renderer::normalize(v);
        assert_vec3_near(normalized, [0.6, 0.8, 0.0], EPSILON);

        // Check magnitude is 1
        let mag = (normalized[0] * normalized[0] + normalized[1] * normalized[1] + normalized[2] * normalized[2]).sqrt();
        assert!((mag - 1.0).abs() < EPSILON, "Normalized vector magnitude is {}", mag);
    }

    #[test]
    fn test_cross_product() {
        // Test with standard basis vectors
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = Renderer::cross(x, y);
        assert_vec3_near(z, [0.0, 0.0, 1.0], EPSILON);

        // Test reverse gives negative
        let neg_z = Renderer::cross(y, x);
        assert_vec3_near(neg_z, [0.0, 0.0, -1.0], EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = Renderer::dot(a, b);
        assert!((result - 32.0).abs() < EPSILON, "Dot product is {}", result);
    }

    #[test]
    fn test_identity_matrix_multiplication() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let test_matrix = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];

        let result = Renderer::multiply_matrices(identity, test_matrix);
        assert_matrix_near(result, test_matrix, EPSILON);
    }

    #[test]
    fn test_look_at_basic() {
        // Camera at origin looking down negative Z (standard)
        let eye = [0.0, 0.0, 0.0];
        let target = [0.0, 0.0, -1.0];
        let up = [0.0, 1.0, 0.0];

        let view = Renderer::look_at(eye, target, up);

        // The view matrix should transform world coordinates to camera space
        // A point at (0, 0, -1) in world space should map to (0, 0, 1) in camera space
        // (camera looks down -Z, so objects in front are at +Z in view space)

        // Print the matrix for debugging
        println!("Look-at matrix (eye at origin, looking at -Z):");
        for row in &view {
            println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
        }

        // At minimum, the matrix should be valid (not NaN)
        for row in &view {
            for &val in row {
                assert!(!val.is_nan(), "View matrix contains NaN");
            }
        }
    }

    #[test]
    fn test_look_at_camera_offset() {
        // Camera at (5, 0, 0) looking at origin
        let eye = [5.0, 0.0, 0.0];
        let target = [0.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];

        let view = Renderer::look_at(eye, target, up);

        println!("Look-at matrix (eye at (5,0,0), looking at origin):");
        for row in &view {
            println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
        }

        // Check no NaN values
        for row in &view {
            for &val in row {
                assert!(!val.is_nan(), "View matrix contains NaN");
            }
        }
    }

    #[test]
    fn test_perspective_basic() {
        let fov_y = 45.0_f32.to_radians();
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;

        let proj = Renderer::perspective(fov_y, aspect, near, far);

        println!("Perspective matrix (45° FOV, 16:9 aspect):");
        for row in &proj {
            println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
        }

        // Check no NaN values
        for row in &proj {
            for &val in row {
                assert!(!val.is_nan(), "Projection matrix contains NaN");
            }
        }

        // Check that [3][2] is -1 (this is the w component divider for perspective)
        assert!((proj[3][2] - (-1.0)).abs() < EPSILON, "proj[3][2] should be -1.0, got {}", proj[3][2]);
    }

    #[test]
    fn test_view_projection_combination() {
        // Set up a simple camera
        let eye = [0.0, 0.0, 5.0];  // Camera back 5 units on Z
        let target = [0.0, 0.0, 0.0];  // Looking at origin
        let up = [0.0, 1.0, 0.0];

        let view = Renderer::look_at(eye, target, up);

        let fov_y = 45.0_f32.to_radians();
        let aspect = 1.0;  // Square viewport
        let near = 0.1;
        let far = 100.0;

        let proj = Renderer::perspective(fov_y, aspect, near, far);

        let view_proj = Renderer::multiply_matrices(proj, view);

        println!("View-Projection matrix:");
        for row in &view_proj {
            println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
        }

        // Check no NaN values
        for row in &view_proj {
            for &val in row {
                assert!(!val.is_nan(), "View-Projection matrix contains NaN");
            }
        }
    }

    #[test]
    fn test_spherical_to_cartesian() {
        // Test that spherical coordinates convert correctly to cartesian
        let distance: f32 = 5.0;
        let azimuth: f32 = 0.0;  // Looking from +X axis
        let elevation: f32 = 0.0;  // On the XZ plane

        let horizontal_distance = distance * elevation.cos();
        let x = horizontal_distance * azimuth.cos();
        let y = distance * elevation.sin();
        let z = horizontal_distance * azimuth.sin();

        assert_vec3_near([x, y, z], [5.0, 0.0, 0.0], EPSILON);

        // Test azimuth = 90° (π/2)
        let azimuth = std::f32::consts::PI / 2.0;
        let horizontal_distance = distance * elevation.cos();
        let x = horizontal_distance * azimuth.cos();
        let y = distance * elevation.sin();
        let z = horizontal_distance * azimuth.sin();

        assert_vec3_near([x, y, z], [0.0, 0.0, 5.0], EPSILON);

        // Test elevation = 45°
        let azimuth: f32 = 0.0;
        let elevation: f32 = std::f32::consts::PI / 4.0;
        let horizontal_distance = distance * elevation.cos();
        let x = horizontal_distance * azimuth.cos();
        let y = distance * elevation.sin();
        let z = horizontal_distance * azimuth.sin();

        let expected_h = 5.0 * (std::f32::consts::PI / 4.0).cos();
        let expected_y = 5.0 * (std::f32::consts::PI / 4.0).sin();
        assert_vec3_near([x, y, z], [expected_h, expected_y, 0.0], EPSILON);
    }
}
