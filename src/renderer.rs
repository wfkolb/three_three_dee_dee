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
}

impl Camera {
    pub fn new(position: [f32; 3], direction: [f32; 3], near: f32, far: f32) -> Self {
        Self {
            position,
            direction,
            near_plane: near,
            far_plane: far,
        }
    }
}

/// Vertex layout for GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    position: [f32; 3],
}

impl GpuVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

/// Individual renderer that renders to a viewport
pub struct Renderer {
    pub camera: Camera,
    pub viewport: Viewport,
    pub frame_offset: u32,
    model_center: [f32; 3],
    camera_distance: f32,
    rotation_speed: f32,
    current_angle: f32,

    // GPU resources
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,
    render_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

impl Renderer {
    pub fn new(camera: Camera, viewport: Viewport, frame_offset: u32, model_center: [f32; 3], camera_distance: f32) -> Self {
        Self {
            camera,
            viewport,
            frame_offset,
            model_center,
            camera_distance,
            rotation_speed: 0.01, // radians per frame (slower, smoother rotation)
            current_angle: 0.0,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            render_pipeline: None,
            uniform_buffer: None,
            bind_group: None,
        }
    }

    /// Update camera position for rotation
    pub fn update_camera(&mut self, queue: &wgpu::Queue) {
        self.current_angle += self.rotation_speed;

        // Orbit camera around model center
        let x = self.model_center[0] + self.camera_distance * self.current_angle.cos();
        let z = self.model_center[2] + self.camera_distance * self.current_angle.sin();

        self.camera.position = [x, self.model_center[1], z];
        self.camera.direction = [
            self.model_center[0] - x,
            self.model_center[1] - self.camera.position[1],
            self.model_center[2] - z,
        ];

        // Update uniforms with new camera position
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

        println!("    Renderer: Created buffers with {} indices", self.num_indices);

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
                visibility: wgpu::ShaderStages::VERTEX,
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
                depth_stencil: None,
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

        let mut view_proj = Self::multiply_matrices(projection, view);

        // Store the rotation angle in an unused part of the matrix
        // Use first row, last column to pass the angle to the shader
        view_proj[0][3] = self.current_angle;

        // Debug output on first frame
        static FIRST_CALL: std::sync::Once = std::sync::Once::new();
        FIRST_CALL.call_once(|| {
            println!("\n=== Camera Debug Info ===");
            println!("Eye: [{}, {}, {}]", eye[0], eye[1], eye[2]);
            println!("Target: [{}, {}, {}]", target[0], target[1], target[2]);
            println!("FOV: {} degrees, Aspect: {}", fov_y.to_degrees(), aspect);
            println!("Near: {}, Far: {}", self.camera.near_plane, self.camera.far_plane);
            println!("View-Projection matrix:");
            for row in &view_proj {
                println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", row[0], row[1], row[2], row[3]);
            }
        });

        Uniforms {
            view_proj,
        }
    }

    /// Simple look-at matrix (column-major for WGSL)
    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        let f = Self::normalize([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ]);
        let s = Self::normalize(Self::cross(f, up));
        let u = Self::cross(s, f);

        // Column-major matrix for WGSL
        [
            [s[0], s[1], s[2], 0.0],
            [u[0], u[1], u[2], 0.0],
            [-f[0], -f[1], -f[2], 0.0],
            [-Self::dot(s, eye), -Self::dot(u, eye), Self::dot(f, eye), 1.0],
        ]
    }

    /// Perspective projection matrix for WebGPU/Metal (0 to 1 depth range, column-major)
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let f = 1.0 / (fov_y / 2.0).tan();
        // WebGPU uses 0 to 1 depth range (not -1 to 1 like OpenGL)
        // Column-major matrix for WGSL
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), (near * far) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ]
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

    fn multiply_matrices(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    /// Render to the specified viewport
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame_number: u32,
    ) {
        // Apply frame offset
        if self.frame_offset > 0 && (frame_number % (self.frame_offset + 1)) != 0 {
            return; // Skip this frame based on offset
        }

        if frame_number == 0 {
            println!("Renderer: First render - drawing {} indices at viewport ({}, {}, {}x{})",
                self.num_indices, self.viewport.x, self.viewport.y, self.viewport.width, self.viewport.height);
        }

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
            depth_stencil_attachment: None,
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

        render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.as_ref().unwrap().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

/// Uniform data sent to GPU
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}
