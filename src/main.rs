use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod fbx_loader;
mod reference_mesh;
mod renderer;
mod screen;

use fbx_loader::FbxLoader;
use reference_mesh::ReferenceMesh;
use renderer::{Camera, Viewport};
use screen::{MeshSource, Screen};

/// Enum to hold either FBX or reference mesh data
enum MeshData {
    Fbx(FbxLoader),
    Reference(ReferenceMesh),
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: std::sync::Arc<winit::window::Window>,
    screen: Screen,
}

impl State {
    async fn new(window: std::sync::Arc<Window>, mesh_data: MeshData) -> Self {
        // Get model info for camera positioning
        let (model_center, model_size) = match &mesh_data {
            MeshData::Reference(ref_mesh) => {
                println!("\nUsing reference mesh for camera setup:");
                (ref_mesh.get_center(), ref_mesh.get_max_dimension())
            }
            MeshData::Fbx(fbx) => {
                println!("\nUsing FBX model for camera setup:");
                (fbx.get_center(), fbx.get_max_dimension())
            }
        };
        println!("  Center: ({}, {}, {})", model_center[0], model_center[1], model_center[2]);
        println!("  Max dimension: {}", model_size);
        let size = window.inner_size();

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window.clone()).unwrap();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Create and configure screen with renderers
        let mut screen = Screen::new();

        // Position camera to orbit around the cube
        // Camera distance = model_size * 2.5 to see the whole model comfortably
        let camera_distance = model_size * 2.5;

        // Start camera at a position on the Z axis, looking at the center
        let cam_pos = [
            model_center[0],
            model_center[1],
            model_center[2] + camera_distance,
        ];
        let cam_dir = [
            model_center[0] - cam_pos[0],
            model_center[1] - cam_pos[1],
            model_center[2] - cam_pos[2],
        ];

        println!("  Camera: pos=({}, {}, {}), dir=({}, {}, {})",
            cam_pos[0], cam_pos[1], cam_pos[2],
            cam_dir[0], cam_dir[1], cam_dir[2]);
        println!("  Camera distance: {}", camera_distance);

        screen.add_renderer(
            Camera::new(
                cam_pos,
                cam_dir,
                0.1,       // near plane
                100.0,     // far plane
            ),
            Viewport {
                x: 0,
                y: 0,
                width: size.width,
                height: size.height,
            },
            0, // no frame offset
            model_center,
            camera_distance,
        );

        // Initialize renderers with mesh data
        let mesh_source = match &mesh_data {
            MeshData::Reference(ref_mesh) => MeshSource::Reference(ref_mesh),
            MeshData::Fbx(fbx) => MeshSource::Fbx(fbx),
        };
        screen.initialize_renderers(&device, &config, mesh_source);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            screen,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // First, clear the entire screen
        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        // Render all viewports via Screen
        self.screen.render(&mut encoder, &view, &self.queue);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct App {
    state: Option<State>,
    mesh_data: Option<MeshData>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("3D Renderer")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = std::sync::Arc::new(
                event_loop
                    .create_window(window_attributes)
                    .unwrap()
            );

            let mesh_data = self.mesh_data.take().expect("Mesh data not initialized");
            self.state = Some(pollster::block_on(State::new(window, mesh_data)));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
fn load_fbx_file(path: &str) -> Result<FbxLoader, Box<dyn std::error::Error>> {
    let fbx_loader = FbxLoader::load(path)?;

    println!("\nLoaded FBX file: {}", path);
    println!("  Total meshes: {}", fbx_loader.meshes.len());
    println!("  Total vertices: {}", fbx_loader.total_vertex_count());
    println!("  Total indices: {}", fbx_loader.total_index_count());

    for (i, mesh) in fbx_loader.meshes.iter().enumerate() {
        println!(
            "  Mesh {}: '{}' - {} vertices, {} indices",
            i,
            mesh.name,
            mesh.vertices.len(),
            mesh.indices.len()
        );
    }

    Ok(fbx_loader)
}

fn main() {
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Determine which mesh to use
    let mesh_data = if args.len() >= 2 {
        // FBX file path provided
        let fbx_path = &args[1];
        match load_fbx_file(fbx_path) {
            Ok(fbx_loader) => MeshData::Fbx(fbx_loader),
            Err(e) => {
                eprintln!("Error loading FBX file '{}': {}", fbx_path, e);
                eprintln!("Falling back to reference cube mesh");
                let reference_mesh = ReferenceMesh::cube();
                println!("\nCreated reference cube mesh:");
                println!("  Vertices: {}", reference_mesh.vertices.len());
                println!("  Indices: {}", reference_mesh.indices.len());
                MeshData::Reference(reference_mesh)
            }
        }
    } else {
        // No FBX file provided, use reference mesh
        let reference_mesh = ReferenceMesh::cube();
        println!("\nNo FBX file provided, using reference cube mesh:");
        println!("  Vertices: {}", reference_mesh.vertices.len());
        println!("  Indices: {}", reference_mesh.indices.len());
        println!("  Center: ({}, {}, {})",
            reference_mesh.get_center()[0],
            reference_mesh.get_center()[1],
            reference_mesh.get_center()[2]
        );
        println!("  Max dimension: {}", reference_mesh.get_max_dimension());
        println!("\nTip: Run with an FBX file path as an argument to load a model");
        println!("  Example: cargo run resources/model.fbx");
        MeshData::Reference(reference_mesh)
    };

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        state: None,
        mesh_data: Some(mesh_data),
    };

    event_loop.run_app(&mut app).unwrap();
}
