use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod fbx_loader;
mod renderer;
mod screen;

use fbx_loader::FbxLoader;
use renderer::{Camera, Viewport};
use screen::Screen;

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
    async fn new(window: std::sync::Arc<Window>, fbx_loader: FbxLoader) -> Self {
        // Get model info for camera positioning
        let model_center = fbx_loader.get_center();
        let model_size = fbx_loader.get_max_dimension();
        println!("\nModel info for camera setup:");
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

        // Position cameras based on model size
        // Camera distance = model_size * 2.5 to see the whole model
        let camera_distance = model_size *0.5;

        // Top half renderer - looking at model center from the front
        let cam1_pos = [
            model_center[0],
            model_center[1],
            model_center[2] + camera_distance,
        ];
        let cam1_dir = [
            model_center[0] - cam1_pos[0],
            model_center[1] - cam1_pos[1],
            model_center[2] - cam1_pos[2],
        ];
        //let cam1_dir = [
         //   1.0,0.0,0.0
        //];

        println!("  Camera 1: pos=({}, {}, {}), dir=({}, {}, {})",
            cam1_pos[0], cam1_pos[1], cam1_pos[2],
            cam1_dir[0], cam1_dir[1], cam1_dir[2]);

        screen.add_renderer(
            Camera::new(
                cam1_pos,
                cam1_dir,  // looking toward model center
                0.1,       // near plane
                camera_distance * 10.0,  // far plane
            ),
            Viewport {
                x: 0,
                y: 0,
                width: size.width,
                height: size.height ,
            },
            0, // no frame offset
            model_center,
            camera_distance,
        );

        // Bottom half renderer - looking at model from the side
        let cam2_pos = [
            model_center[0] - camera_distance,
            model_center[1],
            model_center[2],
        ];
        let cam2_dir = [
            model_center[0] - cam2_pos[0],
            model_center[1] - cam2_pos[1],
            model_center[2] - cam2_pos[2],
        ];
        println!("  Camera 2: pos=({}, {}, {}), dir=({}, {}, {})",
            cam2_pos[0], cam2_pos[1], cam2_pos[2],
            cam2_dir[0], cam2_dir[1], cam2_dir[2]);
        /*
        screen.add_renderer(
            Camera::new(
                cam2_pos,
                cam2_dir,  // looking toward model center
                0.01,       // near plane
                camera_distance * 10.0,  // far plane
            ),
            Viewport {
                x: 0,
                y: size.height / 2,
                width: size.width,
                height: size.height / 2,
            },
            0, // no frame offset for second camera too (removed the 10 frame offset)
            model_center,
            camera_distance,
        );*/

        // Initialize renderers with FBX data
        screen.initialize_renderers(&device, &config, &fbx_loader);

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
    fbx_loader: Option<FbxLoader>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("3D Renderer - Hello wgpu!")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = std::sync::Arc::new(
                event_loop
                    .create_window(window_attributes)
                    .unwrap()
            );

            let fbx_loader = self.fbx_loader.take().expect("FBX loader not initialized");
            self.state = Some(pollster::block_on(State::new(window, fbx_loader)));
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

    println!("\nLoaded FBX Summary:");
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

    // Use default FBX file if none provided
    let fbx_path = if args.len() >= 2 {
        args[1].clone()
    } else {
        "resources/headphones_joined.fbx".to_string()
    };

    // Load the FBX file
    let fbx_loader = match load_fbx_file(&fbx_path) {
        Ok(loader) => {
            println!("Successfully loaded FBX file: {}", fbx_path);
            loader
        }
        Err(e) => {
            eprintln!("Error loading FBX file '{}': {}", fbx_path, e);
            std::process::exit(1);
        }
    };

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        state: None,
        fbx_loader: Some(fbx_loader),
    };

    event_loop.run_app(&mut app).unwrap();
}
