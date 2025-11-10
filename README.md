# 3D Mesh Renderer

A real-time 3D mesh renderer built with Rust, WGPU, and Winit. Supports loading FBX models with interactive camera controls and on-screen orientation display.

## Features

- **FBX Model Loading**: Load and render FBX mesh files
- **Reference Cube**: Built-in cube mesh for testing
- **Interactive Camera**: Mouse-based orbit controls with zoom
- **Edge Rendering**: Black wireframe edges for better visualization
- **Real-time Orientation Display**: Shows azimuth, elevation, and depth
- **Custom Font Support**: Uses TrenchThin font for text rendering

## Building

```bash
cargo build --release
```

## Usage

### Run with Reference Cube
```bash
cargo run
```

### Run with FBX Model
```bash
cargo run resources/model.fbx
```

## Controls

### Mouse Controls
- **Left Click + Drag**: Rotate camera around the model
  - Horizontal drag: Azimuth rotation (yaw)
  - Vertical drag: Elevation rotation (pitch)
- **Mouse Wheel**: Zoom in/out (adjust camera distance)

### Keyboard Controls
- **Arrow Up**: Increase light intensity
- **Arrow Down**: Decrease light intensity
- **Escape**: Exit application

## On-Screen Display

The top-left corner displays:
- **Azimuth**: Horizontal rotation angle (0-360°)
- **Elevation**: Vertical rotation angle (0-360°)
- **Depth**: Camera distance from model center

## Camera System

The camera orbits around the model center using spherical coordinates:
- Initial distance is automatically calculated as 2.5× the model's maximum dimension
- Elevation is clamped to ±89° to prevent gimbal lock
- Zoom range: 0.5 to 50.0 units

## Technical Details

- **Graphics API**: WGPU (WebGPU)
- **Windowing**: Winit 0.30
- **Mesh Format**: FBX via `fbx` crate
- **Text Rendering**: wgpu-text 0.9
- **Lighting**: Phong shading with adjustable intensity
- **Normals**: Per-face normals for accurate lighting across all viewing angles
