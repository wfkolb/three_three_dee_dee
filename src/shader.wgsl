// Vertex shader

struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) @interpolate(flat) face_id: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = model.position;

    // Calculate face ID: each triangle uses 3 vertices
    out.face_id = model.vertex_idx / 3u;

    // Apply rotation manually in the shader for testing
    let angle = uniforms.view_proj[0][3]; // Use first row, last column as angle storage
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    // Rotate around Y axis
    let rotated_x = model.position.x * cos_a - model.position.z * sin_a;
    let rotated_z = model.position.x * sin_a + model.position.z * cos_a;
    let rotated_pos = vec3<f32>(rotated_x, model.position.y, rotated_z);

    // Simple scale and center the cube for testing
    // Scale down by 0.3 and push back slightly in Z
    let pos = rotated_pos * 0.3;
    out.clip_position = vec4<f32>(pos.x, pos.y, pos.z * 0.5 + 0.5, 1.0);

    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Generate color from face ID using a simple hash
    // Each face will get a unique, consistent color
    let id = in.face_id;
    let color = vec3<f32>(
        f32((id * 123u) % 255u) / 255.0,
        f32((id * 456u) % 255u) / 255.0,
        f32((id * 789u) % 255u) / 255.0
    );

    return vec4<f32>(color, 1.0);
}
