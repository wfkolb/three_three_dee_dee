// Vertex shader

struct Uniforms {
    view_proj: mat4x4<f32>,
    light_strength: f32,
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    // Store world position for lighting calculations
    out.world_position = model.position;

    // Transform normal to world space (for now, just use as-is since we have no model matrix)
    out.world_normal = normalize(model.normal);

    // Transform vertex position using view-projection matrix
    let world_pos = vec4<f32>(model.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;

    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use a neutral grey base color
    let base_color = vec3<f32>(0.5, 0.5, 0.5);

    // Use the interpolated normal from the vertex shader
    var face_normal = normalize(in.world_normal);

    // Calculate view direction from surface to camera
    let camera_pos = vec3<f32>(uniforms.camera_pos_x, uniforms.camera_pos_y, uniforms.camera_pos_z);
    let view_dir = normalize(camera_pos - in.world_position);

    // Ensure normal faces toward camera (for two-sided lighting)
    if (dot(face_normal, view_dir) < 0.0) {
        face_normal = -face_normal;
    }

    // Define light direction (pointing from light source toward origin)
    // This is a directional light coming from upper-right-front
    let light_dir = normalize(vec3<f32>(-0.5, -0.7, -0.5));

    // Calculate diffuse lighting (Lambertian)
    // Use max to clamp negative values (back-facing surfaces)
    let diffuse = max(dot(face_normal, -light_dir), 0.0);

    // Calculate specular lighting (Phong)
    // Reflect the light direction around the normal
    let reflect_dir = reflect(light_dir, face_normal);
    let specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0); // 32 is shininess

    // Lighting parameters
    let ambient = 0.3;        // Ambient light strength
    let diffuse_strength = uniforms.light_strength; // Diffuse light strength from uniform
    let specular_strength = 0.5; // Specular highlight strength

    // Combine lighting: ambient + diffuse + specular
    let lighting = ambient + diffuse_strength * diffuse + specular_strength * specular;

    // Apply lighting to the base color
    let final_color = base_color * lighting;

    return vec4<f32>(final_color, 1.0);
}

// Wireframe fragment shader (for edge rendering)
@fragment
fn fs_wireframe(in: VertexOutput) -> @location(0) vec4<f32> {
    // Draw edges in black
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
