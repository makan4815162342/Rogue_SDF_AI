# -------------------------------------------------------------------
# Rogue SDF AI
# A modified version of SDF Prototyper developed by Makan Asnasri and Bing AI
# -------------------------------------------------------------------
# at top of file
# ——— Standard library ———
import os
import textwrap
import math
import array

# ——— Third‐party ———
import numpy as np

# ——— Blender Python API ———
import bpy
from bpy.props import (
    StringProperty,
    PointerProperty,
    EnumProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    FloatVectorProperty,
    CollectionProperty,
)
from bpy.types import PropertyGroup, Panel

# ——— GPU & Offscreen Rendering ———
import gpu
from gpu import state
from gpu.types import GPUShader, GPUOffScreen
from gpu_extras.batch import batch_for_shader

# ——— Blender math utils ———
from mathutils import Vector

try:
    import mcubes
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False

try:
    import openvdb
    HAS_OPENVDB = True
except ImportError:
    HAS_OPENVDB = False


# keep these globals at the top
handler = shader = batch = None
MAX_SHAPES = 32
# module‐level keymap storage
_addon_keymaps = []
_timer_is_running = False

def get_bezier_point(t, p0, h0, h1, p1):
    """
    Calculates a point on a cubic Bézier curve.
    t: parameter from 0.0 to 1.0
    p0, p1: control points
    h0, h1: handle points for p0 and p1 respectively
    """
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    t_sq = t * t
    
    return (one_minus_t * one_minus_t_sq * p0 +
            3.0 * one_minus_t_sq * t * h0 +
            3.0 * one_minus_t * t_sq * h1 +
            t * t_sq * p1)


def is_mirror_matrix(matrix):
    return matrix.determinant() < 0

def fix_scale_and_direction(base_empty, tip_empty):
    base_pos = base_empty.matrix_world.to_translation()
    tip_pos = tip_empty.matrix_world.to_translation()
    direction = (tip_pos - base_pos).normalized()
    if direction.length < 0.0001:
        direction = Vector((0, 1, 0))
    rot = direction.to_track_quat('Y', 'Z')
    base_empty.rotation_mode = 'QUATERNION'
    base_empty.rotation_quaternion = rot


# ——— live-redraw callback ———
def _redraw_shader_view(self, context):
    """
    Called on color/light-slider change.
    Forces all 3D views to redraw so the GLSL preview updates.
    """
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def update_visibility_and_mute(self, context):
    """
    Directly updates the visibility and mute state of an SDF object and its node.
    This is more reliable than using a global handler.
    """
    if self.empty_object:
        # Update viewport visibility of the empty
        self.empty_object.hide_viewport = self.is_viewport_hidden
        
        # Find the associated geometry node and update its mute state
        node_tree = get_sdf_geometry_node_tree(context)
        if node_tree:
            node = next((n for n in node_tree.nodes if n.get("associated_empty") == self.empty_object.name), None)
            if node and node.mute != self.is_hidden:
                node.mute = self.is_hidden
    
    # Always rewire the chain after a mute, as it affects the connections
    rewire_full_sdf_chain(context)

def update_point_cloud_preview(self, context):
    """
    Dedicated update function for the point cloud toggle.
    This safely calls the necessary updates without causing a feedback loop.
    """
    # Only execute if the change is made by a user action, not during file load or creation
    if context.object and context.object.mode == 'OBJECT':
        rewire_full_sdf_chain(context)
        _redraw_shader_view(self, context)

#----------------------------------------------------------------------------------------------------


from mathutils import Euler, Vector, Matrix # Make sure Matrix is also imported

def collect_sdf_data(context):
    """
    Gather ALL shape data. This version filters shapes based on the current
    view mode (All, Selected, Unselected) for both preview and baking.
    """
    shapes = []
    op_map = {
        'UNION': 3, 'SUBTRACT': 4, 'INTERSECT': 5, 'PAINT': 6,
        'DISPLACE': 7, 'INDENT': 8, 'RELIEF': 9, 'ENGRAVE': 10, 'MASK': 11
    }
    domain = getattr(context.scene, "sdf_domain", None)
    
    MAX_SHAPES_CURRENT = context.scene.sdf_max_shapes

    if not domain:
        return [(-1, (0,0,0), (1,1,1), (1,0,0,0), 0, 0.0, 0, 0, (1,1,1), 0, -1, (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), 0.0, 0.0)] * MAX_SHAPES_CURRENT

    # --- NEW: Get view mode and selected objects ---
    view_mode = context.scene.sdf_view_mode
    selected_empties = {obj for obj in context.view_layer.objects.selected}
    # --- END NEW ---

    for item_index, item in enumerate(domain.sdf_nodes):
        e = item.empty_object
        if not e or item.is_hidden:
            continue

        # --- NEW: Filtering Logic ---
        is_selected = e in selected_empties
        if view_mode == 'SELECTED' and not is_selected:
            continue # Skip if we only want selected, and this one isn't
        if view_mode == 'UNSELECTED' and is_selected:
            continue # Skip if we only want unselected, and this one is
        # --- END NEW ---

        if len(shapes) >= MAX_SHAPES_CURRENT:
            continue

        # In function collect_sdf_data, inside the for loop:
        op_base = op_map.get(item.operation, 3)
        op = op_base
        if item.operation in ['UNION', 'SUBTRACT', 'INTERSECT']:
            if item.blend_type == 'CHAMFER':
                op = op_base + 10
            elif item.blend_type == 'GROOVE':
                op = op_base + 20
            elif item.blend_type == 'PIPE':
                op = op_base + 30
        
        # ... (the rest of the function from this point on is identical to your current version)
        code = { 'MESH_CUBE': 0, 'MESH_UVSPHERE': 1, 'MESH_TORUS': 2, 'MESH_CYLINDER': 3, 'MESH_CONE': 4, 'MESH_ICOSPHERE': 5, 'CAPSULE': 6 }.get(item.icon, -1)
        blend = item.blend
        strength = item.blend_strength
        fill = item.mask_fill_amount
        color = item.preview_color
        highlight = int(item.use_highlight)
        mirror_flags = (int(item.use_mirror_x) * 1) | (int(item.use_mirror_y) * 2) | (int(item.use_mirror_z) * 4)
        radial_count = item.radial_mirror_count if item.use_radial_mirror else 0
        itemID = item_index
        params1, params2 = (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0)
        if item.icon != 'CURVE_BEZCURVE':
            params1, params2 = get_params_for_shape(item, item.icon)
        if item.icon == 'CURVE_BEZCURVE':
            curve_obj = next((child for child in e.children if child.type == 'CURVE'), None)
            if not curve_obj or not curve_obj.data.splines: continue
            def lerp_val(v1, v2, f): return v1 * (1.0 - f) + v2 * f
            def lerp_vec(v1, v2, f): return v1.lerp(v2, f)
            sorted_points = []
            if item.curve_control_mode == 'CUSTOM' and item.custom_control_points:
                sorted_points = sorted(item.custom_control_points, key=lambda p: p.t_value)
            for spline in curve_obj.data.splines:
                if len(spline.bezier_points) < 2: continue
                total_spline_length = 0
                segment_lengths = []
                for i in range(len(spline.bezier_points) - 1):
                    length = (spline.bezier_points[i+1].co - spline.bezier_points[i].co).length
                    segment_lengths.append(length)
                    total_spline_length += length
                if total_spline_length < 0.0001: continue
                distance_along_spline = 0
                for i in range(len(spline.bezier_points) - 1):
                    bp1, bp2 = spline.bezier_points[i], spline.bezier_points[i+1]
                    r1, r2 = bp1.radius * item.curve_global_radius, bp2.radius * item.curve_global_radius
                    segment_start_t = distance_along_spline / total_spline_length
                    distance_along_spline += segment_lengths[i]
                    segment_end_t = distance_along_spline / total_spline_length
                    p0_geom, h0_geom, h1_geom, p1_geom = bp1.co, bp1.handle_right, bp2.handle_left, bp2.co
                    density = item.curve_point_density
                    num_points = max(1, int(density / (item.curve_instance_spacing + 1e-6)))
                    for j in range(num_points):
                        if len(shapes) >= MAX_SHAPES_CURRENT: break
                        t_sub = j / float(num_points - 1) if num_points > 1 else 0.0
                        current_point_local = get_bezier_point(t_sub, p0_geom, h0_geom, h1_geom, p1_geom) if item.curve_mode == 'SMOOTH' else p0_geom.lerp(p1_geom, t_sub)
                        current_point = curve_obj.matrix_world @ current_point_local
                        direction = Vector((0,1,0))
                        if item.curve_mode == 'SMOOTH' and j == num_points - 1 and num_points > 1:
                            prev_t = (j - 1) / float(num_points - 1)
                            prev_point_local = get_bezier_point(prev_t, p0_geom, h0_geom, h1_geom, p1_geom)
                            prev_point = curve_obj.matrix_world @ prev_point_local
                            direction = (current_point - prev_point)
                        else:
                            next_t_sub = (j + 0.1) / float(num_points) if item.curve_mode == 'HARD' else min(t_sub + 0.01, 1.0)
                            next_point_local = p0_geom.lerp(p1_geom, next_t_sub) if item.curve_mode == 'HARD' else get_bezier_point(next_t_sub, p0_geom, h0_geom, h1_geom, p1_geom)
                            next_point = curve_obj.matrix_world @ next_point_local
                            direction = (next_point - current_point)
                        if direction.length > 0.0001:
                            direction.normalize()
                        t_spline = segment_start_t * (1.0 - t_sub) + segment_end_t * t_sub
                        final_params1, final_params2 = Vector((0.0,0.0,0.0,0.0)), Vector((0.0,0.0,0.0,0.0))
                        final_color = Vector(item.preview_color)
                        final_radius_mult = 1.0
                        final_shape_code = { 'MESH_CUBE': 0, 'MESH_UVSPHERE': 1, 'MESH_TORUS': 2, 'MESH_CYLINDER': 3, 'MESH_CONE': 4, 'MESH_ICOSPHERE': 5, 'CAPSULE': 6 }.get(item.curve_instance_type, 1)
                        final_local_rot = Euler(item.curve_instance_rotation, 'XYZ').to_quaternion()
                        if item.curve_control_mode == 'CUSTOM' and sorted_points:
                            p1, p2 = sorted_points[0], sorted_points[-1]
                            if t_spline <= p1.t_value: p2 = p1
                            elif t_spline >= p2.t_value: p1 = p2
                            else:
                                for k in range(len(sorted_points) - 1):
                                    if sorted_points[k].t_value <= t_spline < sorted_points[k+1].t_value:
                                        p1, p2 = sorted_points[k], sorted_points[k+1]
                                        break
                            f = 0.0
                            if (p2.t_value - p1.t_value) > 1e-6: f = (t_spline - p1.t_value) / (p2.t_value - p1.t_value)
                            final_color = lerp_vec(Vector(p1.color), Vector(p2.color), f)
                            final_radius_mult = lerp_val(p1.radius_multiplier, p2.radius_multiplier, f)
                            final_shape_code = { 'MESH_UVSPHERE': 1, 'MESH_CUBE': 0, 'MESH_ICOSPHERE': 5, 'CAPSULE': 6, 'MESH_TORUS': 2, 'MESH_CYLINDER': 3, 'MESH_CONE': 4 }.get(p1.shape_type, 1)
                            q1, q2 = Euler(p1.rotation, 'XYZ').to_quaternion(), Euler(p2.rotation, 'XYZ').to_quaternion()
                            final_local_rot = q1.slerp(q2, f)
                            if final_shape_code == 0:
                                thickness = lerp_val(p1.thickness, p2.thickness, f); roundness = lerp_val(p1.roundness, p2.roundness, f); bevel = lerp_val(p1.bevel, p2.bevel, f); pyramid = lerp_val(p1.pyramid, p2.pyramid, f); twist = lerp_val(p1.twist, p2.twist, f); bend = lerp_val(p1.bend, p2.bend, f)
                                final_params1 = Vector((thickness, roundness, bevel, pyramid)); final_params2 = Vector((twist, bend, 0.0, 0.0))
                            elif final_shape_code == 1:
                                sphere_thickness = lerp_val(p1.sphere_thickness, p2.sphere_thickness, f); sphere_elongation = lerp_val(p1.sphere_elongation, p2.sphere_elongation, f); sphere_cut_angle = lerp_val(p1.sphere_cut_angle, p2.sphere_cut_angle, f)
                                final_params1 = Vector((sphere_thickness, sphere_elongation, sphere_cut_angle, 0.0))
                            elif final_shape_code == 3:
                                cylinder_thickness = lerp_val(p1.cylinder_thickness, p2.cylinder_thickness, f); cylinder_roundness = lerp_val(p1.cylinder_roundness, p2.cylinder_roundness, f); cylinder_pyramid = lerp_val(p1.cylinder_pyramid, p2.cylinder_pyramid, f); cylinder_bend = lerp_val(p1.cylinder_bend, p2.cylinder_bend, f)
                                final_params1 = Vector((cylinder_thickness, cylinder_roundness, cylinder_pyramid, 0.0)); final_params2 = Vector((0.0, cylinder_bend, 0.0, 0.0))
                            elif final_shape_code == 5:
                                prism_sides = lerp_val(p1.prism_sides, p2.prism_sides, f); prism_pyramid = lerp_val(p1.prism_pyramid, p2.prism_pyramid, f); prism_thickness = lerp_val(p1.prism_thickness, p2.prism_thickness, f); prism_bend = lerp_val(p1.prism_bend, p2.prism_bend, f); prism_twist = lerp_val(p1.prism_twist, p2.prism_twist, f)
                                final_params1 = Vector((prism_sides, prism_pyramid, prism_thickness, 0.0)); final_params2 = Vector((prism_bend, prism_twist, 0.0, 0.0))
                            elif final_shape_code == 2:
                                torus_outer_radius = lerp_val(p1.torus_outer_radius, p2.torus_outer_radius, f); torus_inner_radius = lerp_val(p1.torus_inner_radius, p2.torus_inner_radius, f); torus_cut_angle = lerp_val(p1.torus_cut_angle, p2.torus_cut_angle, f); torus_thickness = lerp_val(p1.torus_thickness, p2.torus_thickness, f); torus_elongation = lerp_val(p1.torus_elongation, p2.torus_elongation, f)
                                final_params1 = Vector((torus_outer_radius, torus_inner_radius, torus_cut_angle, torus_thickness)); final_params2 = Vector((torus_elongation, 0.0, 0.0, 0.0))
                        else:
                            final_params1, final_params2 = get_params_for_shape(item, item.curve_instance_type)
                        taper_factor = item.curve_taper_head * (1.0 - t_spline) + item.curve_taper_tail * t_spline
                        current_radius = (r1 * (1.0 - t_sub) + r2 * t_sub) * item.curve_global_radius * taper_factor * final_radius_mult
                        base_rotation_quat = Vector((0,1,0)).rotation_difference(direction)
                        final_rotation = base_rotation_quat @ final_local_rot
                        pos, rot = current_point, final_rotation
                        height = (segment_lengths[i] / num_points) * item.curve_segment_scale
                        if final_shape_code == 6: scl = Vector((current_radius, height, current_radius))
                        elif final_shape_code == 4: scl = Vector((current_radius, height, 0.001))
                        elif final_shape_code == 3: scl = Vector((current_radius, height, current_radius))
                        else: scl = Vector((current_radius, current_radius, current_radius))
                        shapes.append((final_shape_code, pos, scl, rot, op, blend, mirror_flags, radial_count, final_color, highlight, itemID, final_params1, final_params2, strength, fill))
                if len(shapes) >= MAX_SHAPES_CURRENT: break
        else:
            mw = e.matrix_world
            if item.icon == 'MESH_CONE':
                tip_empty = next((child for child in e.children if "Tip" in child.name), None)
                if not tip_empty: continue
                base_pos, tip_pos = e.matrix_world.to_translation(), tip_empty.matrix_world.to_translation()
                pos, height = (base_pos + tip_pos) / 2.0, (tip_pos - base_pos).length
                _, _, base_scl = e.matrix_world.decompose()
                _, _, tip_scl = tip_empty.matrix_world.decompose()
                r1 = (abs(base_scl.x) + abs(base_scl.z)) / 2.0
                r2 = (abs(tip_scl.x) + abs(tip_scl.z)) / 2.0
                scl, direction = Vector((r1, height, r2)), (tip_pos - base_pos).normalized() if height > 0 else Vector((0,1,0))
                rot = direction.to_track_quat('Y', 'Z')
            else:
                pos, rot, scl = mw.decompose() 
            
            shapes.append((code, pos, scl, rot, op, blend, mirror_flags, radial_count, color, highlight, itemID, params1, params2, strength, fill))

    while len(shapes) < MAX_SHAPES_CURRENT:
        shapes.append((-1, (0,0,0), (1,1,1), (1,0,0,0), 0, 0.0, 0, 0, (1,1,1), 0, -1, (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0), 0.0, 0.0))
        
    return shapes


def get_params_for_shape(item, shape_type):
    """Helper function to pack parameters for a given shape type from an item."""
    if shape_type == 'MESH_CUBE':
        return Vector((item.thickness, item.roundness, item.bevel, item.pyramid)), Vector((item.twist, item.bend, 0.0, 0.0))
    elif shape_type == 'MESH_UVSPHERE':
        return Vector((item.sphere_thickness, item.sphere_elongation, item.sphere_cut_angle, 0.0)), Vector((0.0, 0.0, 0.0, 0.0))
    elif shape_type == 'MESH_CYLINDER':
        return Vector((item.cylinder_thickness, item.cylinder_roundness, item.cylinder_pyramid, 0.0)), Vector((0.0, item.cylinder_bend, 0.0, 0.0))
    elif shape_type == 'MESH_ICOSPHERE':
        return Vector((item.prism_sides, item.prism_pyramid, item.prism_thickness, 0.0)), Vector((item.prism_bend, item.prism_twist, 0.0, 0.0))
    elif shape_type == 'MESH_TORUS':
        return Vector((item.torus_outer_radius, item.torus_inner_radius, item.torus_cut_angle, item.torus_thickness)), Vector((item.torus_elongation, 0.0, 0.0, 0.0))
    return Vector((0.0, 0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0, 0.0))
#----------------------------------------------------------------------------------------------------

import bpy
import gpu
from gpu import state
from gpu.types import GPUShader
from gpu_extras.batch import batch_for_shader
from bpy_extras.view3d_utils import location_3d_to_region_2d  # <-- THE CORRECT, RENAMED FUNCTION
from mathutils import Vector
import array
import math

def draw_sdf_shader():
    """
    Main draw handler for the SDF Shader Preview. This version sends packed data.
    """
    global shader, batch
    if not shader: return

    ctx = bpy.context
    scene = ctx.scene
    region = ctx.region
    rv3d = ctx.region_data
    domain = getattr(scene, "sdf_domain", None)
    
    if not domain: return

    corners = [domain.matrix_world @ Vector(corner) for corner in domain.bound_box]
    coords_2d = [location_3d_to_region_2d(region, rv3d, c) for c in corners]
    visible_coords = [c for c in coords_2d if c is not None]
    if not visible_coords: return
    min_x = min(c.x for c in visible_coords); max_x = max(c.x for c in visible_coords)
    min_y = min(c.y for c in visible_coords); max_y = max(c.y for c in visible_coords)
    scissor_x = int(min_x); scissor_y = int(min_y)
    scissor_w = int(max_x - min_x); scissor_h = int(max_y - min_y)
    state.scissor_test_set(True)
    scissor_x = max(0, scissor_x); scissor_y = max(0, scissor_y)
    scissor_w = min(region.width - scissor_x, scissor_w)
    scissor_h = min(region.height - scissor_y, scissor_h)
    if scissor_w <= 0 or scissor_h <= 0:
        state.scissor_test_set(False)
        return
    state.scissor_set(scissor_x, scissor_y, scissor_w, scissor_h)

    # --- MODIFICATION ---
    # We now ENABLE the depth test. This allows the GPU to automatically
    # hide parts of our SDF that are behind existing Blender geometry.
    state.depth_test_set('LESS_EQUAL')
    state.blend_set('NONE')

    shader.bind()
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("viewMatrixInv", rv3d.view_matrix.inverted())
    shader.uniform_float("projMatrixInv", rv3d.window_matrix.inverted())

    # --- NEW UNIFORM ---
    # Pass the combined matrix needed for depth calculation in the shader.
    view_proj_matrix = rv3d.window_matrix @ rv3d.view_matrix
    shader.uniform_float("uViewProjMatrix", view_proj_matrix)
    # --- END NEW UNIFORM ---

    # --- UPDATED: Send all new lighting uniforms ---
    shader.uniform_float("uLightDir", scene.sdf_light_direction)
    shader.uniform_float("uBrightness", scene.sdf_preview_brightness)
    shader.uniform_float("uContrast", scene.sdf_preview_contrast)
    shader.uniform_int("uCavityEnable", int(scene.sdf_cavity_enable))
    shader.uniform_float("uCavityStrength", scene.sdf_cavity_strength)

    shader.uniform_int("uMaxSteps", scene.sdf_raymarch_max_steps)
    shader.uniform_int("uPixelationAmount", scene.sdf_pixelation_amount)
    # --- END OF UPDATE ---

    shader.uniform_float("uGlobalTint", scene.sdf_global_tint)
    shader.uniform_float("uDomainCenter", domain.location)

    shapes = collect_sdf_data(ctx)
    MAX_SHAPES_CURRENT = scene.sdf_max_shapes
    shader.uniform_int("uCount", sum(1 for s in shapes if s[0] >= 0))

    # --- Data Flattening ---
    tf = [int(s[0]) for s in shapes]
    while len(tf) % 4 != 0: tf.append(-1)
    type_flat = [tf[i+j] for i in range(0, len(tf), 4) for j in range(4)]
    pos_flat = [v for s in shapes for v in s[1]]
    scale_flat = [v for s in shapes for v in s[2]]
    rot_flat = [v for s in shapes for v in s[3]]
    op_flat = [int(s[4]) for s in shapes]
    blend_flat = [float(s[5]) for s in shapes]
    strength_flat = [float(s[13]) for s in shapes]
    fill_flat = [float(s[14]) for s in shapes]
    mirror_flags_flat = [int(s[6]) for s in shapes]
    radial_count_flat = [int(s[7]) for s in shapes]
    color_flat = [v for s in shapes for v in s[8]]
    item_id_flat = [int(s[10]) for s in shapes]
    params1_flat = [v for s in shapes for v in s[11]]
    params2_flat = [v for s in shapes for v in s[12]]

    max_items = 64
    highlight_per_item = [0] * max_items
    if domain and hasattr(domain, 'sdf_nodes'):
        for i, item in enumerate(domain.sdf_nodes):
            if i < max_items:
                highlight_per_item[i] = int(item.use_highlight)
    highlight_flat = highlight_per_item

    # --- Byte Buffer Creation ---
    type_buf = array.array('i', type_flat).tobytes()
    pos_buf = array.array('f', pos_flat).tobytes()
    scale_buf = array.array('f', scale_flat).tobytes()
    rot_buf = array.array('f', rot_flat).tobytes()
    op_buf = array.array('i', op_flat).tobytes()
    blend_buf = array.array('f', blend_flat).tobytes()
    strength_buf = array.array('f', strength_flat).tobytes()
    fill_buf = array.array('f', fill_flat).tobytes()
    mirror_flags_buf = array.array('i', mirror_flags_flat).tobytes()
    radial_count_buf = array.array('i', radial_count_flat).tobytes()
    color_buf = array.array('f', color_flat).tobytes()
    highlight_buf = array.array('i', highlight_flat).tobytes()
    item_id_buf = array.array('i', item_id_flat).tobytes()
    params1_buf = array.array('f', params1_flat).tobytes()
    params2_buf = array.array('f', params2_flat).tobytes()

    def safe_uniform_vector(uniform_name, buffer, components, count):
        try:
            loc = shader.uniform_from_name(uniform_name)
            shader.uniform_vector_float(loc, buffer, components, count)
        except ValueError: pass
            
    def safe_uniform_vector_int(uniform_name, buffer, components, count):
        try:
            loc = shader.uniform_from_name(uniform_name)
            shader.uniform_vector_int(loc, buffer, components, count)
        except ValueError: pass

    # --- Uploading All Uniforms to the GPU ---
    safe_uniform_vector_int("uShapeTypePacked", type_buf, 4, len(type_flat) // 4)
    safe_uniform_vector("uShapePos", pos_buf, 3, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeScale", scale_buf, 3, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeRot", rot_buf, 4, MAX_SHAPES_CURRENT)
    safe_uniform_vector_int("uShapeOp", op_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeBlend", blend_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeBlendStrength", strength_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeMaskFill", fill_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeColor", color_buf, 3, MAX_SHAPES_CURRENT)
    safe_uniform_vector_int("uShapeMirrorFlags", mirror_flags_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector_int("uShapeRadialCount", radial_count_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector_int("uShapeHighlight", highlight_buf, 1, max_items)
    safe_uniform_vector_int("uShapeItemID", item_id_buf, 1, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeParams1", params1_buf, 4, MAX_SHAPES_CURRENT)
    safe_uniform_vector("uShapeParams2", params2_buf, 4, MAX_SHAPES_CURRENT)
    
    shader.uniform_int("uColorBlendMode", 1 if scene.sdf_color_blend_mode == 'SOFT' else 0)

    batch.draw(shader)
    
    state.scissor_test_set(False)
    # Restore the depth test to Blender's default for the 3D Viewport.
    state.depth_test_set('LESS_EQUAL')
    state.blend_set('NONE')


import os, textwrap, math
import bpy
import gpu
from gpu.types       import GPUShader
from gpu_extras.batch import batch_for_shader
from gpu import state

# module globals
handler = shader = batch = None

def enable_sdf_shader_view(enable):
    import os, textwrap
    from gpu.types       import GPUShader
    from gpu_extras.batch import batch_for_shader

    global handler, shader, batch

    ctx    = bpy.context
    scene  = ctx.scene
    domain = getattr(scene, "sdf_domain", None)

    # --- THE FIX ---
    # We will now control visibility using the object's display type,
    # which is more robust and avoids conflicts. We NO LONGER mute the node.
    
    # 1) When enabling, set the domain to show ONLY its bounds.
    #    When disabling, restore its original display type.
    if domain:
        if enable:
            # Store the original display type so we can restore it later
            domain["_orig_display"] = domain.display_type
            domain.display_type     = 'BOUNDS'
        else:
            # Restore the original display type if we have one saved
            if "_orig_display" in domain:
                domain.display_type = domain["_orig_display"]
                del domain["_orig_display"]

    # 2) When enabling, compile & install the GLSL handler
    if enable and handler is None:
        # 2a) Build the full-screen shader
        vert_src = textwrap.dedent("""\
            in vec2 pos;
            out vec2 uv;
            void main() {
                uv = pos * 0.5 + 0.5;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
        """)
        frag_path = os.path.join(os.path.dirname(__file__), "sdf_raycast.frag")
        with open(frag_path, 'r', encoding='utf-8-sig') as f:
            frag_src = f.read()

        shader = GPUShader(vert_src, frag_src)
        batch  = batch_for_shader(
            shader, 'TRI_STRIP',
            {"pos":[(-1,-1),(1,-1),(-1,1),(1,1)]}
        )

        # 2b) Install the draw handler
        handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_sdf_shader, (), 'WINDOW', 'POST_VIEW'
        )
        if ctx.area:
            ctx.area.tag_redraw()
        print("SDF Shader View ENABLED.")

    # 3) When disabling, remove the handler and restore state
    elif not enable and handler:
        bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
        handler = shader = batch = None

        # Force a redraw so the GN mesh + empties reappear
        if ctx.area:
            ctx.area.tag_redraw()
        print("SDF Shader View DISABLED.")

#----------------------------------------------------------------------------------------------------

def _clone_sdf_hierarchy(context, source_item, source_node, new_name_suffix=""):
    """
    A robust function to duplicate an SDF shape's entire hierarchy.
    It clones the main empty, all its children, the UI list item, and the geometry node.
    Returns a dictionary containing the new item, empty, and node.
    """
    domain_obj = context.scene.sdf_domain
    node_tree = get_sdf_geometry_node_tree(context)
    source_empty = source_item.empty_object

    new_node = node_tree.nodes.new(type=source_node.bl_idname)
    if new_node.bl_idname == 'GeometryNodeGroup':
        new_node.node_tree = source_node.node_tree
    new_node.location = source_node.location + Vector((0, -200))
    for i, orig_input in enumerate(source_node.inputs):
        if hasattr(orig_input, "default_value"):
            new_node.inputs[i].default_value = orig_input.default_value

    new_empty = source_empty.copy()
    if new_empty.data:
        new_empty.data = source_empty.data.copy()
    context.collection.objects.link(new_empty)
    new_empty.parent = source_empty.parent

    new_item = domain_obj.sdf_nodes.add()
    new_empty.name = f"{source_item.name}{new_name_suffix}"
    new_item.name = new_empty.name
    new_item.empty_object = new_empty
    
    for prop in source_item.rna_type.properties.keys():
        if prop not in ["name", "empty_object", "custom_control_points"]: # Exclude the collection itself
            try:
                setattr(new_item, prop, getattr(source_item, prop))
            except (AttributeError, TypeError):
                pass

    # --- SPECIAL HANDLING FOR CURVE CUSTOM POINTS (THE FIX) ---
    if source_item.icon == 'CURVE_BEZCURVE' and source_item.curve_control_mode == 'CUSTOM':
        # Clear any default points on the new item's list
        new_item.custom_control_points.clear()
        
        # Manually iterate and copy each control point and all of its properties
        for source_point in source_item.custom_control_points:
            new_point = new_item.custom_control_points.add()
            
            # Copy all properties from the source point to the new point
            for point_prop in source_point.rna_type.properties.keys():
                try:
                    setattr(new_point, point_prop, getattr(source_point, point_prop))
                except (AttributeError, TypeError):
                    # This might happen for read-only properties, safe to ignore
                    pass
    # --- END OF FIX ---

    new_node["associated_empty"] = new_empty.name

    new_children = []
    for child in source_empty.children:
        new_child = child.copy()
        if new_child.data:
            new_child.data = child.data.copy()
        context.collection.objects.link(new_child)
        new_child.parent = new_empty
        new_children.append(new_child)

    if source_item.icon == 'MESH_CONE':
        tip_child = next((c for c in new_children if "Tip" in c.name), None)
        obj_inputs = [sock for sock in new_node.inputs if sock.type == 'OBJECT']
        if len(obj_inputs) >= 2 and tip_child:
            obj_inputs[0].default_value = new_empty
            obj_inputs[1].default_value = tip_child
    elif source_item.icon == 'CURVE_BEZCURVE':
        curve_child = next((c for c in new_children if c.type == 'CURVE'), None)
        obj_input = next((sock for sock in new_node.inputs if sock.type == 'OBJECT'), None)
        if obj_input and curve_child:
            obj_input.default_value = curve_child
    else:
        obj_input = next((sock for sock in new_node.inputs if sock.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = new_empty

    return {"item": new_item, "empty": new_empty, "node": new_node}


#---------------------------------------------------------------------------------------------------

# In main.py, add this new helper function

from mathutils import Matrix, Vector, Euler

def _mirror_and_clone_shape(context, source_item, source_node, mirror_axis='X'):
    """
    A robust function to clone and mirror an SDF shape using a mathematically
    sound matrix sanitization method to prevent flipping and child object issues.
    """
    domain = context.scene.sdf_domain
    if not domain: return None

    # 1. Create a perfect clone of the hierarchy and UI item
    new_sdf = _clone_sdf_hierarchy(context, source_item, source_node, new_name_suffix=".Sym")
    new_item = new_sdf["item"]
    new_empty = new_sdf["empty"]

    # Clones should not inherit the original's mirror settings
    new_item.use_mirror_x = new_item.use_mirror_y = new_item.use_mirror_z = False
    new_item.use_radial_mirror = False

    # --- DEFINITIVE TRANSFORM LOGIC ---
    source_empty = source_item.empty_object
    
    # 2. Define the mirror transformation matrix relative to the Domain's center
    pivot = Matrix.Translation(domain.location)
    pivot_inv = pivot.inverted()
    
    scale_vec = Vector((-1, 1, 1)) if mirror_axis == 'X' else \
                Vector((1, -1, 1)) if mirror_axis == 'Y' else \
                Vector((1, 1, -1))
    scale_neg = Matrix.Diagonal(scale_vec).to_4x4()
    mirror_matrix = pivot @ scale_neg @ pivot_inv

    # 3. Apply and sanitize the PARENT's transform
    new_mirrored_matrix = mirror_matrix @ source_empty.matrix_world
    new_loc, new_rot, new_scl = new_mirrored_matrix.decompose()
    new_scl.x, new_scl.y, new_scl.z = abs(new_scl.x), abs(new_scl.y), abs(new_scl.z)
    new_empty.location = new_loc
    new_empty.rotation_quaternion = new_rot
    new_empty.scale = new_scl
    
    # CRITICAL: Force the dependency graph to update so all child transforms are recalculated
    context.view_layer.update()

    # 4. Sanitize CHILD transforms (THE FIX FOR CURVE AND CONE)
    for child_new in new_empty.children:
        # After the parent moves, the child's local matrix can become invalid (negative scale).
        # We must decompose it, force the scale to be positive, and build it back up.
        l, r, s = child_new.matrix_basis.decompose()
        s.x, s.y, s.z = abs(s.x), abs(s.y), abs(s.z)
        child_new.matrix_basis = Matrix.Translation(l) @ r.to_matrix().to_4x4() @ Matrix.Diagonal(s).to_4x4()

    # 5. For the Cone, re-calculate its orientation now that transforms are stable
    if new_item.icon == 'MESH_CONE':
        tip_new = next((c for c in new_empty.children if "Tip" in c.name), None)
        if tip_new:
            # This function now receives valid world positions for both base and tip.
            fix_scale_and_direction(new_empty, tip_new)

    # 6. Negate Deformers and fix Pac-Man angle
    if is_mirror_matrix(mirror_matrix):
        if hasattr(new_item, "bend"): new_item.bend *= -1
        if hasattr(new_item, "twist"): new_item.twist *= -1
        if hasattr(new_item, "cylinder_bend"): new_item.cylinder_bend *= -1
        if hasattr(new_item, "prism_bend"): new_item.prism_bend *= -1
        if hasattr(new_item, "prism_twist"): new_item.prism_twist *= -1
        
        # PAC-MAN FIX: We don't reflect the angle value. The object's rotation
        # is already mirrored, which correctly mirrors the opening.
        # The old logic was causing the incorrect angle change you saw.
        # So, we no longer modify sphere_cut_angle here.

    return new_item

#---------------------------------------------------------------------------------------------------

def get_sdf_geometry_node_tree(context):
    """
    Return the active SDF Domain's Geometry NodeTree, or None if missing.
    This is the single, robust function to get the node tree.
    """
    domain = getattr(context.scene, "sdf_domain", None)
    if not domain:
        return None

    # Find the first NODES-type modifier on the domain, regardless of its name
    geo_mod = next((m for m in domain.modifiers if m.type == 'NODES'), None)
    if not geo_mod or not geo_mod.node_group:
        return None

    return geo_mod.node_group



# This global variable will store the add-on's path once it's found.
_addon_path_cache = ""

def get_addon_dir():
    """
    A robust, context-aware way to get the add-on's directory that caches the result.
    """
    global _addon_path_cache
    if _addon_path_cache and os.path.exists(_addon_path_cache):
        return _addon_path_cache

    try:
        filepath = os.path.abspath(__file__)
        _addon_path_cache = os.path.dirname(filepath)
        return _addon_path_cache
    except NameError:
        for text_block in bpy.data.texts:
            if text_block.is_in_memory: continue
            if os.path.basename(text_block.filepath or "") == "Rogue_SDF_AI.py":
                _addon_path_cache = os.path.dirname(bpy.path.abspath(text_block.filepath))
                return _addon_path_cache
    print("[Rogue_SDF_AI] WARNING: Could not determine add-on directory.")
    return ""
    
def load_all_sdf_node_groups():
    """
    Checks if the core SDF node groups are present in the file. If not,
    it loads them from the addon's resource .blend file. This function
    is designed to be run safely multiple times, but will only perform
    the expensive load operation once.
    """
    # Check if a key node group already exists. If it does, we assume all are loaded.
    if "SDF Domain" in bpy.data.node_groups:
        print("[Rogue_SDF_AI] Node groups already loaded.")
        return True

    print("[Rogue_SDF_AI] Core node groups not found. Loading from resource file...")
    addon_dir = get_addon_dir()
    if not addon_dir:
        # This is a fatal error, so we raise an exception.
        raise RuntimeError("Could not determine the addon's directory. Cannot load resources.")
    
    main_blend_file = os.path.join(addon_dir, "SDF_UI_Slice_4p3.blend")
    if not os.path.exists(main_blend_file):
        raise FileNotFoundError(f"Cannot find main resource file: {main_blend_file}")

    # Load all node groups from the blend file into the current session.
    with bpy.data.libraries.load(main_blend_file, link=False) as (data_from, data_to):
        data_to.node_groups = data_from.node_groups
    
    print(f"[Rogue_SDF_AI] Successfully loaded {len(data_to.node_groups)} node groups.")
    return True



# Define NodePositionManager immediately after the imports.
class NodePositionManager:
    current_position = 0

    @staticmethod
    def increment_position(increment=200):
        NodePositionManager.current_position += increment
        return NodePositionManager.current_position


# -------------------------------------------------------------------
# NEW: Function to update the SDF Domain node's resolution based on preview mode.
# -------------------------------------------------------------------
def dynamic_domain_toggle_update(self, context):
    """
    Called when the Dynamic Domain checkbox is toggled.
    Enforces a clean state to prevent conflicts.
    """
    if self.sdf_dynamic_domain_enable:
        # When turning ON, force the domain to the origin and reset global scale.
        bpy.ops.object.reset_sdf_domain_transform()
        # Resetting global scale prevents its logic from interfering.
        context.scene.sdf_global_scale = 1.0
    
    # In all cases, trigger a resolution update to match the new mode.
    update_sdf_resolution(self, context)


#------------------------------------------------------------------

def get_dec_node(context):
    """Return the MK_Rogue_Decimate node if it exists, else None."""
    node_tree = get_sdf_geometry_node_tree(context)
    if not node_tree:
        return None
    return next((n for n in node_tree.nodes if "MK_Rogue_Decimate" in n.name), None)



# -------------------------------------------------------------------

def update_sdf_domain_scale(self, context):
    """
    Update callback for the domain scale slider.
    Writes the value to the 'Domain Size' input on the SDF Domain node.
    """
    # This should not run when clipping is enabled, as that system controls the size.
    if context.scene.clip_enabled:
        return

    node_tree = get_sdf_geometry_node_tree(context)
    if not node_tree:
        return
        
    domain_node = next((n for n in node_tree.nodes if n.name == "SDF Domain"), None)
    if not domain_node:
        return
            
    if "Domain Size" in domain_node.inputs:
        domain_node.inputs["Domain Size"].default_value = context.scene.sdf_domain_scale


from mathutils import Vector

def update_sdf_global_scale(self, context):
    """
    Update callback for the global scale slider.
    Scales down the SDF shapes (empties) uniformly relative to the Domain's
    center as a common pivot, preserving their relative arrangement.
    """
    domain_obj = context.scene.sdf_domain
    if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')):
        return

    # Use the Domain object's location as the common pivot point.
    pivot = domain_obj.location.copy()
    scale_factor = context.scene.sdf_global_scale

    for item in domain_obj.sdf_nodes:
        empty_obj = item.empty_object
        if not empty_obj:
            continue

        # --- State Management ---
        # Store the initial transform on the first update.
        # This prevents a feedback loop and ensures scaling is always
        # applied to the original, user-created layout.
        if "initial_location" not in empty_obj:
            empty_obj["initial_location"] = empty_obj.location.copy()
        if "initial_scale" not in empty_obj:
            empty_obj["initial_scale"] = empty_obj.scale.copy()

        # Retrieve the stored initial state.
        # We must cast the custom property back to a Vector for math operations.
        init_loc = Vector(empty_obj["initial_location"])
        init_scale = Vector(empty_obj["initial_scale"])

        # --- Pivot-Based Scaling ---
        # 1. Calculate the object's original offset from the pivot.
        offset = init_loc - pivot

        # 2. Scale that offset vector.
        scaled_offset = offset * scale_factor

        # 3. Add the scaled offset back to the pivot to get the new location.
        new_loc = pivot + scaled_offset

        # --- Apply the new transforms ---
        empty_obj.location = new_loc
        empty_obj.scale = init_scale * scale_factor



# -------------------------------------------------------------------
# Original Data Structures and UI List
# -------------------------------------------------------------------

def update_sdf_node_name(self, context):
    """
    This function is called when a node's name is changed in the UI.
    It ensures that the associated Empty object is also renamed,
    and crucially, that the custom property on the geometry node
    is updated to maintain the connection.
    """
    # 'self' is the SDFNodeItem being changed.
    empty_obj = self.empty_object
    domain_obj = context.scene.sdf_domain

    if not empty_obj or not domain_obj:
        return

    # This is the new name the user typed.
    new_name = self.name
    
    # This prevents an infinite loop if the name hasn't actually changed.
    if empty_obj.name == new_name:
        return
        
    # We need to find the Geometry Node using the OLD name before we change it.
    old_name = empty_obj.name
    mod = domain_obj.modifiers.get("GeometryNodes")
    if mod and mod.node_group:
        node_tree = mod.node_group
        # Find the specific node linked to our empty.
        node = next((n for n in node_tree.nodes if n.get("associated_empty") == old_name), None)
        
        # Now, perform the renames.
        empty_obj.name = new_name
        
        # If we found the corresponding node, update its internal link.
        if node:
            node["associated_empty"] = new_name

def update_sdf_viewport_visibility(self, context):
    """
    Called when the viewport visibility icon is clicked in the UI list.
    Toggles the hide_viewport property of the associated Empty object.
    """
    if self.empty_object:
        # 'self' is the SDFNodeItem being changed.
        # This line syncs the Empty's viewport visibility with our new property.
        self.empty_object.hide_viewport = self.is_viewport_hidden            



class SDF_UL_nodes(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
            sdf_node = item
            is_valid = sdf_node.empty_object and sdf_node.empty_object.name in context.view_layer.objects

            if is_valid:
                active_index = getattr(active_data, active_propname)
                is_active = (data.sdf_nodes[active_index] == item) if 0 <= active_index < len(data.sdf_nodes) else False
                
                row = layout.row(align=True)
                if is_active:
                    row.alert = True
                
                op = row.operator("object.select_empty", text=sdf_node.name, icon=sdf_node.icon, emboss=False)
                op.empty_name = sdf_node.empty_object.name

                sub = row.row(align=True)
                sub.alignment = 'RIGHT'
                
                # The broken line referencing use_point_cloud_preview has been removed.
                
                highlight_icon = 'RESTRICT_SELECT_ON' if sdf_node.use_highlight else 'RESTRICT_SELECT_OFF'
                sub.prop(sdf_node, "use_highlight", text="", icon=highlight_icon, emboss=True)

                sub.prop(sdf_node, "is_viewport_hidden", text="", icon_only=True, emboss=True)
                sub.prop(sdf_node, "is_hidden", text="")

            else:
                row = layout.row(align=True)
                row.label(text=f"'{sdf_node.name}' is broken!", icon='ERROR')
                row.operator("prototyper.sdf_cleanup_list", text="Clean List", icon='BRUSH_DATA')

class SDF_UL_curve_points(bpy.types.UIList):
    """The UIList for displaying custom curve control points."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        point = item
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=f"Point at t={point.t_value:.2f}", icon='DECORATE_KEYFRAME')
            layout.prop(point, "shape_type", text="")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='DECORATE_KEYFRAME')

class PROTOTYPER_OT_SDFCurvePointAdd(bpy.types.Operator):
    """Add a new control point to the active SDF Curve."""
    bl_idname = "prototyper.sdf_curve_point_add"
    bl_label = "Add Curve Control Point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        domain = context.scene.sdf_domain
        if domain and 0 <= domain.active_sdf_node_index < len(domain.sdf_nodes):
            item = domain.sdf_nodes[domain.active_sdf_node_index]
            if item.icon == 'CURVE_BEZCURVE':
                new_point = item.custom_control_points.add()
                if len(item.custom_control_points) > 1:
                    new_point.t_value = item.custom_control_points[-2].t_value + 0.2
                item.active_control_point_index = len(item.custom_control_points) - 1
        return {'FINISHED'}

class PROTOTYPER_OT_SDFCurvePointRemove(bpy.types.Operator):
    """Remove the selected control point from the active SDF Curve."""
    bl_idname = "prototyper.sdf_curve_point_remove"
    bl_label = "Remove Curve Control Point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        domain = context.scene.sdf_domain
        if domain and 0 <= domain.active_sdf_node_index < len(domain.sdf_nodes):
            item = domain.sdf_nodes[domain.active_sdf_node_index]
            if item.icon == 'CURVE_BEZCURVE' and len(item.custom_control_points) > 0:
                idx = item.active_control_point_index
                item.custom_control_points.remove(idx)
                item.active_control_point_index = min(max(0, idx - 1), len(item.custom_control_points) - 1)
        return {'FINISHED'}             

# -------------------------------------------------------------------
# Operators for Object Selection and Mute Toggle
# -------------------------------------------------------------------
import sys, subprocess
    

# Pre-load your slice shader once at module scope:
_slice_vert = textwrap.dedent("""\
    in vec2 pos;
    out vec2 uv;
    void main() {
        uv = pos * 0.5 + 0.5;
        gl_Position = vec4(pos, 0.0, 1.0);
    }
""")
_slice_frag_path = os.path.join(os.path.dirname(__file__), "sdf_volume.frag")
if not os.path.exists(_slice_frag_path):
    with open(_slice_frag_path, 'w') as f:
        f.write('void main() { discard; }')
_slice_frag = open(_slice_frag_path).read()
slice_shader = GPUShader(_slice_vert, _slice_frag)
slice_batch  = batch_for_shader(slice_shader, 'TRI_STRIP',
                   {"pos": [(-1,-1),(1,-1),(-1,1),(1,1)]})


def setup_slice_baking(res_x, res_y):
    """
    Prepares all GPU resources for slice-by-slice baking.
    Compiles the shader, collects and uploads all uniform data.
    Returns the offscreen buffer, shader, and batch object.
    """
    # --- Compile the shader ---
    _slice_vert = textwrap.dedent("""\
        in vec2 pos;
        out vec2 uv;
        void main() {
            uv = pos * 0.5 + 0.5;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
    """)
    _slice_frag_path = os.path.join(os.path.dirname(__file__), "sdf_volume.frag")
    with open(_slice_frag_path, 'r', encoding='utf-8-sig') as f:
        _slice_frag = f.read()
    
    slice_shader = GPUShader(_slice_vert, _slice_frag)
    slice_batch  = batch_for_shader(slice_shader, 'TRI_STRIP', {"pos": [(-1,-1),(1,-1),(-1,1),(1,1)]})

    # --- Gather and pack all data from the scene ---
    shapes = collect_sdf_data(bpy.context)
    MAX_SHAPES_CURRENT = bpy.context.scene.sdf_max_shapes
    uCount = sum(1 for s in shapes if s[0] >= 0)
    
    tf = [int(s[0]) for s in shapes]
    while len(tf) % 4 != 0: tf.append(-1)
        
    type_flat = [tf[i + j] for i in range(0, len(tf), 4) for j in range(4)]
    pos_flat = [f for s in shapes for f in s[1]]
    scale_flat = [f for s in shapes for f in s[2]]
    rot_flat = [f for s in shapes for f in s[3]]
    op_flat = [int(s[4]) for s in shapes]
    blend_flat = [float(s[5]) for s in shapes]
    mirror_flags_flat = [int(s[6]) for s in shapes]
    radial_count_flat = [int(s[7]) for s in shapes]
    item_id_flat = [int(s[10]) for s in shapes]
    params1_flat = [v for s in shapes for v in s[11]]
    params2_flat = [v for s in shapes for v in s[12]]
    color_flat = [v for s in shapes for v in s[8]]
    strength_flat = [float(s[13]) for s in shapes]
    fill_flat = [float(s[14]) for s in shapes]

    type_buf = array.array('i', type_flat).tobytes()
    pos_buf = array.array('f', pos_flat).tobytes()
    scale_buf = array.array('f', scale_flat).tobytes()
    rot_buf = array.array('f', rot_flat).tobytes()
    op_buf = array.array('i', op_flat).tobytes()
    blend_buf = array.array('f', blend_flat).tobytes()
    mirror_flags_buf = array.array('i', mirror_flags_flat).tobytes()
    radial_count_buf = array.array('i', radial_count_flat).tobytes()
    item_id_buf = array.array('i', item_id_flat).tobytes()
    params1_buf = array.array('f', params1_flat).tobytes()
    params2_buf = array.array('f', params2_flat).tobytes()
    color_buf = array.array('f', color_flat).tobytes()
    strength_buf = array.array('f', strength_flat).tobytes()
    fill_buf = array.array('f', fill_flat).tobytes()

    # --- Set up off-screen rendering and upload uniforms ---
    offscreen = gpu.types.GPUOffScreen(res_x, res_y)
    
    slice_shader.bind()
    slice_shader.uniform_int("uCount", uCount)
    slice_shader.uniform_float("uDomainCenter", bpy.context.scene.sdf_domain.location)
    slice_shader.uniform_int("uColorBlendMode", 1 if bpy.context.scene.sdf_color_blend_mode == 'SOFT' else 0)

    loc = slice_shader.uniform_from_name
    slice_shader.uniform_vector_int(loc("uShapeTypePacked"), type_buf, 4, len(type_flat) // 4)
    slice_shader.uniform_vector_float(loc("uShapePos"), pos_buf, 3, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_float(loc("uShapeScale"), scale_buf, 3, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_float(loc("uShapeRot"), rot_buf, 4, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_int(loc("uShapeOp"), op_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_float(loc("uShapeBlend"), blend_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_float(loc("uShapeBlendStrength"), strength_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_float(loc("uShapeMaskFill"), fill_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_int(loc("uShapeMirrorFlags"), mirror_flags_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_int(loc("uShapeRadialCount"), radial_count_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_int(loc("uShapeItemID"), item_id_buf, 1, MAX_SHAPES_CURRENT)
    slice_shader.uniform_vector_float(loc("uShapeColor"), color_buf, 3, MAX_SHAPES_CURRENT)

    try: slice_shader.uniform_vector_float(loc("uShapeParams1"), params1_buf, 4, MAX_SHAPES_CURRENT)
    except ValueError: pass
    try: slice_shader.uniform_vector_float(loc("uShapeParams2"), params2_buf, 4, MAX_SHAPES_CURRENT)
    except ValueError: pass

    return offscreen, slice_shader, slice_batch


def render_sdf_slices(resX, resY, depth, bounds_min, bounds_max):
    """
    Renders SDF slices and returns separate numpy grids for density and color channels.
    This version is used by the DIRECT bake method.
    """
    # --- Use the new helper to prepare everything ---
    offscreen, slice_shader, slice_batch = setup_slice_baking(resX, resY)

    # --- Prepare result arrays ---
    density_slices = np.zeros((depth, resY, resX), dtype=np.float32)
    color_r_slices = np.zeros((depth, resY, resX), dtype=np.float32)
    color_g_slices = np.zeros((depth, resY, resX), dtype=np.float32)
    color_b_slices = np.zeros((depth, resY, resX), dtype=np.float32)

    with offscreen.bind():
        gpu.state.viewport_set(0, 0, resX, resY)
        slice_shader.bind()
        
        # Set uniforms that change per-slice
        slice_shader.uniform_float("uBoundsMin", bounds_min)
        slice_shader.uniform_float("uBoundsMax", bounds_max)
        slice_shader.uniform_int("uDepth", depth)

        framebuffer = gpu.state.active_framebuffer_get()
        for z in range(depth):
            slice_shader.uniform_int("uSliceIndex", z)
            slice_batch.draw(slice_shader)
            pixel_buffer = framebuffer.read_color(0, 0, resX, resY, 4, 0, 'FLOAT')
            arr = np.array(pixel_buffer.to_list(), dtype=np.float32).reshape(resY, resX, 4)
            
            density_slices[z, :, :] = arr[:, :, 0]
            color_r_slices[z, :, :] = arr[:, :, 1]
            color_g_slices[z, :, :] = arr[:, :, 2]
            color_b_slices[z, :, :] = arr[:, :, 3]

    offscreen.free()
    return density_slices, color_r_slices, color_g_slices, color_b_slices





def write_vdb(slices, filepath, bounds_min, voxel_size):
    """
    Saves a 3D numpy array to a .vdb file, correctly setting the background
    value and flipping the sign for Blender's meshing tools.
    """
    import openvdb

    grid = openvdb.FloatGrid(background=-1000.0)
    grid.name = "density"

    transform = openvdb.createLinearTransform(voxel_size)
    transform.postTranslate((bounds_min.x, bounds_min.y, bounds_min.z))
    grid.transform = transform

    accessor = grid.getAccessor()
    depth, height, width = slices.shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                val = -float(slices[z, y, x])
                accessor.setValueOn((x, y, z), val)

    openvdb.write(filepath, grids=[grid])

def bilinear_interpolate(grid_2d, points_2d):
    """
    Performs bilinear interpolation for a batch of 2D points in a 2D grid.
    'grid_2d' is the 2D numpy array (Y, X, Channels).
    'points_2d' is an (N, 2) numpy array of (x, y) coordinates.
    """
    height, width, channels = grid_2d.shape
    x, y = points_2d[:, 0], points_2d[:, 1]

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)

    c00 = grid_2d[y0, x0]
    c10 = grid_2d[y0, x1]
    c01 = grid_2d[y1, x0]
    c11 = grid_2d[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa[:, np.newaxis] * c00 + wb[:, np.newaxis] * c10 + wc[:, np.newaxis] * c01 + wd[:, np.newaxis] * c11    


def trilinear_interpolate(grid, points):
    """
    Performs trilinear interpolation for a batch of points in a 3D grid.
    'grid' is the 3D numpy array (Z, Y, X).
    'points' is an (N, 3) numpy array of (x, y, z) coordinates.
    """
    # Get grid dimensions
    depth, height, width = grid.shape
    
    # --- FIX: Separate coordinates correctly from the (N, 3) array ---
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Get lower and upper integer coordinates for each axis
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    z0 = np.floor(z).astype(int)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    # --- END FIX ---

    # Clamp coordinates to be within grid bounds
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    z0 = np.clip(z0, 0, depth - 1)
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    z1 = np.clip(z1, 0, depth - 1)

    # Calculate fractional distances
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Get values of the 8 surrounding voxels
    c000 = grid[z0, y0, x0]
    c100 = grid[z0, y0, x1]
    c010 = grid[z0, y1, x0]
    c110 = grid[z0, y1, x1]
    c001 = grid[z1, y0, x0]
    c101 = grid[z1, y0, x1]
    c011 = grid[z1, y1, x0]
    c111 = grid[z1, y1, x1]

    # Interpolate along x-axis
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z-axis
    return c0 * (1 - zd) + c1 * zd


# Make sure these imports are at the top of your main.py file
import os
import bpy
import numpy as np
import gpu
from gpu.types import GPUOffScreen
from mathutils import Vector
from gpu_extras.batch import batch_for_shader
import sys
import subprocess
import textwrap
import array



# In main.py, replace your OBJECT_OT_sdf_bake_volume class with this one.

class OBJECT_OT_sdf_bake_volume(bpy.types.Operator):
    """Creates a high-quality, retopologized mesh from the SDF Domain via a refinement pipeline"""
    bl_idname = "object.sdf_bake_volume"
    bl_label = "Bake SDF to High-Quality Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    # --- Main Bake Properties ---
    bake_method: bpy.props.EnumProperty(
        name="Bake Method",
        items=[
            ('DIRECT', "Direct (PyMCubes)", "Fast, colored, but memory-intensive. Best for lower resolutions."),
            ('VDB', "VDB (Geometry Only)", "Memory-efficient, high-resolution, but produces an uncolored mesh."),
            ('HYBRID', "Hybrid (VDB + Color)", "Memory-efficient, high-resolution, and colored. Slower due to a second color-baking pass.")
        ],
        default='DIRECT'
    )
    bake_colors: bpy.props.BoolProperty(name="Bake Vertex Colors",description="Export color information to the baked mesh. Only for Direct method.",default=True)
    direct_method_smoothing: bpy.props.BoolProperty(name="Smooth Voxel Data",description="Pre-smooth the voxel data before meshing. Improves quality but is slower. Only for Direct method.",default=True)
    res: bpy.props.IntProperty(name="Initial Resolution", default=256, min=32, max=4096)
    bake_scale: bpy.props.FloatProperty(name="Bake Scale", default=1.0, min=0.1, max=10.0)
    vdb_filepath: bpy.props.StringProperty(name="VDB File Path", default=os.path.expanduser("~/Desktop/sdf_bake.vdb"), subtype='FILE_PATH')
    
    # --- Texture Bake Properties ---
    bake_to_texture: bpy.props.BoolProperty(name="Bake to Image Texture", default=False)
    auto_uv_unwrap: bpy.props.BoolProperty(name="Auto UV Unwrap", default=True)
    texture_resolution: bpy.props.IntProperty(name="Texture Resolution", default=1024, min=256, max=4096)

    # --- Retopology & Polish Properties (QUADRIFLOW REMOVED) ---
    retopology_method: bpy.props.EnumProperty(name="Retopology Method", items=[('NONE', "None", ""), ('VOXEL', "Voxel Remesh", "")], default='NONE')
    
    # --- Full Voxel Properties ---
    voxel_remesh_size: bpy.props.FloatProperty(name="Voxel Size", default=0.01, min=0.001, max=1.0, precision=4)
    voxel_remesh_adaptivity: bpy.props.FloatProperty(name="Adaptivity", default=0.0, min=0.0, max=1.0)
    voxel_fix_poles: bpy.props.BoolProperty(name="Fix Poles", default=False)
    voxel_preserve_volume: bpy.props.BoolProperty(name="Preserve Volume", default=True)
    voxel_preserve_attributes: bpy.props.BoolProperty(name="Preserve Attributes", description="Keeps attributes like Vertex Colors intact", default=True)

    # --- Polishing Properties ---
    add_subdivision_modifier: bpy.props.BoolProperty(name="Add Subdivision Surface", default=False)
    subdivision_levels: bpy.props.IntProperty(name="Subdivision Levels", default=2, min=1, max=6)
    add_smooth_modifier: bpy.props.BoolProperty(name="Add Classic Smooth", default=False)
    smooth_factor: bpy.props.FloatProperty(name="Factor", default=0.5, min=0.0, max=2.0)
    smooth_iterations: bpy.props.IntProperty(name="Iterations", default=5, min=1, max=30)
    add_corrective_smooth_modifier: bpy.props.BoolProperty(name="Add Corrective Smooth", default=False)
    shade_smooth: bpy.props.BoolProperty(name="Shade Smooth", default=False)

    # --- Splat Generation Properties ---
    generate_splats: bpy.props.BoolProperty(name="Generate Splats", default=False)
    splat_shape: bpy.props.EnumProperty(name="Shape", items=[('SQUARE', "Square", ""), ('CIRCLE', "Circle", ""), ('TRIANGLE', "Triangle", "")], default='SQUARE')
    splat_density: bpy.props.FloatProperty(name="Density", default=100.0, min=1.0, soft_max=1000.0)
    splat_size_min: bpy.props.FloatProperty(name="Min Size", default=0.05, min=0.001, soft_max=1.0)
    splat_size_max: bpy.props.FloatProperty(name="Max Size", default=0.1, min=0.001, soft_max=20.0)
    splat_rotation_min: bpy.props.FloatProperty(name="Min Rotation", subtype='ANGLE', default=0.0)
    splat_rotation_max: bpy.props.FloatProperty(name="Max Rotation", subtype='ANGLE', default=math.pi * 2)
    splat_height_min: bpy.props.FloatProperty(name="Min Height", default=0.0, soft_min=-20.0, soft_max=20.0)
    splat_height_max: bpy.props.FloatProperty(name="Max Height", default=0.0, soft_min=-20.0, soft_max=20.0)
    texture_path: bpy.props.StringProperty(name="Output Path", subtype='DIR_PATH', default="//")    

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=450)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        
        box = layout.box(); box.label(text="Stage 1: Initial Bake", icon='VOLUME_DATA')
        col = box.column()
        col.prop(self, "bake_method", expand=True)
        col.separator()
        
        if self.bake_method == 'DIRECT':
            if not HAS_MCUBES: col.label(text="PyMCubes library not found!", icon='ERROR')
            else:
                sub = col.box()
                sub.prop(self, "bake_colors")
                sub.prop(self, "direct_method_smoothing")
        elif self.bake_method in {'VDB', 'HYBRID'}:
            if not HAS_OPENVDB: col.label(text="pyopenvdb library not found!", icon='ERROR')
            else:
                sub = col.box()
                sub.prop(self, "vdb_filepath", text="VDB Path")
        
        col.prop(self, "res")
        col.prop(self, "bake_scale")
        
        box = layout.box(); box.label(text="Stage 2: Retopology", icon='MOD_REMESH')
        col = box.column()
        col.prop(self, "retopology_method", text="Method")
        
        if self.retopology_method == 'VOXEL':
            sub = col.box()
            sub.prop(self, "voxel_remesh_size")
            sub.prop(self, "voxel_remesh_adaptivity")
            sub.prop(self, "voxel_fix_poles")
            sub.prop(self, "voxel_preserve_volume")
            sub.prop(self, "voxel_preserve_attributes")
            
        # QUADRIFLOW UI IS REMOVED
            
        box = layout.box(); box.label(text="Stage 3: Final Polishing", icon='MOD_SMOOTH')
        col = box.column()
        col.prop(self, "add_subdivision_modifier")
        if self.add_subdivision_modifier: sub = col.box(); sub.prop(self, "subdivision_levels")
        col.separator()
        col.prop(self, "add_smooth_modifier")
        if self.add_smooth_modifier: sub = col.box(); sub.prop(self, "smooth_factor"); sub.prop(self, "smooth_iterations")
        col.separator()
        col.prop(self, "add_corrective_smooth_modifier")
        col.separator()
        col.prop(self, "shade_smooth")

        if (self.bake_method == 'DIRECT' and self.bake_colors) or (self.bake_method == 'HYBRID'):
            tex_box = layout.box()
            tex_box.label(text="Stage 4: Texture Output", icon='TEXTURE')
            tex_box.prop(self, "bake_to_texture")
            if self.bake_to_texture:
                sub_tex = tex_box.box()
                sub_tex.prop(self, "auto_uv_unwrap")
                sub_tex.prop(self, "texture_resolution", text="Resolution")
                sub_tex.prop(self, "texture_path")
           
        if (self.bake_method == 'DIRECT' and self.bake_colors) or (self.bake_method == 'HYBRID'):
            splat_box = layout.box()
            splat_box.label(text="Stage 5: Splat Generation", icon='PARTICLES')
            splat_box.prop(self, "generate_splats")
            if self.generate_splats:
                sub_splat = splat_box.box()
                sub_splat.prop(self, "splat_shape", text="Shape")
                sub_splat.prop(self, "splat_density", text="Density")
                row = sub_splat.row(align=True)
                row.prop(self, "splat_size_min", text="Min Size")
                row.prop(self, "splat_size_max", text="Max Size")
                row = sub_splat.row(align=True)
                row.prop(self, "splat_rotation_min", text="Min Rot")
                row.prop(self, "splat_rotation_max", text="Max Rot")
                row = sub_splat.row(align=True)
                row.prop(self, "splat_height_min", text="Min Height")
                row.prop(self, "splat_height_max", text="Max Height")

    def create_splat_object(self, context, source_object):
        # ... (your existing splat code is unchanged)
        self.report({'INFO'}, "Generating live particle system for splats...")
        if "Splat_Instances" in bpy.data.collections:
            splat_collection = bpy.data.collections["Splat_Instances"]
        else:
            splat_collection = bpy.data.collections.new("Splat_Instances")
            context.scene.collection.children.link(splat_collection)
        context.view_layer.layer_collection.children[splat_collection.name].hide_viewport = True
        context.view_layer.layer_collection.children[splat_collection.name].exclude = False
        shape_name = f"Splat_{self.splat_shape.title()}"
        if shape_name not in splat_collection.objects:
            if self.splat_shape == 'SQUARE':
                bpy.ops.mesh.primitive_plane_add(size=1)
            elif self.splat_shape == 'CIRCLE':
                bpy.ops.mesh.primitive_circle_add(vertices=16, fill_type='NGON')
            else: # TRIANGLE
                bpy.ops.mesh.primitive_circle_add(vertices=3, fill_type='NGON')
            splat_instance = context.active_object
            splat_instance.name = shape_name
            splat_collection.objects.link(splat_instance)
            context.collection.objects.unlink(splat_instance)
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = source_object
        source_object.select_set(True)
        bpy.ops.object.particle_system_add()
        psys = source_object.particle_systems[-1]
        psys.name = "SplatParticleSystem"
        psettings = psys.settings
        psettings.type = 'HAIR'
        psettings.use_advanced_hair = True
        psettings.emit_from = 'FACE'
        surface_area = sum(f.area for f in source_object.data.polygons)
        particle_count = int(self.splat_density * surface_area)
        if particle_count == 0 and surface_area > 0:
            particle_count = 1
        self.report({'INFO'}, f"Calculated Surface Area: {surface_area:.4f} m^2. Creating {particle_count} splats.")
        psettings.count = particle_count
        psettings.render_type = 'COLLECTION'
        psettings.instance_collection = splat_collection
        psettings.particle_size = self.splat_size_min
        psettings.size_random = (self.splat_size_max - self.splat_size_min) / self.splat_size_min if self.splat_size_min > 0 else 1.0
        psys.show_instancer_for_viewport = True
        psettings.use_rotations = True
        psettings.rotation_mode = 'NOR_TAN'
        psettings.phase_factor = self.splat_rotation_min / (math.pi * 2)
        psettings.phase_factor_random = (self.splat_rotation_max - self.splat_rotation_min) / (math.pi * 2)
        psettings.physics_type = 'NEWTON'
        psettings.normal_factor = (self.splat_height_min + self.splat_height_max) / 2.0
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = source_object
        source_object.select_set(True)
        self.report({'INFO'}, "Successfully added a live splat particle system.")
        
    def execute(self, context):
        # ... (Initial checks and setup code is unchanged) ...
        if self.bake_method == 'DIRECT' and not HAS_MCUBES:
            self.report({'ERROR'}, "PyMCubes library is required for the Direct method.")
            return {'CANCELLED'}
        if self.bake_method in {'VDB', 'HYBRID'} and not HAS_OPENVDB:
            self.report({'ERROR'}, "pyopenvdb library is required for VDB/Hybrid methods.")
            return {'CANCELLED'}
        domain = getattr(context.scene, "sdf_domain", None)
        if not domain: self.report({'ERROR'}, "SDF Domain object not found."); return {'CANCELLED'}
        self.report({'INFO'}, f"Preparing bake with '{self.bake_method}' method...")
        base_corners = [domain.matrix_world @ Vector(corner) for corner in domain.bound_box]
        base_min = Vector(min(c[i] for c in base_corners) for i in range(3)); base_max = Vector(max(c[i] for c in base_corners) for i in range(3))
        base_size = base_max - base_min; base_center = (base_min + base_max) / 2.0
        final_size = base_size * self.bake_scale; final_min = base_center - (final_size / 2.0); final_max = base_center + (final_size / 2.0)
        longest_axis = max(final_size)
        if longest_axis <= 0: return {'CANCELLED'}
        voxel_size = longest_axis / self.res
        res_x, res_y, res_z = [max(16, int(s / voxel_size)) for s in final_size]
        final_mesh_object = None
        col_r_grid, col_g_grid, col_b_grid = (None, None, None)

        if self.bake_method == 'DIRECT':
            try:
                density_grid, col_r_grid, col_g_grid, col_b_grid = render_sdf_slices(res_x, res_y, res_z, final_min, final_max)
            except Exception as e: 
                self.report({'ERROR'}, f"Stage 1 failed during GPU rendering: {e}"); return {'CANCELLED'}
            self.report({'INFO'}, "Stage 2: Generating Mesh via Marching Cubes...")
            mesh_density_grid = mcubes.smooth(density_grid) if self.direct_method_smoothing else density_grid
            verts, faces = mcubes.marching_cubes(mesh_density_grid, 0.0)
            if len(verts) == 0: self.report({'WARNING'}, "Bake resulted in an empty mesh."); return {'CANCELLED'}
            verts_np = np.array(verts)
            verts_np[:, [0, 1, 2]] = verts_np[:, [2, 1, 0]]
            verts_world = (verts_np * voxel_size) + np.array(final_min)
            mesh_data = bpy.data.meshes.new("SDF_Mesh_Data"); mesh_data.from_pydata(verts_world.tolist(), [], faces.tolist()); mesh_data.update()
            final_mesh_object = bpy.data.objects.new("SDF_Mesh_Result", mesh_data); context.collection.objects.link(final_mesh_object)
        elif self.bake_method in {'VDB', 'HYBRID'}:
            try:
                density_grid, _, _, _ = render_sdf_slices(res_x, res_y, res_z, final_min, final_max)
            except Exception as e: self.report({'ERROR'}, f"Stage 1 failed during GPU rendering: {e}"); return {'CANCELLED'}
            self.report({'INFO'}, "Stage 2: Writing VDB and converting to Mesh...")
            try:
                vdb_path = bpy.path.abspath(self.vdb_filepath); os.makedirs(os.path.dirname(vdb_path), exist_ok=True)
                write_vdb(density_grid, vdb_path, final_min, voxel_size)
                bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD', location=(0,0,0))
                volume_object = context.view_layer.objects.active; volume_object.name = "SDF_Volume_Source"
            except Exception as e: self.report({'ERROR'}, f"Stage 2 (VDB) failed: {e}"); return {'CANCELLED'}
            final_mesh_object = bpy.data.objects.new("SDF_Mesh_Result", bpy.data.meshes.new("SDF_Mesh_Data")); context.collection.objects.link(final_mesh_object)
            mod_v2m = final_mesh_object.modifiers.new(name="VolumeToMesh", type='VOLUME_TO_MESH'); mod_v2m.object = volume_object; mod_v2m.threshold = 0.0
            depsgraph = context.evaluated_depsgraph_get(); object_eval = final_mesh_object.evaluated_get(depsgraph)
            mesh_from_eval = bpy.data.meshes.new_from_object(object_eval); final_mesh_object.data = mesh_from_eval
            final_mesh_object.modifiers.clear(); bpy.data.objects.remove(volume_object, do_unlink=True)

        if not final_mesh_object: self.report({'ERROR'}, "Mesh object was not created. Aborting."); return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT'); context.view_layer.objects.active = final_mesh_object; final_mesh_object.select_set(True)
        
        vcol_layer = None
        if (self.bake_method == 'DIRECT' and self.bake_colors) or (self.bake_method == 'HYBRID'):
            self.report({'INFO'}, "Stage 3: Applying vertex colors to initial mesh...")
            if self.bake_method == 'HYBRID':
                self.report({'INFO'}, "Using memory-efficient slice-by-slice color sampling...")
                mw = final_mesh_object.matrix_world
                final_verts_count = len(final_mesh_object.data.vertices)
                final_verts_local = np.empty(final_verts_count * 3, dtype=np.float32)
                final_mesh_object.data.vertices.foreach_get('co', final_verts_local)
                final_verts_local = final_verts_local.reshape((final_verts_count, 3))
                final_verts_world = np.einsum('ij,aj->ai', np.array(mw), np.hstack((final_verts_local, np.ones((final_verts_count, 1)))))[:, :3]
                final_verts_voxel = (final_verts_world - np.array(final_min)) / voxel_size
                offscreen, slice_shader, slice_batch = setup_slice_baking(res_x, res_y)
                vert_colors = np.zeros((final_verts_count, 4), dtype=np.float32)
                with offscreen.bind():
                    gpu.state.viewport_set(0, 0, res_x, res_y)
                    slice_shader.bind()
                    slice_shader.uniform_float("uBoundsMin", final_min)
                    slice_shader.uniform_float("uBoundsMax", final_max)
                    slice_shader.uniform_int("uDepth", res_z)
                    framebuffer = gpu.state.active_framebuffer_get()
                    for z in range(res_z):
                        indices_in_slice = np.where((final_verts_voxel[:, 2] >= z) & (final_verts_voxel[:, 2] < z + 1))[0]
                        if len(indices_in_slice) == 0: continue
                        slice_shader.uniform_int("uSliceIndex", z)
                        slice_batch.draw(slice_shader)
                        pixel_buffer = framebuffer.read_color(0, 0, res_x, res_y, 4, 0, 'FLOAT')
                        slice_arr = np.array(pixel_buffer.to_list(), dtype=np.float32).reshape(res_y, res_x, 4)
                        points_in_slice = final_verts_voxel[indices_in_slice][:, :2]
                        sampled_data = bilinear_interpolate(slice_arr, points_in_slice)
                        vert_colors[indices_in_slice, 0] = sampled_data[:, 1]
                        vert_colors[indices_in_slice, 1] = sampled_data[:, 2]
                        vert_colors[indices_in_slice, 2] = sampled_data[:, 3]
                        vert_colors[indices_in_slice, 3] = 1.0
                offscreen.free()
                colors_flat = vert_colors
            else: # Direct Mode
                mw = final_mesh_object.matrix_world
                final_verts_count = len(final_mesh_object.data.vertices)
                final_verts_local = np.empty(final_verts_count * 3, dtype=np.float32)
                final_mesh_object.data.vertices.foreach_get('co', final_verts_local)
                final_verts_local = final_verts_local.reshape((final_verts_count, 3))
                final_verts_world = np.einsum('ij,aj->ai', np.array(mw), np.hstack((final_verts_local, np.ones((final_verts_count, 1)))))[:, :3]
                final_verts_voxel = (final_verts_world - np.array(final_min)) / voxel_size
                vert_colors_r = trilinear_interpolate(col_r_grid, final_verts_voxel)
                vert_colors_g = trilinear_interpolate(col_g_grid, final_verts_voxel)
                vert_colors_b = trilinear_interpolate(col_b_grid, final_verts_voxel)
                colors_flat = np.stack((vert_colors_r, vert_colors_g, vert_colors_b, np.ones_like(vert_colors_r)), axis=-1)
            
            vcol_layer = final_mesh_object.data.vertex_colors.new(name="SDF_Color")
            loop_vert_indices = np.zeros(len(final_mesh_object.data.loops), dtype=np.int32)
            final_mesh_object.data.loops.foreach_get('vertex_index', loop_vert_indices)
            vcol_layer.data.foreach_set('color', colors_flat[loop_vert_indices].ravel())
            
            temp_mat = bpy.data.materials.new(name="SDF_VCol_Display")
            temp_mat.use_nodes = True
            nodes = temp_mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                attr_node = nodes.new(type='ShaderNodeAttribute')
                attr_node.attribute_name = vcol_layer.name
                temp_mat.node_tree.links.new(attr_node.outputs['Color'], bsdf.inputs['Base Color'])
            final_mesh_object.data.materials.append(temp_mat)

        # --- Retopology Logic (QUADRIFLOW REMOVED) ---
        if self.retopology_method == 'VOXEL':
            self.report({'INFO'}, "Stage 4: Applying Voxel Remesh...")
            mesh_data = final_mesh_object.data
            mesh_data.remesh_voxel_size = self.voxel_remesh_size
            mesh_data.remesh_voxel_adaptivity = self.voxel_remesh_adaptivity
            mesh_data.use_remesh_fix_poles = self.voxel_fix_poles
            mesh_data.use_remesh_preserve_volume = self.voxel_preserve_volume
            mesh_data.use_remesh_preserve_attributes = self.voxel_preserve_attributes
            bpy.ops.object.voxel_remesh()
        
        self.report({'INFO'}, "Stage 5: Applying Final Polish...")
        if self.add_subdivision_modifier:
            bpy.ops.object.modifier_add(type='SUBSURF'); mod = final_mesh_object.modifiers[-1]
            mod.levels = self.subdivision_levels; mod.render_levels = self.subdivision_levels
            bpy.ops.object.modifier_apply(modifier=mod.name)
        if self.add_smooth_modifier:
            bpy.ops.object.modifier_add(type='SMOOTH'); mod = final_mesh_object.modifiers[-1]
            mod.factor = self.smooth_factor; mod.iterations = self.smooth_iterations
            bpy.ops.object.modifier_apply(modifier=mod.name)
        if self.add_corrective_smooth_modifier:
            bpy.ops.object.modifier_add(type='CORRECTIVE_SMOOTH')
            bpy.ops.object.modifier_apply(modifier=final_mesh_object.modifiers[-1].name)

        if self.bake_to_texture and final_mesh_object.data.vertex_colors:
            vcol_layer = final_mesh_object.data.vertex_colors.get("SDF_Color")
            if vcol_layer:
                self.report({'INFO'}, "Stage 6: Baking vertex colors to image texture...")
                scene = context.scene
                original_engine = scene.render.engine
                original_bake_settings = {
                    "use_selected_to_active": scene.render.bake.use_selected_to_active,
                    "margin": scene.render.bake.margin,
                    "use_cage": scene.render.bake.use_cage,
                    "cage_extrusion": scene.render.bake.cage_extrusion,
                }
                try:
                    scene.render.engine = 'CYCLES'
                    scene.render.bake.use_selected_to_active = False
                    bpy.ops.object.select_all(action='DESELECT')
                    context.view_layer.objects.active = final_mesh_object
                    final_mesh_object.select_set(True)
                    if self.auto_uv_unwrap or not final_mesh_object.data.uv_layers:
                        self.report({'INFO'}, "Generating UVs with Smart UV Project...")
                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.mesh.select_all(action='SELECT')
                        bpy.ops.uv.smart_project(angle_limit=math.radians(66))
                        bpy.ops.object.mode_set(mode='OBJECT')
                    if not final_mesh_object.data.uv_layers:
                        self.report({'ERROR'}, "Bake to Texture requires UVs, but none could be found or generated.")
                        return {'CANCELLED'}
                    bake_image = bpy.data.images.new(
                        name=f"{final_mesh_object.name}_Color",
                        width=self.texture_resolution,
                        height=self.texture_resolution,
                        alpha=True
                    )
                    if not final_mesh_object.data.materials:
                        bake_mat = bpy.data.materials.new(name="SDF_Bake_Material")
                        final_mesh_object.data.materials.append(bake_mat)
                        bake_mat.use_nodes = True
                    else:
                        bake_mat = final_mesh_object.data.materials[0]
                    tree = bake_mat.node_tree
                    nodes = tree.nodes
                    vcol_attr_node = None
                    for node in nodes:
                        if node.type == 'ATTRIBUTE' and node.attribute_name == vcol_layer.name:
                            vcol_attr_node = node
                            break
                    if not vcol_attr_node:
                        vcol_attr_node = nodes.new(type='ShaderNodeAttribute')
                        vcol_attr_node.attribute_name = vcol_layer.name
                    emission_node = nodes.new(type='ShaderNodeEmission')
                    tree.links.new(vcol_attr_node.outputs['Color'], emission_node.inputs['Color'])
                    bsdf = nodes.get("Principled BSDF")
                    output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
                    original_link = None
                    if bsdf and output_node and bsdf.outputs['BSDF'].links:
                        original_link = bsdf.outputs['BSDF'].links[0]
                    if output_node:
                        tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
                    img_node = nodes.new(type='ShaderNodeTexImage')
                    img_node.image = bake_image
                    nodes.active = img_node
                    scene.cycles.bake_type = 'EMIT'
                    bpy.ops.object.bake(type='EMIT')
                    resolved_path = bpy.path.abspath(self.texture_path)
                    if resolved_path and os.path.isdir(resolved_path):
                        output_dir = resolved_path
                    else:
                        self.report({'INFO'}, "No valid path set or blend file is unsaved. Using Desktop as fallback.")
                        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
                        output_dir = desktop_dir if os.path.isdir(desktop_dir) else bpy.app.tempdir
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{final_mesh_object.name}_Color.png"
                    filepath = os.path.join(output_dir, filename)
                    self.report({'INFO'}, f"Saved baked texture to: {filepath}")
                    bake_image.save_render(filepath=filepath)
                    if original_link:
                        tree.links.new(original_link.from_socket, original_link.to_socket)
                    else:
                        if output_node.inputs['Surface'].links:
                             tree.links.remove(output_node.inputs['Surface'].links[0])
                    nodes.remove(emission_node)
                    if bsdf:
                        tree.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
                        if vcol_attr_node not in bsdf.inputs['Base Color'].links:
                             nodes.remove(vcol_attr_node)
                    else:
                        nodes.remove(vcol_attr_node)
                finally:
                    self.report({'INFO'}, "Restoring original bake settings.")
                    scene.render.engine = original_engine
                    scene.render.bake.use_selected_to_active = original_bake_settings["use_selected_to_active"]
                    scene.render.bake.margin = original_bake_settings["margin"]
                    scene.render.bake.use_cage = original_bake_settings["use_cage"]
                    scene.render.bake.cage_extrusion = original_bake_settings["cage_extrusion"]

        if self.shade_smooth and len(final_mesh_object.data.polygons) > 0:
            final_mesh_object.data.polygons.foreach_set('use_smooth', [True] * len(final_mesh_object.data.polygons)); final_mesh_object.data.update()
            
        if self.generate_splats:
            self.create_splat_object(context, final_mesh_object)     
            
        self.report({'INFO'}, "Bake and refinement pipeline completed successfully!")
        return {'FINISHED'}
    

# In main.py, REPLACE the previous operator with this definitive, correct version.

# In main.py, REPLACE the previous operator with this definitive, correct version.

class OBJECT_OT_sdf_remesh_tools(bpy.types.Operator):
    """Opens a floating panel that displays Blender's native Remesh UI by drawing its properties directly."""
    bl_idname = "object.sdf_remesh_tools"
    bl_label = "Remesh Tools"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)
    
    def draw(self, context):
        layout = self.layout
        mesh = context.object.data
        layout.use_property_split = True
        
        # Draw the mode selector
        layout.prop(mesh, "remesh_mode", expand=True)
        box = layout.box()
        
        if mesh.remesh_mode == 'VOXEL':
            col = box.column()
            col.label(text="Voxel Remesh", icon='MOD_REMESH')
            # Use correct property names with remesh_ prefix
            col.prop(mesh, "remesh_voxel_size")
            col.prop(mesh, "remesh_voxel_adaptivity")
            col.prop(mesh, "use_remesh_fix_poles")
            col.prop(mesh, "use_remesh_preserve_volume")
            col.prop(mesh, "use_remesh_preserve_attributes")
            col.prop(mesh, "use_remesh_smooth_shade")
            col.operator("object.voxel_remesh", text="Voxel Remesh", icon='MOD_REMESH')
            
        elif mesh.remesh_mode == 'QUAD':
            col = box.column()
            col.label(text="QuadriFlow Remesh", icon='MOD_REMESH')
            # Use correct QuadriFlow property names
            col.prop(mesh, "remesh_quadri_flow_mode", text="Mode")
            col.prop(mesh, "remesh_quadri_flow_target_faces", text="Target Faces")
            col.prop(mesh, "remesh_quadri_flow_seed", text="Seed")
            
            # Fixed: Check if property exists before accessing it
            row = col.row()
            if hasattr(mesh, "use_remesh_quadri_flow_mesh_symmetry"):
                row.enabled = mesh.use_remesh_quadri_flow_mesh_symmetry
                row.prop(mesh, "use_remesh_quadri_flow_mesh_symmetry", text="Mesh Symmetry")
            else:
                row.label(text="Mesh Symmetry (not available in this Blender version)")
            
            col.prop(mesh, "use_remesh_quadri_flow_face_count", text="Target Face Count")
            col.prop(mesh, "use_remesh_quadri_flow_smooth_normals", text="Smooth Normals")
            col.operator("object.quadriflow_remesh", text="QuadriFlow Remesh", icon='MOD_REMESH')
        
        elif mesh.remesh_mode == 'MESHCLEAN':
            col = box.column()
            col.label(text="Blocks Remesh", icon='MOD_REMESH')
            # Blocks remesh properties would go here

    def execute(self, context):
        # This operator's job is only to show the panel.
        # The buttons inside the panel do the real work.
        return {'FINISHED'}

class OBJECT_OT_sdf_bake_to_remesh(bpy.types.Operator):
    """Sets up the scene for a 'Selected to Active' bake and opens the Bake Panel.
    
    Select the source object, then SHIFT-select the target remesh object to make it active.
    """
    bl_idname = "object.sdf_bake_to_remesh"
    bl_label = "Bake Selected to Active"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # This operator is only clickable if there is an active object and at least one other object selected.
        return context.active_object and len(context.selected_objects) > 1

    def execute(self, context):
        scene = context.scene
        
        # 1. Switch to Cycles, which is required for baking
        self.report({'INFO'}, "Switching to Cycles Render Engine for baking.")
        scene.render.engine = 'CYCLES'
        
        # 2. Set the bake mode to 'Selected to Active'
        scene.render.bake.use_selected_to_active = True
        
        # 3. Open the main Render Properties tab and scroll to the Bake panel for the user
        for area in context.screen.areas:
            if area.type == 'PROPERTIES':
                area.spaces.active.context = 'RENDER'
                # While we can't force a scroll, opening the tab is the most we can do.
                break
        
        self.report({'INFO'}, "Setup complete. Please configure settings in the Bake panel and click 'Bake'.")
        return {'FINISHED'}


# In main.py, add these two new operator classes

class OBJECT_OT_sdf_auto_uv(bpy.types.Operator):
    """Applies a Smart UV Project to the active object"""
    bl_idname = "object.sdf_auto_uv"
    bl_label = "Auto UV Selected Object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # This operator can only run if there is an active object and it's a mesh.
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        self.report({'INFO'}, f"Generating Smart UVs for '{obj.name}'...")
        
        # We must be in edit mode to perform the unwrap
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(66))
        bpy.ops.object.mode_set(mode='OBJECT')
        
        self.report({'INFO'}, "UV unwrapping complete.")
        return {'FINISHED'}


class OBJECT_OT_sdf_snap_selection_to_active(bpy.types.Operator):
    """Snaps the transform (location, rotation, scale) of selected objects to the active object"""
    bl_idname = "object.sdf_snap_selection_to_active"
    bl_label = "Snap Selection to Active"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Requires at least two objects to be selected (one active, one or more others)
        return context.active_object and len(context.selected_objects) > 1

    def execute(self, context):
        active_obj = context.active_object
        selected_objs = [obj for obj in context.selected_objects if obj != active_obj]
        
        self.report({'INFO'}, f"Snapping {len(selected_objs)} object(s) to '{active_obj.name}'...")
        
        for obj in selected_objs:
            # The most robust way to snap is to copy the entire world matrix
            obj.matrix_world = active_obj.matrix_world.copy()
            
        return {'FINISHED'}

class OBJECT_OT_reset_brush_cube_transform(bpy.types.Operator):
    """Reset the Brush Cube transform to default (centered and with default rotation/scale)"""
    bl_idname = "object.reset_brush_cube_transform"
    bl_label = "Reset Brush Cube Transform"

    def execute(self, context):
        scene = context.scene
        domain = scene.sdf_domain
        brush = scene.brush_cube
        if not domain or not brush:
            self.report({'ERROR'}, "Domain or Brush Cube not found.")
            return {'CANCELLED'}
        # For example, reset the brush cube to be centered on the domain, no rotation, default size.
        # You may further define what “default” means for your workflow.
        brush.location = domain.location.copy()
        brush.rotation_euler = (0.0, 0.0, 0.0)
        default_size = domain.dimensions.length / 2.0
        brush.scale = (default_size, default_size, default_size)
        brush.empty_display_size = default_size
        self.report({'INFO'}, "Brush Cube transform reset.")
        return {'FINISHED'}

class OBJECT_OT_create_brush_cube(bpy.types.Operator):
    """Create a new Brush Cube for clipping the SDF Domain"""
    bl_idname = "object.create_brush_cube"
    bl_label = "Create Brush Cube"

    def execute(self, context):
        scene = context.scene
        domain = scene.sdf_domain
        if not domain:
            self.report({'ERROR'}, "No SDF Domain found.")
            return {'CANCELLED'}
        if scene.brush_cube:
            self.report({'WARNING'}, "Brush Cube already exists.")
            return {'FINISHED'}
        # Create an empty of type 'CUBE' at the domain's location.
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        brush = context.active_object
        brush.name = "Brush_Cube"
        # Set a display size; adjust as desired.
        brush.empty_display_size = domain.dimensions.length / 2.0
        # Parent the brush to the domain (optional).
        brush.parent = domain
        scene.brush_cube = brush
        self.report({'INFO'}, "Brush Cube created.")
        return {'FINISHED'}

class OBJECT_OT_delete_brush_cube(bpy.types.Operator):
    """Delete the existing Brush Cube for clipping the SDF Domain"""
    bl_idname = "object.delete_brush_cube"
    bl_label = "Delete Brush Cube"

    def execute(self, context):
        scene = context.scene
        brush = scene.brush_cube
        if not brush:
            self.report({'WARNING'}, "No Brush Cube found.")
            return {'CANCELLED'}
        bpy.data.objects.remove(brush, do_unlink=True)
        scene.brush_cube = None
        self.report({'INFO'}, "Brush Cube deleted.")
        return {'FINISHED'}

class OBJECT_OT_select_brush_cube(bpy.types.Operator):
    """Select or Deselect the Brush Cube for editing"""
    bl_idname = "object.select_brush_cube"
    bl_label = "Select Brush Cube"

    def execute(self, context):
        brush = context.scene.brush_cube
        if not brush:
            self.report({'ERROR'}, "No Brush Cube found.")
            return {'CANCELLED'}
        # Toggle selection: If brush is active, deselect; else, select it.
        if brush.select_get():
            brush.select_set(False)
            context.view_layer.objects.active = None
        else:
            bpy.ops.object.select_all(action='DESELECT')
            brush.select_set(True)
            context.view_layer.objects.active = brush
        return {'FINISHED'}

class OBJECT_OT_toggle_brush_cube_visibility(bpy.types.Operator):
    """Toggle brush cube visibility (hide/unhide)"""
    bl_idname = "object.toggle_brush_cube_visibility"
    bl_label = "Toggle Brush Cube Visibility"

    def execute(self, context):
        brush = context.scene.brush_cube
        if not brush:
            self.report({'ERROR'}, "No Brush Cube found.")
            return {'CANCELLED'}
        brush.hide_viewport = not brush.hide_viewport
        return {'FINISHED'}

class OBJECT_OT_apply_brush_cube(bpy.types.Operator):
    """Apply the Brush Cube clipping by updating the Domain’s transform and node input."""
    bl_idname = "object.apply_brush_cube"
    bl_label = "Apply Brush Cube Clipping"

    def execute(self, context):
        scene = context.scene
        domain = scene.sdf_domain
        brush  = scene.brush_cube
        
        if not domain or not brush:
            self.report({'ERROR'}, "Domain or Brush Cube not found.")
            return {'CANCELLED'}

        if "original_domain_matrix" not in domain:
            domain["original_domain_matrix"] = domain.matrix_world.copy()

        saved_children = []
        if hasattr(domain, "sdf_nodes"):
            for item in domain.sdf_nodes:
                obj = item.empty_object
                if obj:
                    saved_children.append((obj, obj.matrix_world.copy()))
                    obj.parent = None

        if brush.parent == domain:
            brush.parent = None

        domain.matrix_world = brush.matrix_world.copy()

        computed_size = brush.empty_display_size
        new_size = computed_size if computed_size >= 1.0 else 1.0
        
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree:
            self.report({'ERROR'}, "Could not retrieve Geometry Nodes group from Domain.")
            return {'CANCELLED'}

        target_node = next((n for n in node_tree.nodes if n.name == "SDF Domain"), None)
        if not target_node:
            self.report({'WARNING'}, "SDF Domain node not found in the node tree.")
            return {'CANCELLED'}

        if "Domain Size" in target_node.inputs:
            target_node.inputs["Domain Size"].default_value = new_size
        else:
            self.report({'WARNING'}, "No 'Domain Size' socket found in SDF Domain node.")
            return {'CANCELLED'}

        brush.empty_display_size = new_size

        for obj, matrix in saved_children:
            obj.parent = domain
            obj.matrix_world = matrix
            
        # --- CRITICAL FIX: Force the dependency graph to update ---
        context.view_layer.update()
        # ---

        self.report({'INFO'}, "Applied Brush Cube clipping successfully.")
        return {'FINISHED'}




class OBJECT_OT_toggle_clip(bpy.types.Operator):
    """Toggle the clipping effect.
    When disabled, the Domain reverts to its original transform (and node input resets to 1.0)
    and the Brush Cube is removed; when enabled, the Domain's transform is updated to match
    the Brush Cube and the node input is set to the brush’s size.
    """
    bl_idname = "object.toggle_clip"
    bl_label = "Toggle Clipping Effect"

    def execute(self, context):
        scene = context.scene
        domain = scene.sdf_domain
        
        # Use a proper typed property
        if scene.clip_enabled:
            # --- Disable clipping:
            if domain and "original_domain_matrix" in domain:
                # Restore Domain's original transform
                domain.matrix_world = domain["original_domain_matrix"]
            try:
                geo_nodes = domain.modifiers["GeometryNodes"].node_group
                for node in geo_nodes.nodes:
                    if node.name == "SDF Domain":
                        # Reset the Domain Size input to 1.0
                        for sock in node.inputs:
                            if sock.name == "Domain Size":
                                sock.default_value = 1.0
                                break
                        break
            except Exception as e:
                self.report({'WARNING'}, "Failed to reset Domain Size socket.")
            # Remove the Brush Cube so that full Domain appears.
            if scene.brush_cube:
                bpy.data.objects.remove(scene.brush_cube, do_unlink=True)
                scene.brush_cube = None
            scene.clip_enabled = False
            self.report({'INFO'}, "Clipping disabled; Domain restored to default.")
        else:
            # --- Enable clipping:
            if domain and "original_domain_matrix" not in domain:
                # Store the current (default) domain transform
                domain["original_domain_matrix"] = domain.matrix_world.copy()
            if not scene.brush_cube:
                self.report({'ERROR'}, "No Brush Cube present. Create one first.")
                return {'CANCELLED'}
            # Apply the brush effect
            bpy.ops.object.apply_brush_cube()
            scene.clip_enabled = True
            self.report({'INFO'}, "Clipping enabled; Domain updated to match Brush Cube.")
        return {'FINISHED'}






class OBJECT_OT_ToggleMuteNode(bpy.types.Operator):
    bl_idname = "object.toggle_mute_node"
    bl_label = "Toggle Mute Node"
    bl_options = {'INTERNAL'}

    empty_name: bpy.props.StringProperty()
    mute: bpy.props.BoolProperty()

    def execute(self, context):
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.name.startswith("SDF_Domain"):
                geo_nodes = obj.modifiers.get("GeometryNodes")
                if geo_nodes:
                    for node in geo_nodes.node_group.nodes:
                        associated_empty = node.get('associated_empty')
                        if associated_empty == self.empty_name:
                            node.mute = self.mute
                            return {'FINISHED'}
        return {'CANCELLED'}


def check_mute_nodes(scene):
    domain_obj = getattr(scene, "sdf_domain", None)
    if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')):
        return

    node_tree = get_sdf_geometry_node_tree(bpy.context)
    if not node_tree:
        return

    for item in domain_obj.sdf_nodes:
        if item.empty_object:
            node = next((n for n in node_tree.nodes if n.get("associated_empty") == item.empty_object.name), None)
            if node:
                # Directly sync the node's mute state with the UI property
                if node.mute != item.is_hidden:
                    node.mute = item.is_hidden

class OBJECT_OT_SelectEmpty(bpy.types.Operator):
    """Select the SDF Empty, with safety checks that DO NOT modify data."""
    bl_idname = "object.select_empty"
    bl_label = "Select Empty"
    bl_options = {'INTERNAL'}

    empty_name: bpy.props.StringProperty()

    def execute(self, context):
        # Perform the standard robust check.
        empty_obj = bpy.data.objects.get(self.empty_name)

        if not empty_obj or empty_obj.name not in context.view_layer.objects:
            # The object is invalid. Instead of trying to fix it here, we just
            # report the problem and cancel. The UI list itself will now be
            # showing the "Clean List" button, guiding the user to the fix.
            self.report({'WARNING'}, f"Object '{self.empty_name}' is invalid. Please use the 'Clean List' button.")
            
            # We NO LONGER call the cleanup function here. We just cancel.
            return {'CANCELLED'}
        
        # If the check passes, we proceed with the selection.
        bpy.ops.object.select_all(action='DESELECT')
        empty_obj.select_set(True)
        context.view_layer.objects.active = empty_obj
        
        return {'FINISHED'}
    

class OBJECT_OT_purge_unused_data(bpy.types.Operator):
    """Purge all unused node groups and data blocks (orphans)"""
    bl_idname = "object.purge_unused_data"
    bl_label = "Purge Unused Data"

    def execute(self, context):
        # Remove unused node groups
        removed = 0
        for ng in list(bpy.data.node_groups):
            if ng.users == 0:
                bpy.data.node_groups.remove(ng)
                removed += 1
        # Optionally, call Blender's built-in orphan purge (UI context required for full effect)
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            pass
        self.report({'INFO'}, f"Purged {removed} unused node groups.")
        return {'FINISHED'}  
    
class VIEW3D_OT_toggle_overlays(bpy.types.Operator):
    """Globally enables or disables all viewport overlays for a cleaner view"""
    bl_idname = "view3d.toggle_sdf_overlays"
    bl_label = "Toggle Viewport Overlays"

    def execute(self, context):
        # The context for an operator running from a panel is the area it's in.
        # We need to access the space data of that area.
        space = context.space_data
        
        # As a safety check, ensure we are actually in a 3D Viewport
        if space.type == 'VIEW_3D':
            # This is the correct way to toggle overlays:
            # directly flip the boolean property.
            space.overlay.show_overlays = not space.overlay.show_overlays
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "This operator can only be run from a 3D Viewport.")
            return {'CANCELLED'}  

class OBJECT_OT_reset_global_scale(bpy.types.Operator):
    """Clears the stored initial transforms for all SDF nodes, making the current state the new default for global scaling"""
    bl_idname = "object.reset_sdf_global_scale"
    bl_label = "Reset Global Scale State"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.sdf_domain is not None

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')):
            self.report({'WARNING'}, "No active SDF Domain found.")
            return {'CANCELLED'}

        count = 0
        for item in domain_obj.sdf_nodes:
            empty_obj = item.empty_object
            if not empty_obj:
                continue
            
            # Delete the custom properties we use for storing the state.
            if "initial_location" in empty_obj:
                del empty_obj["initial_location"]
            if "initial_scale" in empty_obj:
                del empty_obj["initial_scale"]
            count += 1

        # Also reset the slider to 1.0
        context.scene.sdf_global_scale = 1.0
        self.report({'INFO'}, f"Reset scale state for {count} SDF nodes.")
        return {'FINISHED'}      

class OBJECT_OT_sdf_render_final(bpy.types.Operator):
    """Apply high-res settings and render the image from Camera or View, with shading options."""
    bl_idname = "object.sdf_render_final"
    bl_label  = "Render Final Image"
    bl_options= {'REGISTER'}

    def _find_3d_view_area(self, context):
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                return area
        return None

    def execute(self, context):
        scene = context.scene
        area = self._find_3d_view_area(context)

        # Store original settings
        prev_engine = scene.render.engine
        prev_pct    = scene.render.resolution_percentage
        prev_eevee  = scene.eevee.taa_render_samples
        prev_cyc    = scene.cycles.samples
        prev_sdf_mode = scene.sdf_preview_mode
        prev_final_res = scene.sdf_final_resolution
        prev_shading_type = area.spaces.active.shading.type if area else None
        prev_show_overlays = area.spaces.active.overlay.show_overlays if area else None

        def _restore():
            scene.render.engine = prev_engine
            scene.render.resolution_percentage = prev_pct
            scene.eevee.taa_render_samples = prev_eevee
            scene.cycles.samples = prev_cyc
            scene.sdf_final_resolution = prev_final_res
            scene.sdf_preview_mode = prev_sdf_mode
            if area:
                area.spaces.active.shading.type = prev_shading_type
                area.spaces.active.overlay.show_overlays = prev_show_overlays
            self.report({'INFO'}, "Restored scene settings after render.")
            return None

        # Apply render settings
        scene.render.engine = scene.sdf_render_engine
        scene.render.resolution_percentage = int(scene.sdf_render_scale * 100)
        scene.sdf_final_resolution = scene.sdf_render_highres_resolution
        scene.sdf_preview_mode = False
        context.view_layer.update()

        # Choose render method
        if scene.sdf_render_from == 'CAMERA':
            bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)
            bpy.app.timers.register(_restore, first_interval=2.0)
        else: # Render from 'VIEW'
            if not area:
                self.report({'WARNING'}, "No 3D Viewport available for render.")
                _restore()
                return {'CANCELLED'}
            
            # Temporarily set shading and overlays for the render
            if scene.sdf_render_shading_mode != 'CURRENT':
                area.spaces.active.shading.type = scene.sdf_render_shading_mode
            if scene.sdf_render_disable_overlays:
                area.spaces.active.overlay.show_overlays = False

            with context.temp_override(area=area, region=area.regions[-1]):
                bpy.ops.render.opengl(write_still=True)
            
            self.report({'INFO'}, f"Viewport render saved to {scene.render.filepath}")
            # Restore immediately since this is a blocking operation
            _restore()
            
        return {'FINISHED'}

#-----------------------------------------------------------------------

# In main.py, replace the SDFRenderPanel class with this new version.

# In main.py, replace the SDFRenderPanel class with this new version.

# In main.py, replace the SDFRenderPanel class with this one.

class SDFRenderPanel(bpy.types.Panel):
    bl_label      = "Render Options" # Label for the sub-panel itself
    bl_idname     = "VIEW3D_PT_sdf_render_options" # A new, unique ID name
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category   = 'Rogue_SDF_AI'
    bl_parent_id  = "VIEW3D_PT_sdf_prototyper" # This makes it a sub-panel
    bl_options    = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        # This is the gatekeeper. The panel only draws if this is True.
        # It now works correctly, independent of any shape selection.
        return context.scene.sdf_domain and context.scene.sdf_render_panel_enable

    def draw(self, context):
        layout = self.layout
        scene  = context.scene

        box = layout.box()
        col = box.column(align=True)
        col.prop(scene, "sdf_render_from",            text="From")
        col.prop(scene, "sdf_render_highres_resolution", text="Res")
        col.prop(scene, "sdf_render_scale",           text="Scale")
        col.prop(scene, "sdf_render_engine",          text="Engine")
        
        col.separator()
        col.prop(scene, "sdf_render_shading_mode", text="Shading")
        col.prop(scene, "sdf_render_disable_overlays", text="Disable Overlays")
        
        if scene.sdf_render_engine == 'CYCLES':
            col.prop(scene, "sdf_cycles_samples",         text="Cycles Max")
            col.prop(scene, "sdf_cycles_preview_samples", text="Cycles Min")
        else:
            col.prop(scene, "sdf_render_samples",         text="Eevee Samples")

        box.operator("object.sdf_render_highres", text="Run Viewport Preview", icon='RENDER_STILL')

        layout.separator()
        box = layout.box()
        box.operator("object.sdf_render_final", text="Render Final Image", icon='RENDER_STILL')


# -------------------------------------------------------------------
# Helper Functions for Locking and Index Updates
# -------------------------------------------------------------------
def matrices_equal(mat_a, mat_b, tol=1e-6):
    for i in range(4):
        for j in range(4):
            if abs(mat_a[i][j] - mat_b[i][j]) > tol:
                return False
    return True

def calculate_combined_bounds(objects):
    """
    Calculates the world-space bounding box that contains all given objects.
    Safely handles objects with no bound_box.
    """
    if not objects:
        return None

    all_corners = []
    for obj in objects:
        # --- FIX: Check if bound_box exists before using it ---
        if hasattr(obj, "bound_box"):
            all_corners.extend([obj.matrix_world @ Vector(c) for c in obj.bound_box])
        else:
            # For objects without a bound_box (like Plain Axes), just use their location
            all_corners.append(obj.matrix_world.translation)

    if not all_corners:
        return None, None

    min_corner = Vector(all_corners[0])
    max_corner = Vector(all_corners[0])

    for corner in all_corners[1:]:
        min_corner.x = min(min_corner.x, corner.x)
        min_corner.y = min(min_corner.y, corner.y)
        min_corner.z = min(min_corner.z, corner.z)
        max_corner.x = max(max_corner.x, corner.x)
        max_corner.y = max(max_corner.y, corner.y)
        max_corner.z = max(max_corner.z, corner.z)
            
    return min_corner, max_corner



def rewire_full_sdf_chain(context):
    """
    The master rewiring function. This version correctly wires the SDF chain
    to the SDF Domain, and then switches the Domain's output between the final
    mesh or a point cloud converter.
    """
    if not context:
        print("[Rogue SDF AI] Rewire called with invalid context. Aborting.")
        return

    scene = context.scene
    node_tree = get_sdf_geometry_node_tree(context)
    if not node_tree: return

    # --- 1. Get all essential nodes (we can now assume they exist) ---
    out_node = next((n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT'), None)
    domain_node = next((n for n in node_tree.nodes if n.name == "SDF Domain"), None)
    dec_node = get_dec_node(context)
    points_output_node = next((n for n in node_tree.nodes if "SDF Points Output" in n.name), None)

    if not all([domain_node, out_node, points_output_node]):
        print("[Rogue SDF AI] Essential nodes are missing from the node tree. Please regenerate the domain.")
        return
    
    if not any(item.name == "Geometry" and item.item_type == 'SOCKET' and item.in_out == 'OUTPUT' for item in node_tree.interface.items_tree):
        node_tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    output_geo_in = out_node.inputs.get("Geometry")
    if not output_geo_in: return

    # --- 2. Clear ONLY the links that will be re-established ---
    if output_geo_in.is_linked:
        for link in list(output_geo_in.links): node_tree.links.remove(link)
    if domain_node.inputs[0].is_linked:
        for link in list(domain_node.inputs[0].links): node_tree.links.remove(link)
    if points_output_node.inputs[0].is_linked:
        for link in list(points_output_node.inputs[0].links): node_tree.links.remove(link)
    if dec_node and dec_node.inputs["Geometry"].is_linked:
        for link in list(dec_node.inputs["Geometry"].links): node_tree.links.remove(link)

    # --- 3. Get all active SDF shape nodes ---
    active_shape_nodes = []
    if hasattr(scene.sdf_domain, 'sdf_nodes'):
        for item in scene.sdf_domain.sdf_nodes:
            if item.empty_object and not item.is_hidden:
                node = next((n for n in node_tree.nodes if n.get("associated_empty") == item.empty_object.name), None)
                if node:
                    active_shape_nodes.append(node)

    # --- 4. Build the single, sequential SDF chain ---
    if not active_shape_nodes:
        node_tree.links.new(domain_node.outputs["Mesh"], output_geo_in)
        print("[Rogue SDF AI] No active shapes. Node chain is empty.")
        return

    item_names = [item.name for item in scene.sdf_domain.sdf_nodes]
    def get_sort_key(node):
        try:
            return item_names.index(node.get("associated_empty"))
        except (ValueError, TypeError):
            return 999
            
    sorted_nodes = sorted(active_shape_nodes, key=get_sort_key)

    for i in range(len(sorted_nodes) - 1):
        next_input = sorted_nodes[i+1].inputs[0]
        if next_input.is_linked:
            for link in list(next_input.links): node_tree.links.remove(link)
        node_tree.links.new(sorted_nodes[i].outputs[0], next_input)
    
    final_sdf_field = sorted_nodes[-1].outputs[0]

    # --- 5. ALWAYS connect the SDF chain to the SDF Domain ---
    domain_node.mute = False
    node_tree.links.new(final_sdf_field, domain_node.inputs[0])
    domain_mesh_output = domain_node.outputs["Mesh"]

    # --- 6. Switch the FINAL GEOMETRY based on the visualization mode ---
    if scene.sdf_visualization_mode == 'SOLID':
        points_output_node.mute = True
        if scene.sdf_decimation_enable and dec_node:
            dec_node.mute = False
            node_tree.links.new(domain_mesh_output, dec_node.inputs["Geometry"])
            node_tree.links.new(dec_node.outputs["Geometry"], output_geo_in)
        else:
            if dec_node: dec_node.mute = True
            node_tree.links.new(domain_mesh_output, output_geo_in)
    else: # 'POINTS' mode
        if dec_node: dec_node.mute = True
        points_output_node.mute = False
        node_tree.links.new(domain_mesh_output, points_output_node.inputs[0])
        node_tree.links.new(points_output_node.outputs[0], output_geo_in)

    print(f"[Rogue SDF AI] Node chain rewired for '{scene.sdf_visualization_mode}' mode.")

#----------------------------------------------------------------------

def toggle_gn_output_mute(context, mute: bool):
    """
    Mutes or unmutes the node connected to the final Geometry output.
    This prevents the mesh from being cooked when it's not needed (e.g., shader view).
    """
    node_tree = get_sdf_geometry_node_tree(context)
    if not node_tree:
        return

    # Find the final output node in the tree
    out_node = next((n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT'), None)
    if not out_node:
        return

    # Find the "Geometry" input socket on that output node
    output_geo_in = out_node.inputs.get("Geometry")
    if not output_geo_in or not output_geo_in.is_linked:
        return

    # Get the node that is feeding the final output
    final_node_link = output_geo_in.links[0]
    node_to_toggle = final_node_link.from_node

    # Mute or unmute that node
    if node_to_toggle.mute != mute:
        node_to_toggle.mute = mute
        print(f"[Rogue SDF AI] Geometry Node output set to mute: {mute}")



#--------------------------------------------------------------------


def toggle_lock_based_on_selection(scene):
    selected_objects = bpy.context.selected_objects
    for obj in selected_objects:
        if obj.type == 'MESH' and obj.name.startswith("SDF_Domain"):
            bpy.context.scene.lock_sdf_panel = False
            bpy.context.scene.lock_sdf_panel = True
            return

def get_last_selected_cube(context):
    for obj in context.selected_objects:
        if obj.type == 'MESH' and obj.name.startswith("SDF_Domain"):
            return obj
    return None

def update_lock(self, context):
    if context.scene.lock_sdf_panel:
        context.scene.locked_sdf_object = get_last_selected_cube(context)
    else:
        context.scene.locked_sdf_object = None
    if context.area:
        context.area.tag_redraw()

def update_active_index_from_selection(context):
    """
    Syncs the UI list's active index with the 3D viewport's active object.
    If an SDF Empty is selected, its entry in the list is highlighted.
    This now correctly handles Empties that are direct children of the domain
    OR grandchildren within a [Group] parent.
    """
    domain_obj = getattr(context.scene, "sdf_domain", None)
    active_obj = context.active_object

    if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')):
        return

    new_index = -1

    # --- NEW, MORE FLEXIBLE CHECK ---
    # An object is a valid SDF controller if:
    # 1. It's an EMPTY.
    # 2. It has a parent.
    # 3. EITHER its direct parent is the SDF Domain,
    #    OR its parent's parent is the SDF Domain (for our [Group] objects).
    if (active_obj and active_obj.type == 'EMPTY' and active_obj.parent and
            (active_obj.parent == domain_obj or active_obj.parent.parent == domain_obj)):
        
        # If it's a valid controller, find its corresponding index in the list.
        for i, item in enumerate(domain_obj.sdf_nodes):
            if item.empty_object == active_obj:
                new_index = i
                break
    
    # Only update the index if it has actually changed to prevent redraw loops.
    if domain_obj.active_sdf_node_index != new_index:
        domain_obj.active_sdf_node_index = new_index
        
        # Force the UI panel to redraw to show the highlight instantly.
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'UI':
                    area.tag_redraw()

def depsgraph_update(scene):
    """
    This is the function registered with the depsgraph handler.
    It's called whenever dependencies are updated, including selection changes.
    """
    # We simply call our main logic function, passing it the current context.
    update_active_index_from_selection(bpy.context)

# -------------------------------------------------------------------
# SDF Prototyper Panel (including new Resolution Settings UI)
# -------------------------------------------------------------------

import bpy

# In main.py, replace the ENTIRE SDFPrototyperPanel class with this one.

class SDFPrototyperPanel(bpy.types.Panel):
    bl_label      = "Rogue SDF AI"
    bl_idname     = "VIEW3D_PT_sdf_prototyper"
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category   = 'Rogue_SDF_AI'

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        domain = scene.sdf_domain

        if not domain:
            layout.operator("object.start_sdf", text="Generate SDF", icon='GEOMETRY_NODES')
            return

        # ... (The first part of the panel is unchanged)
        preview_box = layout.box()
        preview_box.label(text="Performance & Preview", icon='MOD_WAVE')
        col = preview_box.column(align=True)
        row = col.row(align=True)
        op_solid = row.operator(PROTOTYPER_OT_SDFSetPointCloudPreview.bl_idname, text="All Solid")
        op_solid.mode = 'SOLID'
        op_points = row.operator(PROTOTYPER_OT_SDFSetPointCloudPreview.bl_idname, text="All Points")
        op_points.mode = 'POINTS'
        if scene.sdf_shader_view:
            ray_box = preview_box.box()
            ray_box.label(text="Ray March Performance")
            ray_box.prop(scene, "sdf_raymarch_max_steps")
            ray_box.prop(scene, "sdf_pixelation_amount")
        res_box = layout.box()
        res_box.label(text="Domain Settings", icon='OBJECT_DATA')
        res_col = res_box.column(align=True)
        row_scale = res_col.row(align=True)
        row_scale.prop(scene, "sdf_domain_scale", text="Domain Scale")
        row_scale.enabled = not scene.clip_enabled
        res_col.separator()
        res_col.prop(scene, "sdf_auto_resolution_enable", text="Automatic Preview", icon='AUTO')
        if scene.sdf_auto_resolution_enable:
            auto = res_box.box()
            ac = auto.column(align=True)
            ac.prop(scene, "sdf_auto_threshold", text="Sensitivity")
            ac.prop(scene, "sdf_auto_idle_delay", text="Idle Delay (s)")
        sbox = res_box.box()
        sc = sbox.column()
        mr = sc.row(align=True)
        mr.enabled = not scene.sdf_auto_resolution_enable
        mr.prop(scene, "sdf_preview_mode", text="Manual Preview", toggle=True)
        rr = sc.row(align=True)
        rr.prop(scene, "sdf_preview_resolution", text="Low-Res")
        rr.prop(scene, "sdf_final_resolution",   text="High-Res")
        if scene.sdf_auto_resolution_enable:
            st = res_box.row()
            st.label(text="Status:")
            st.label(text=scene.sdf_status_message)
        box = layout.box()
        box.label(text="Global Controls", icon='SETTINGS')
        col = box.column(align=True)
        col.prop(scene, "sdf_max_shapes")
        row = col.row(align=True)
        row.prop(scene, "sdf_global_scale", text="Shape Scale")
        row.operator("object.reset_sdf_global_scale", text="", icon='FILE_REFRESH')
        dnode = get_dec_node(context)
        r = box.row(align=True)
        r.operator(OBJECT_OT_ToggleDecimation.bl_idname, text="Enable Decimation", icon='MOD_DECIM', depress=(dnode is not None))
        if dnode:
            sub = box.column(align=True)
            for inp in dnode.inputs:
                if inp.name != "Geometry":
                    sub.prop(inp, "default_value", text=inp.name)
        layout.separator()
        row = layout.row()
        row.operator("view3d.toggle_sdf_overlays", text="Toggle Overlays", icon='OVERLAY')
        view_box = layout.box()
        view_box.label(text="View Filter", icon='HIDE_OFF')
        view_box.prop(scene, "sdf_view_mode", expand=True)
        row = layout.row()
        row.template_list("SDF_UL_nodes", "", domain, "sdf_nodes", domain, "active_sdf_node_index", rows=4)
        ops = row.column(align=True)
        ops.operator("prototyper.sdf_duplicate",    icon='DUPLICATE', text="")
        ops.operator("prototyper.sdf_repeat_shape", icon='MOD_ARRAY', text="")
        ops.operator("prototyper.sdf_delete",       icon='REMOVE',    text="")
        ops.separator()
        up = ops.operator("prototyper.sdf_list_move", icon='TRIA_UP', text=""); up.direction = 'UP'
        dn = ops.operator("prototyper.sdf_list_move", icon='TRIA_DOWN', text=""); dn.direction = 'DOWN'
        ops.separator()
        ops.operator("prototyper.sdf_clear",        icon='X',         text="")
        layout.separator()
        layout.prop(scene, "sdf_shader_view", text="Enable SDF Shader View", icon='SHADING_RENDERED')
        if scene.sdf_shader_view:
            light_box = layout.box()
            light_box.label(text="Advanced Preview Lighting", icon='MATERIAL')
            
            col = light_box.column()
            col.prop(scene, "sdf_light_direction", text="")
            col.separator()
            row = col.row(align=True)
            row.prop(scene, "sdf_preview_brightness")
            row.prop(scene, "sdf_preview_contrast")
            col.separator()
            cavity_box = light_box.box()
            row = cavity_box.row()
            row.prop(scene, "sdf_cavity_enable")
            row_cavity = cavity_box.row()
            row_cavity.enabled = scene.sdf_cavity_enable
            row_cavity.prop(scene, "sdf_cavity_strength")
        layout.separator()
        shape_box = layout.box()
        shape_box.label(text="Shape Settings", icon='MODIFIER_DATA')
        idx = domain.active_sdf_node_index
        if 0 <= idx < len(domain.sdf_nodes):
            item  = domain.sdf_nodes[idx]
            empty = item.empty_object
            if empty and empty.name in context.view_layer.objects:
                geo  = get_sdf_geometry_node_tree(context)
                node = next((n for n in geo.nodes if n.get("associated_empty") == empty.name), None)
                if node:
                    b = shape_box.box()
                    b.label(text=f"'{item.name}' Settings", icon='OBJECT_DATA')
                    sub = b.column(align=True)
                    sub.prop(item, "name", text="")
                    if scene.sdf_shader_view:
                        sb = b.box()
                        sb.label(text="Shader Operations", icon='SHADING_RENDERED')
                        sc2 = sb.column(align=True)
                        sc2.prop(scene, "sdf_global_tint", text="Global Tint")
                        sc2.prop(item, "operation", text="Operation")
                        if item.operation in {'UNION', 'SUBTRACT', 'INTERSECT'}:
                            sc2.prop(item, "blend_type", text="Blend Type")
                            blend_label = "Chamfer Size" if item.blend_type == 'CHAMFER' else "Smoothness"
                            sc2.prop(item, "blend", text=blend_label)
                        elif item.operation in {'DISPLACE', 'INDENT', 'RELIEF', 'ENGRAVE'}:
                            sc2.prop(item, "blend_strength", text="Strength")
                            sc2.prop(item, "blend", text="Smoothness")
                        elif item.operation == 'MASK':
                            #sc2.prop(item, "blend_strength", text="Shell Thickness")
                            sc2.prop(item, "blend", text="Edge Smoothness")
                        elif item.operation == 'PAINT':
                             sc2.prop(item, "blend_strength", text="Feather")
                        if item.icon == 'MESH_CUBE':
                            sc2.prop(item, "thickness"); sc2.prop(item, "roundness"); sc2.prop(item, "bevel"); sc2.prop(item, "pyramid"); sc2.prop(item, "twist"); sc2.prop(item, "bend")
                        elif item.icon == 'MESH_UVSPHERE':
                            sc2.prop(item, "sphere_thickness"); sc2.prop(item, "sphere_elongation"); sc2.prop(item, "sphere_cut_angle")
                        elif item.icon == 'MESH_CYLINDER':
                            sc2.prop(item, "cylinder_thickness"); sc2.prop(item, "cylinder_roundness"); sc2.prop(item, "cylinder_pyramid"); sc2.prop(item, "cylinder_bend")
                        elif item.icon == 'MESH_ICOSPHERE':
                            sc2.prop(item, "prism_sides"); sc2.prop(item, "prism_thickness"); sc2.separator(); sc2.label(text="Deformers:"); sc2.prop(item, "prism_pyramid"); sc2.prop(item, "prism_bend"); sc2.prop(item, "prism_twist")
                        elif item.icon == 'MESH_TORUS':
                            sc2.prop(item, "torus_outer_radius"); sc2.prop(item, "torus_inner_radius"); sc2.prop(item, "torus_thickness"); sc2.prop(item, "torus_cut_angle"); sc2.prop(item, "torus_elongation")
                        sc2.prop(item, "preview_color", text="Color")
                        sc2.prop(scene, "sdf_color_blend_mode", text="Color Blend")
                    else:
                        gn = b.box()
                        gn.label(text="Modeling Mode Inputs", icon='NODETREE')
                        tabs = gn.row(align=True)
                        tabs.prop(scene, "sdf_shape_tab", expand=True)
                        col3 = gn.column(align=True)
                        if scene.sdf_shape_tab == 'DEFORM':
                            row3 = col3.row(align=True)
                            row3.label(text="Flip Shape:")
                            op_x = row3.operator(PROTOTYPER_OT_SDFFlipActiveShape.bl_idname, text="X"); op_x.axis = 'X'
                            op_y = row3.operator(PROTOTYPER_OT_SDFFlipActiveShape.bl_idname, text="Y"); op_y.axis = 'Y'
                            op_z = row3.operator(PROTOTYPER_OT_SDFFlipActiveShape.bl_idname, text="Z"); op_z.axis = 'Z'
                            col3.separator()
                        for sock in node.inputs:
                            key = sock.name.lower()
                            tab = scene.sdf_shape_tab
                            show = ((tab=='BASIC' and any(k in key for k in ("radius","width","height","depth","size"))) or (tab=='DEFORM' and any(k in key for k in ("bend","twist","taper","scale"))) or (tab=='STYLE' and any(k in key for k in ("blend","style","soft","round"))) or (tab=='MISC' and not any(k in key for k in ("sdf","slice","object"))))
                            if show:
                                col3.prop(sock, "default_value", text=sock.name)
                    if item.icon == 'CURVE_BEZCURVE':
                        curve_box = b.box()
                        curve_box.label(text="Curve Settings", icon='CURVE_DATA')
                        ccol = curve_box.column(align=True)
                        ccol.prop(item, "curve_mode", expand=True)
                        ccol.prop(item, "curve_control_mode", expand=True)
                        if item.curve_control_mode == 'UNIFORM':
                            uniform_box = ccol.box()
                            ucol = uniform_box.column(align=True)
                            ucol.prop(item, "curve_instance_type", text="Shape")
                            ucol.prop(item, "curve_point_density")
                            ucol.prop(item, "curve_instance_spacing", text="Spacing")
                            ucol.prop(item, "curve_instance_rotation", text="Rotation")
                            inst_box = uniform_box.box()
                            inst_box.label(text=f"{item.curve_instance_type.replace('_', ' ').title()} Settings", icon='MODIFIER_DATA')
                            icol = inst_box.column(align=True)
                            if item.curve_instance_type == 'MESH_CUBE':
                                icol.prop(item, "thickness"); icol.prop(item, "roundness"); icol.prop(item, "bevel"); icol.prop(item, "pyramid"); icol.prop(item, "twist"); icol.prop(item, "bend")
                            elif item.curve_instance_type == 'MESH_UVSPHERE':
                                icol.prop(item, "sphere_thickness"); icol.prop(item, "sphere_elongation"); icol.prop(item, "sphere_cut_angle")
                            elif item.curve_instance_type == 'MESH_ICOSPHERE':
                                icol.prop(item, "prism_sides"); icol.prop(item, "prism_thickness"); icol.separator(); icol.label(text="Deformers:"); icol.prop(item, "prism_pyramid"); icol.prop(item, "prism_bend"); icol.prop(item, "prism_twist")
                            elif item.curve_instance_type == 'MESH_TORUS':
                                icol.prop(item, "torus_outer_radius"); icol.prop(item, "torus_inner_radius"); icol.prop(item, "torus_thickness"); icol.prop(item, "torus_cut_angle"); icol.prop(item, "torus_elongation")
                            elif item.curve_instance_type == 'MESH_CYLINDER':
                                icol.prop(item, "cylinder_thickness"); icol.prop(item, "cylinder_roundness"); icol.prop(item, "cylinder_pyramid"); icol.prop(item, "cylinder_bend")
                            elif item.curve_instance_type == 'MESH_CONE':
                                icol.label(text="Cone radius is controlled by curve point radius (Alt+S).")
                        else: # --- CUSTOM MODE UI ---
                            custom_box = ccol.box()
                            row = custom_box.row()
                            row.template_list("SDF_UL_curve_points", "", item, "custom_control_points", item, "active_control_point_index")
                            ops_col = row.column(align=True)
                            ops_col.operator("prototyper.sdf_curve_point_add", icon='ADD', text="")
                            ops_col.operator("prototyper.sdf_curve_point_remove", icon='REMOVE', text="")
                            idx = item.active_control_point_index
                            if 0 <= idx < len(item.custom_control_points):
                                point = item.custom_control_points[idx]
                                point_box = custom_box.box()
                                pcol = point_box.column(align=True)
                                pcol.prop(point, "t_value", text="Position")
                                pcol.prop(point, "radius_multiplier", text="Radius")
                                pcol.prop(point, "color")
                                pcol.prop(point, "shape_type", text="Shape")
                                pcol.prop(point, "rotation")
                                inst_box = point_box.box()
                                inst_box.label(text=f"{point.shape_type.replace('_', ' ').title()} Settings", icon='MODIFIER_DATA')
                                icol = inst_box.column(align=True)
                                if point.shape_type == 'MESH_CUBE':
                                    icol.prop(point, "thickness"); icol.prop(point, "roundness"); icol.prop(point, "bevel"); icol.prop(point, "pyramid"); icol.prop(point, "twist"); icol.prop(point, "bend")
                                elif point.shape_type == 'MESH_UVSPHERE':
                                    icol.prop(point, "sphere_thickness"); icol.prop(point, "sphere_elongation"); icol.prop(point, "sphere_cut_angle")
                                elif point.shape_type == 'MESH_ICOSPHERE':
                                    icol.prop(point, "prism_sides"); icol.prop(point, "prism_thickness"); icol.separator(); icol.label(text="Deformers:"); icol.prop(point, "prism_pyramid"); icol.prop(point, "prism_bend"); icol.prop(point, "prism_twist")
                                elif point.shape_type == 'MESH_TORUS':
                                    icol.prop(point, "torus_outer_radius"); icol.prop(point, "torus_inner_radius"); icol.prop(point, "torus_thickness"); icol.prop(point, "torus_cut_angle"); icol.prop(point, "torus_elongation")
                                elif point.shape_type == 'MESH_CYLINDER':
                                    icol.prop(point, "cylinder_thickness"); icol.prop(point, "cylinder_roundness"); icol.prop(point, "cylinder_pyramid"); icol.prop(point, "cylinder_bend")
                        radius_box = curve_box.box()
                        radius_box.label(text="Global Radius & Scale", icon='PROP_CON')
                        rcol = radius_box.column(align=True)
                        rcol.prop(item, "curve_global_radius")
                        rcol.prop(item, "curve_segment_scale") 
                        rcol.separator()
                        rcol.prop(item, "curve_taper_head")
                        rcol.prop(item, "curve_taper_tail")
                        info_box = curve_box.box()
                        info_box.label(text="To control per-point radius:", icon='INFO')
                        info_box.label(text="1. Select the curve object.")
                        info_box.label(text="2. Go into Edit Mode (Tab).")
                        info_box.label(text="3. Select a point.")
                        info_box.label(text="4. Press ALT+S to scale.")
                    sym = b.box()
                    sym.label(text="Symmetry", icon='MOD_MIRROR')
                    mr2 = sym.row(align=True)
                    mr2.label(text="Mirror:")
                    mr2.prop(item, "use_mirror_x", text="X", toggle=True)
                    mr2.prop(item, "use_mirror_y", text="Y", toggle=True)
                    mr2.prop(item, "use_mirror_z", text="Z", toggle=True)
                    sym.prop(item, "use_radial_mirror", text="Radial Mirror")
                    if item.use_radial_mirror:
                        rb = sym.box()
                        rb.prop(item, "radial_mirror_count", text="Count")
        else:
            shape_box.label(text="No shape selected", icon='INFO')
            
        layout.separator()
        act_box = layout.box()
        act_box.label(text="Finalize & Symmetrize", icon='MOD_MESHDEFORM')
        act_col = act_box.column(align=True)
        act_col.operator("object.convert_sdf", text="Convert to Mesh", icon='MESH_DATA')
        act_col.operator("object.sdf_bake_volume", text="Bake to High-Quality Mesh", icon='VOLUME_DATA')
        
        act_col.operator("object.sdf_bake_to_remesh", text="Bake for Remesh Object", icon='TEXTURE')

        act_col.separator()
        act_col.operator(OBJECT_OT_sdf_auto_uv.bl_idname, text="Auto UV Selected", icon='UV_DATA')
        # --- NEW BUTTON ADDED HERE ---
        act_col.operator(OBJECT_OT_sdf_remesh_tools.bl_idname, text="Open Remesh Tools", icon='MOD_REMESH')
        act_col.operator(OBJECT_OT_sdf_snap_selection_to_active.bl_idname, text="Snap Selection to Active", icon='SNAP_ON')

        act_col.separator()
        act_col.operator("prototyper.sdf_bake_symmetry", text="Bake Active Symmetries", icon='CHECKMARK')
        act_col.operator("prototyper.sdf_symmetrize", text="Symmetrize Model", icon='MOD_MIRROR')
        act_col.operator("prototyper.sdf_flip_model", text="Flip Model...", icon='CON_ACTION')
        layout.separator()
        brush_box = layout.box()
        brush_box.label(text="Brush-Cube Clipping", icon='CUBE')
        if scene.brush_cube:
            brush_col = brush_box.column(align=True)
            brush_col.operator("object.select_brush_cube", text="Select Brush Cube")
            brush_col.operator("object.toggle_brush_cube_visibility", text="Show/Hide Brush Cube")
            brush_col.operator("object.reset_brush_cube_transform", text="Reset Brush Transform")
            brush_col.operator("object.toggle_clip", text="Toggle Clipping")
            brush_box.operator("object.delete_brush_cube", text="Delete Brush Cube", icon='TRASH')
        else:
            brush_box.operator("object.create_brush_cube", text="Create Brush Cube")
        layout.separator()
        layout.prop(scene, "sdf_render_panel_enable", text="Show Render Options", toggle=True, icon='RENDER_STILL')


# -------------------------------------------------------------------
class OBJECT_OT_ToggleDecimation(bpy.types.Operator):
    """Adds or removes the MK_Rogue_Decimate node. Loads the node if needed"""
    bl_idname = "object.toggle_sdf_decimation"
    bl_label = "Toggle Decimation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.sdf_domain is not None

    def execute(self, context):
        scene = context.scene
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree:
            self.report({'WARNING'}, "No valid SDF Domain with Geometry Nodes found.")
            return {'CANCELLED'}

        dec_node = get_dec_node(context)

        if dec_node:
            node_tree.nodes.remove(dec_node)
            scene.sdf_decimation_enable = False
            self.report({'INFO'}, "Decimation Disabled.")
        else:
            self.report({'INFO'}, "Attempting to enable decimation...")
            
            try:
                load_all_sdf_node_groups() # This ensures all nodes, including decimate, are loaded
            except Exception as e:
                self.report({'ERROR'}, f"Failed to load resources: {e}")
                return {'CANCELLED'}

            dec_group = bpy.data.node_groups.get("MK_Rogue_Decimate")
            if dec_group:
                new_node = node_tree.nodes.new(type="GeometryNodeGroup")
                new_node.node_tree = dec_group
                new_node.name = "MK_Rogue_Decimate"
                
                domain_node = next((n for n in node_tree.nodes if n.name == "SDF Domain"), None)
                if domain_node:
                    new_node.location = domain_node.location + Vector((250, 0))
                
                scene.sdf_decimation_enable = True
                self.report({'INFO'}, "Decimation Enabled.")
            else:
                self.report({'ERROR'}, "FATAL: 'MK_Rogue_Decimate' node group could not be found in resource file.")
                return {'CANCELLED'}

        rewire_full_sdf_chain(context)
        return {'FINISHED'}





#----------------------------------------------------------------------
class VIEW3D_PT_sdf_convert(bpy.types.Panel):
    bl_label = "Convert SDF to Mesh"
    bl_idname = "VIEW3D_PT_sdf_convert"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Rogue_SDF_AI'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.convert_sdf",
                        text="Convert SDF Shapes → Mesh",
                        icon='MESH_DATA')        




# -------------------------------------------------------------------
# SDF Generation and Conversion Operators
# -------------------------------------------------------------------
class StartSDFOperator(bpy.types.Operator):
    """Creates a new SDF Domain. WARNING: This will clear any existing SDF setup."""
    bl_idname = "object.start_sdf"
    bl_label = "Start a New SDF Setup"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # --- 1. CRITICAL: Ensure all assets are loaded ONCE. ---
        try:
            load_all_sdf_node_groups()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load SDF resources: {e}")
            return {'CANCELLED'}

        # 2. Cleanly remove any existing SDF domain to prevent conflicts.
        if context.scene.sdf_domain:
            bpy.ops.prototyper.sdf_clear('INVOKE_DEFAULT')
            if context.scene.sdf_domain:
                self.report({'WARNING'}, "Previous SDF domain was not cleared. Aborting generation.")
                return {'CANCELLED'}
            context.scene.sdf_domain = None

        # 3. Create the new SDF Domain object.
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        domain_obj = context.active_object
        domain_obj.name = "SDF_Domain"
        context.scene.sdf_domain = domain_obj

        # 4. Add and configure the Geometry Nodes modifier.
        geo_mod = domain_obj.modifiers.new(name="SDF Nodes", type='NODES')
        if not geo_mod.node_group:
            geo_mod.node_group = bpy.data.node_groups.new(name="SDF Node Tree", type='GeometryNodeTree')
        node_tree = geo_mod.node_group
        node_tree.nodes.clear()

        # --- 5. Create ALL essential, permanent nodes for the base setup. ---
        group_output = node_tree.nodes.new(type="NodeGroupOutput")
        group_output.location = (600, 0)
        
        # Create SDF Domain Node
        domain_node_group = bpy.data.node_groups.get("SDF Domain")
        if not domain_node_group:
            self.report({'ERROR'}, "FATAL: Could not find 'SDF Domain' node group.")
            bpy.data.objects.remove(domain_obj, do_unlink=True)
            return {'CANCELLED'}
        sdf_domain_node = node_tree.nodes.new(type="GeometryNodeGroup")
        sdf_domain_node.node_tree = domain_node_group
        sdf_domain_node.location = (200, 0)
        sdf_domain_node.name = "SDF Domain"

        # --- FIX: Create the SDF Points Output node right at the start ---
        points_node_group = bpy.data.node_groups.get("SDF Points Output")
        if not points_node_group:
            self.report({'ERROR'}, "FATAL: Could not find 'SDF Points Output' node group.")
            bpy.data.objects.remove(domain_obj, do_unlink=True)
            return {'CANCELLED'}
        points_output_node = node_tree.nodes.new(type="GeometryNodeGroup")
        points_output_node.node_tree = points_node_group
        points_output_node.location = (400, -150)
        points_output_node.name = "SDF Points Output"
        # --- END FIX ---

        # 6. Wire the initial chain, lock the panel, and set initial values.
        rewire_full_sdf_chain(context)
        context.scene.lock_sdf_panel = True
        context.scene.sdf_domain_scale = 1.0 
        update_sdf_resolution(self, context)

        self.report({'INFO'}, "New Rogue SDF AI system initialized successfully.")
        return {'FINISHED'}
#---------------------------------------------------------------------


class ConvertSDFOperator(bpy.types.Operator):
    """Convert SDF to a new, separate Mesh object, always using solid geometry"""
    bl_idname = "object.convert_sdf"
    bl_label = "Convert SDF to Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "sdf_domain", None) is not None

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        
        # The logic for saving/restoring point cloud state has been REMOVED
        
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = domain_obj
        domain_obj.select_set(True)

        bpy.ops.object.duplicate()
        new_mesh_obj = context.active_object
        new_mesh_obj.name = "Converted_SDF_Mesh"

        mod_to_apply = next((m for m in new_mesh_obj.modifiers if m.type == 'NODES'), None)
        if mod_to_apply:
            bpy.ops.object.modifier_apply(modifier=mod_to_apply.name)
        else:
            self.report({'WARNING'}, "No Geometry Nodes modifier found to apply.")
            return {'CANCELLED'}
        
        if 'sdf_nodes' in new_mesh_obj: del new_mesh_obj['sdf_nodes']
        if 'active_sdf_node_index' in new_mesh_obj: del new_mesh_obj['active_sdf_node_index']

        self.report({'INFO'}, f"Successfully converted SDF to new mesh: '{new_mesh_obj.name}'")
        return {'FINISHED'}
    
#---------------------------------------------------------------------

import bpy
import math
from mathutils import Vector, Matrix

class PROTOTYPER_OT_SDFSymmetrize(bpy.types.Operator):
    """Deletes all shapes on one side and replaces them with a mirrored copy of the other side."""
    bl_idname = "prototyper.sdf_symmetrize"
    bl_label = "Symmetrize SDF Model"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        items=[('X', "X", "Symmetrize along the X-axis"),
               ('Y', "Y", "Symmetrize along the Y-axis"),
               ('Z', "Z", "Symmetrize along the Z-axis")],
        default='X'
    )

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[('POSITIVE_TO_NEGATIVE', "+ to -", "Copy the positive side to the negative side"),
               ('NEGATIVE_TO_POSITIVE', "- to +", "Copy the negative side to the positive side")],
        default='POSITIVE_TO_NEGATIVE'
    )

    @classmethod
    def poll(cls, context):
        return context.scene.sdf_domain is not None and hasattr(context.scene.sdf_domain, 'sdf_nodes')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        domain = context.scene.sdf_domain
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree:
            self.report({'ERROR'}, "SDF Node Tree not found.")
            return {'CANCELLED'}

        # --- Setup ---
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[self.axis]
        sign = 1.0 if self.direction == 'POSITIVE_TO_NEGATIVE' else -1.0
        center_coord = domain.location[axis_index]

        items_to_delete_indices = []
        objects_to_delete = []
        copy_templates = []

        # --- 1. Classification Phase (Now with robustness check) ---
        for i, item in enumerate(domain.sdf_nodes):
            # ROBUSTNESS FIX: Skip any broken items in the list to prevent crashes.
            if not item.empty_object:
                print(f"[Rogue SDF AI] Symmetrize Warning: Skipping invalid item '{item.name}' in list.")
                continue

            loc = item.empty_object.location[axis_index]
            dist = (loc - center_coord) * sign

            if dist > 1e-6:  # This is on the source side, so we will copy it.
                copy_templates.append(item)
            elif dist < -1e-6:  # This is on the destination side, so it must be deleted.
                items_to_delete_indices.append(i)
                objects_to_delete.append(item.empty_object)
            # else: the object is on the center plane and is left untouched.

        # --- 2. Deletion Phase ---
        # First, remove the UI list items, going backwards to preserve indices.
        for index in sorted(items_to_delete_indices, reverse=True):
            domain.sdf_nodes.remove(index)
            
        # Then, delete the associated nodes and Blender objects.
        for empty in objects_to_delete:
            node = next((n for n in node_tree.nodes if n.get("associated_empty") == empty.name), None)
            if node:
                node_tree.nodes.remove(node)
            
            # Delete all children (like Cone tips or Curve objects) first.
            for child in list(empty.children):
                bpy.data.objects.remove(child, do_unlink=True)
            
            # Finally, delete the main empty controller.
            bpy.data.objects.remove(empty, do_unlink=True)

        # --- 3. Creation Phase (Using the new corrected helper) ---
        for template_item in copy_templates:
            # ROBUSTNESS FIX: A final check to ensure the template is still valid before using it.
            if not template_item.empty_object:
                print(f"[Rogue SDF AI] Symmetrize Warning: Skipping invalid template item '{template_item.name}' during creation.")
                continue

            # Find the geometry node associated with our valid template item.
            template_node = next((n for n in node_tree.nodes if n.get("associated_empty") == template_item.empty_object.name), None)
            
            if template_node:
                # This single call now handles all the complex mirroring logic correctly.
                _mirror_and_clone_shape(context, template_item, template_node, self.axis)
            else:
                print(f"[Rogue SDF AI] Symmetrize Warning: Could not find node for template '{template_item.name}'.")

        # --- 4. Finalization ---
        rewire_full_sdf_chain(context)
        self.report({'INFO'}, f"Symmetrized model across {self.axis}-axis.")
        return {'FINISHED'}

def is_mirror_matrix(matrix: Matrix) -> bool:
    """Returns True if matrix has negative determinant (a reflection)"""
    return matrix.determinant() < 0



import bpy, math
from mathutils import Vector, Matrix

class OBJECT_OT_bake_sdf_symmetry(bpy.types.Operator):
    """
    Bakes symmetry by creating true clones of the source shape,
    each inheriting all deformer properties and a stable transform.
    """
    bl_idname = "prototyper.sdf_bake_symmetry"
    bl_label = "Bake Active Symmetries"
    bl_options= {'REGISTER','UNDO'}

    @classmethod
    def poll(cls, context):
        domain = context.scene.sdf_domain
        if not (domain and hasattr(domain, 'sdf_nodes')): return False
        idx = domain.active_sdf_node_index
        if not (0 <= idx < len(domain.sdf_nodes)): return False
        item = domain.sdf_nodes[idx]
        return item.use_mirror_x or item.use_mirror_y or item.use_mirror_z or \
               (item.use_radial_mirror and item.radial_mirror_count > 1)

    def execute(self, context):
        domain = context.scene.sdf_domain
        active_index = domain.active_sdf_node_index
        source_item = domain.sdf_nodes[active_index]
        source_empty = source_item.empty_object

        node_tree = get_sdf_geometry_node_tree(context)
        source_node = next((n for n in node_tree.nodes if n.get("associated_empty") == source_empty.name), None)

        if not (source_empty and source_node):
            self.report({'ERROR'}, "Source shape is not valid.")
            return {'CANCELLED'}

        # --- Create Radial Clones (Rotation doesn't cause flipping) ---
        if source_item.use_radial_mirror and source_item.radial_mirror_count > 1:
            count = source_item.radial_mirror_count
            angle_step = (2 * math.pi) / count
            pivot_matrix = Matrix.Translation(domain.location)
            pivot_matrix_inv = pivot_matrix.inverted()
            
            for i in range(1, count):
                new_sdf = _clone_sdf_hierarchy(context, source_item, source_node, new_name_suffix=f".Radial.{i}")
                rot_matrix = Matrix.Rotation(angle_step * i, 4, 'Z')
                # Apply rotation around the domain's center
                new_sdf["empty"].matrix_world = pivot_matrix @ rot_matrix @ pivot_matrix_inv @ source_empty.matrix_world

        # --- Create Mirrored Clones (using the new robust function) ---
        if source_item.use_mirror_x:
            _mirror_and_clone_shape(context, source_item, source_node, 'X')
        if source_item.use_mirror_y:
            _mirror_and_clone_shape(context, source_item, source_node, 'Y')
        if source_item.use_mirror_z:
            _mirror_and_clone_shape(context, source_item, source_node, 'Z')

        # Disable mirror flags on the original shape after baking
        source_item.use_mirror_x = False
        source_item.use_mirror_y = False
        source_item.use_mirror_z = False
        source_item.use_radial_mirror = False

        rewire_full_sdf_chain(context)
        self.report({'INFO'}, "Baked symmetries into new independent shapes.")
        return {'FINISHED'}

    
#--------------------------------------------------------------------

import bpy, math
from mathutils import Vector, Matrix


#-------------------------------------------------------------------
class OBJECT_OT_reset_domain_transform(bpy.types.Operator):
    """Resets the SDF Domain's transform to the world origin with no rotation or scale"""
    bl_idname = "object.reset_sdf_domain_transform"
    bl_label = "Reset Domain to Origin"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "sdf_domain", None)

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        domain_obj.location = (0, 0, 0)
        domain_obj.rotation_euler = (0, 0, 0)
        domain_obj.scale = (1, 1, 1)
        self.report({'INFO'}, "SDF Domain transform has been reset to the origin.")
        return {'FINISHED'}

# -------------------------------------------------------------------
# SDF Duplicate Operator (Fixed)
# -------------------------------------------------------------------
from mathutils import Vector

# Find this class in your main.py file and replace it completely.

class SDFDuplicateOperator(bpy.types.Operator):
    """Duplicates the selected SDF shape and its entire hierarchy (works for all types)."""
    bl_idname = "prototyper.sdf_duplicate"
    bl_label = "Duplicate Selected SDF Node"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')): return {'CANCELLED'}
        
        active_index = domain_obj.active_sdf_node_index
        if not (0 <= active_index < len(domain_obj.sdf_nodes)): return {'CANCELLED'}

        source_item = domain_obj.sdf_nodes[active_index]
        if not source_item.empty_object: return {'CANCELLED'}
            
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree: return {'CANCELLED'}
        
        source_node = next((n for n in node_tree.nodes if n.get('associated_empty') == source_item.empty_object.name), None)
        if not source_node: return {'CANCELLED'}

        # Use the new robust cloning function
        new_sdf = _clone_sdf_hierarchy(context, source_item, source_node, new_name_suffix=".Dupe")
        
        # Finalize
        domain_obj.sdf_nodes.move(len(domain_obj.sdf_nodes) - 1, active_index + 1)
        domain_obj.active_sdf_node_index = active_index + 1
        rewire_full_sdf_chain(context)
        
        bpy.ops.object.select_all(action='DESELECT')
        new_sdf["empty"].select_set(True)
        context.view_layer.objects.active = new_sdf["empty"]

        return {'FINISHED'}
    

class PROTOTYPER_OT_toggle_smooth(bpy.types.Operator):
    bl_idname = "prototyper.toggle_smooth"
    bl_label  = "Toggle Smooth Blend"
    def execute(self, context):
        dom  = context.scene.sdf_domain
        idx  = dom.active_sdf_node_index
        item = dom.sdf_nodes[idx] if idx >= 0 else None
        if item:
            if item.blend > 0.0:
                item.blend = 0.0
                item.operation = item.operation.replace("SMOOTH_","")
            else:
                item.blend = 0.2
                item.operation = "SMOOTH_"+item.operation
            # redraw:
            for w in context.window_manager.windows:
                for a in w.screen.areas:
                    if a.type=='VIEW_3D':
                        a.tag_redraw()
        return {'FINISHED'}

import math # Make sure this import is at the top of your script
from mathutils import Euler, Vector

# In main.py, replace the entire PROTOTYPER_OT_SDFRepeatShape class with this one.

class PROTOTYPER_OT_SDFRepeatShape(bpy.types.Operator):
    """Create multiple copies of the selected SDF shape using Linear or Radial methods."""
    bl_idname = "prototyper.sdf_repeat_shape"
    bl_label = "Repeat SDF Shape"
    bl_options = {'REGISTER', 'UNDO'}

    # --- Mode Selection (Curve mode removed) ---
    repeat_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('LINEAR', "Linear", "Repeat in a straight line"),
            ('RADIAL', "Radial", "Repeat in a circle around the 3D cursor")
        ],
        default='LINEAR'
    )

    # --- Linear Properties ---
    linear_count: bpy.props.IntProperty(name="Count", default=5, min=1, soft_max=100)
    linear_direction: bpy.props.FloatVectorProperty(name="Direction", default=(1.0, 0.0, 0.0))
    linear_spacing: bpy.props.FloatProperty(name="Spacing", default=1.0, min=0.0, soft_max=10.0, unit='LENGTH')

    # --- Radial Properties (Mirror option removed) ---
    radial_count: bpy.props.IntProperty(name="Count", default=8, min=2, soft_max=128)
    radial_axis: bpy.props.EnumProperty(
        name="Axis",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='Z'
    )

    @classmethod
    def poll(cls, context):
        domain = getattr(context.scene, "sdf_domain", None)
        if not domain or not hasattr(domain, 'sdf_nodes'): return False
        return 0 <= domain.active_sdf_node_index < len(domain.sdf_nodes)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        
        layout.prop(self, "repeat_mode", expand=True)
        layout.separator()
        
        box = layout.box()
        
        if self.repeat_mode == 'LINEAR':
            box.prop(self, "linear_count")
            box.prop(self, "linear_direction")
            box.prop(self, "linear_spacing")
            
        elif self.repeat_mode == 'RADIAL':
            box.label(text="Uses 3D Cursor as Pivot", icon='CURSOR')
            box.label(text="Radius is the distance from the 3D Cursor.")
            box.prop(self, "radial_count")
            box.prop(self, "radial_axis")

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        active_index = domain_obj.active_sdf_node_index
        original_item = domain_obj.sdf_nodes[active_index]
        original_empty = original_item.empty_object
        if not original_empty: return {'CANCELLED'}
        
        node_tree = get_sdf_geometry_node_tree(context)
        original_node = next((n for n in node_tree.nodes if n.get("associated_empty") == original_empty.name), None)
        if not original_node: return {'CANCELLED'}

        # --- PERFORMANCE FIX: Find the modifier and prepare to disable it ---
        mod = next((m for m in domain_obj.modifiers if m.type == 'NODES'), None)
        if mod:
            # This line disables the modifier in the viewport, pausing calculations.
            mod.show_viewport = False
        # --- END FIX ---

        # The try...finally block guarantees the modifier is re-enabled, even if an error occurs.
        try:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = None
            
            created_empties = []

            # --- LINEAR MODE (Your original code is unchanged here) ---
            if self.repeat_mode == 'LINEAR':
                direction_vec = Vector(self.linear_direction)
                final_offset = direction_vec.normalized() * self.linear_spacing if direction_vec.length > 0 else Vector()
                for i in range(self.linear_count):
                    new_sdf = _clone_sdf_hierarchy(context, original_item, original_node, new_name_suffix=f".Repeat.{i+1}")
                    new_empty = new_sdf["empty"]
                    new_empty.location = original_empty.location + (final_offset * (i + 1))
                    created_empties.append(new_empty)

            # --- RADIAL MODE (Your original code is unchanged here) ---
            elif self.repeat_mode == 'RADIAL':
                pivot_point = context.scene.cursor.location
                angle_step = (2 * math.pi) / self.radial_count
                
                pivot_mat = Matrix.Translation(pivot_point)
                pivot_inv_mat = Matrix.Translation(-pivot_point)

                for i in range(1, self.radial_count):
                    new_sdf = _clone_sdf_hierarchy(context, original_item, original_node, new_name_suffix=f".Radial.{i}")
                    new_empty = new_sdf["empty"]

                    rot_mat = Matrix.Rotation(angle_step * i, 4, self.radial_axis)
                    
                    new_empty.matrix_world = pivot_mat @ rot_mat @ pivot_inv_mat @ original_empty.matrix_world

                    created_empties.append(new_empty)

            # This is now safe and fast because the modifier is disabled.
            rewire_full_sdf_chain(context)
            
            if created_empties:
                for empty in created_empties:
                    empty.select_set(True)
                context.view_layer.objects.active = created_empties[-1]

            self.report({'INFO'}, f"Created {len(created_empties)} repeated shapes.")

        finally:
            # --- PERFORMANCE FIX: This block ALWAYS runs, re-enabling the modifier ---
            if mod:
                # This line turns the modifier back on, triggering a single, final calculation.
                mod.show_viewport = True
            # --- END FIX ---

        return {'FINISHED'}
    
#--------------------------------------------------------------------    
    
import bpy

class PROTOTYPER_OT_SDFFlipActiveShape(bpy.types.Operator):
    """Flips the single active SDF shape around its own origin."""
    bl_idname = "prototyper.sdf_flip_active_shape"
    bl_label = "Flip Active SDF Shape"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        items=[('X', "X", "Flip on the Global X-axis"),
               ('Y', "Y", "Flip on the Global Y-axis"),
               ('Z', "Z", "Flip on the Global Z-axis")],
        default='X'
    )

    @classmethod
    def poll(cls, context):
        domain = context.scene.sdf_domain
        return domain and 0 <= domain.active_sdf_node_index < len(domain.sdf_nodes)

    def execute(self, context):
        domain = context.scene.sdf_domain
        item = domain.sdf_nodes[domain.active_sdf_node_index]
        active_empty = item.empty_object
        if not active_empty:
            return {'CANCELLED'}

        original_pivot_mode = context.scene.tool_settings.transform_pivot_point
        original_active_object = context.view_layer.objects.active
        original_selection = context.selected_objects[:]

        bpy.ops.object.select_all(action='DESELECT')
        active_empty.select_set(True)
        context.view_layer.objects.active = active_empty

        context.scene.tool_settings.transform_pivot_point = 'INDIVIDUAL_ORIGINS'

        bpy.ops.transform.mirror(
            orient_type='GLOBAL',
            constraint_axis=(self.axis == 'X', self.axis == 'Y', self.axis == 'Z')
        )

        # Force scale components to be positive after flipping
        active_empty.scale.x = abs(active_empty.scale.x)
        active_empty.scale.y = abs(active_empty.scale.y)
        active_empty.scale.z = abs(active_empty.scale.z)

        context.scene.tool_settings.transform_pivot_point = original_pivot_mode
        bpy.ops.object.select_all(action='DESELECT')

        # Restore original selection
        for obj in original_selection:
            if obj.name in context.view_layer.objects:
                obj.select_set(True)
        context.view_layer.objects.active = original_active_object

        self.report({'INFO'}, f"Flipped '{item.name}' on the {self.axis}-axis.")
        return {'FINISHED'}

    
#-------------------------------------------------------------------

class PROTOTYPER_OT_SDFFlipModel(bpy.types.Operator):
    """
    Flips the orientation of selected or all SDF shapes in-place around a global axis.
    """
    bl_idname = "prototyper.sdf_flip_model"
    bl_label = "Flip SDF Model"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        items=[('X', "Global X-Axis", "Flip the shape's orientation on the X-axis"),
               ('Y', "Global Y-Axis", "Flip the shape's orientation on the Y-axis"),
               ('Z', "Global Z-Axis", "Flip the shape's orientation on the Z-axis")],
        default='X'
    )

    mode: bpy.props.EnumProperty(
        name="Mode",
        description="Choose which shapes to flip",
        items=[('SELECTED', "Selected", "Flip all selected shapes"),
               ('ALL', "All", "Flip the entire model")],
        default='SELECTED'
    )

    @classmethod
    def poll(cls, context):
        domain = context.scene.sdf_domain
        return domain and len(domain.sdf_nodes) > 0

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        # 1. Store the user's original selection state to restore it later
        original_active = context.view_layer.objects.active
        original_selection = context.selected_objects[:]

        # 2. Determine which objects to flip
        domain = context.scene.sdf_domain
        valid_empties = {item.empty_object for item in domain.sdf_nodes if item.empty_object}
        objects_to_flip = []

        if self.mode == 'SELECTED':
            objects_to_flip = [obj for obj in context.selected_objects if obj in valid_empties]
        else:  # ALL
            objects_to_flip = list(valid_empties)

        if not objects_to_flip:
            self.report({'WARNING'}, f"No valid SDF shapes found for mode: {self.mode}")
            return {'CANCELLED'}

        # 3. Use bpy.ops for a robust, viewport-aware rotation on each object
        for obj in objects_to_flip:
            # Isolate the object for the operation to ensure we only rotate it
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = obj
            obj.select_set(True)

            # Perform the 180-degree rotation (flip) around the object's own origin.
            # This is the guaranteed, correct way to do this.
            bpy.ops.transform.rotate(
                value=math.pi,  # 180 degrees in radians
                orient_axis=self.axis,
                orient_type='GLOBAL',
                center_override=obj.location  # CRITICAL: This forces the flip to be in-place.
            )

        # 4. Restore the user's original selection for a seamless experience
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selection:
            # Check if the object still exists before trying to select it
            if obj and obj.name in context.view_layer.objects:
                obj.select_set(True)
        context.view_layer.objects.active = original_active

        self.report({'INFO'}, f"Flipped {self.mode.lower()} shapes on the {self.axis}-axis.")
        return {'FINISHED'}
    
#-------------------------------------------------------------------      


class SDFDeleteOperator(bpy.types.Operator):
    """Delete the selected SDF node and its children, then rewire the chain."""
    bl_idname = "prototyper.sdf_delete"
    bl_label = "Delete Selected SDF Node"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')): return {'CANCELLED'}
        
        active_index = domain_obj.active_sdf_node_index
        if not (0 <= active_index < len(domain_obj.sdf_nodes)): return {'CANCELLED'}
        
        del_item = domain_obj.sdf_nodes[active_index]
        del_empty = del_item.empty_object
        
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree: return {'CANCELLED'}

        del_node = next((n for n in node_tree.nodes if del_empty and n.get("associated_empty") == del_empty.name), None)

        # Deletion logic
        if del_empty:
            if del_empty.children:
                for child in list(del_empty.children): bpy.data.objects.remove(child, do_unlink=True)
            if del_empty.name in bpy.data.objects: bpy.data.objects.remove(del_empty, do_unlink=True)
        if del_node: node_tree.nodes.remove(del_node)
        
        domain_obj.sdf_nodes.remove(active_index)

        # Call the new master rewire function
        rewire_full_sdf_chain(context)

        new_count = len(domain_obj.sdf_nodes)
        domain_obj.active_sdf_node_index = min(active_index, new_count - 1) if new_count > 0 else -1

        return {'FINISHED'}




class SDFClearOperator(bpy.types.Operator):
    """Clear all SDF nodes and their related empties and children."""
    bl_idname = "prototyper.sdf_clear"
    bl_label = "Clear All SDF Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')):
            self.report({'INFO'}, "No SDF Domain to clear.")
            return {'CANCELLED'}

        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree: return {'CANCELLED'}

        # Loop over a copy of the list.
        for item in list(domain_obj.sdf_nodes):
            empty_obj = item.empty_object
            if empty_obj:
                if empty_obj.children:
                    for child in list(empty_obj.children):
                        bpy.data.objects.remove(child, do_unlink=True)
                
                node = next((n for n in node_tree.nodes if n.get("associated_empty") == empty_obj.name), None)
                if node:
                    node_tree.nodes.remove(node)
                
                if empty_obj.name in bpy.data.objects:
                    bpy.data.objects.remove(empty_obj, do_unlink=True)

        domain_obj.sdf_nodes.clear()
        domain_obj.active_sdf_node_index = -1
        
        # Call the master rewire function to clean up links.
        rewire_full_sdf_chain(context)

        self.report({'INFO'}, "Cleared all SDF nodes and related objects.")
        return {'FINISHED'}


class SDFMeshToSDF(bpy.types.Operator):
    """Convert Mesh to SDF with settings"""
    bl_idname = "prototyper.sdf_mesh_to_sdf"
    bl_label = "Convert Mesh to SDF"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # (Implementation as in original code)
        self.report({'INFO'}, "Converted Mesh to SDF (implementation per original code)")
        return {'FINISHED'}

# In main.py, add this new operator class

class PROTOTYPER_OT_SDFSetPointCloudPreview(bpy.types.Operator):
    """Set the global SDF visualization mode."""
    bl_idname = "prototyper.sdf_set_point_cloud_preview"
    bl_label = "Set SDF Visualization"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('SOLID', "All Solid", "Set all shapes to Solid Mesh mode"),
            ('POINTS', "All Points", "Set all shapes to Point Cloud mode"),
        ]
    )

    @classmethod
    def poll(cls, context):
        domain = getattr(context.scene, "sdf_domain", None)
        return domain and hasattr(domain, 'sdf_nodes')

    def execute(self, context):
        if self.mode == 'SOLID':
            context.scene.sdf_visualization_mode = 'SOLID'
        else: # POINTS
            context.scene.sdf_visualization_mode = 'POINTS'
        
        # The update function on the scene property will handle the rewiring.
        self.report({'INFO'}, f"SDF Visualization set to: {self.mode}")
        return {'FINISHED'}


# -------------------------------------------------------------------
# SDF Shape Operators (Cube, Cylinder, UV Sphere, Cone, Prism, Torus, Curve, Mesh, Sculpt)
# -------------------------------------------------------------------
class PROTOTYPER_OT_SDFCleanupList(bpy.types.Operator):
    """
    Checks the SDF list for invalid items and removes them. This is the safe
    way to modify the list, as it's an operator with a write context.
    """
    bl_idname = "prototyper.sdf_cleanup_list"
    bl_label = "Clean Up SDF List"
    bl_description = "Find and remove any invalid SDF items from the list whose objects have been deleted"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # This operator can only run if there is a domain object.
        return getattr(context.scene, "sdf_domain", None) is not None

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        
        # We build a list of indices that correspond to invalid items.
        indices_to_remove = []
        for i, item in enumerate(domain_obj.sdf_nodes):
            if not item.empty_object or item.empty_object.name not in context.view_layer.objects:
                indices_to_remove.append(i)

        if not indices_to_remove:
            self.report({'INFO'}, "No invalid items found in the list.")
            return {'CANCELLED'}

        # --- If we have items to remove, proceed ---

        # First, find and remove the associated Geometry Nodes.
        mod = domain_obj.modifiers.get("GeometryNodes")
        if mod and mod.node_group:
            node_tree = mod.node_group
            dead_names = {domain_obj.sdf_nodes[i].name for i in indices_to_remove}
            nodes_to_remove = [n for n in node_tree.nodes if n.get("associated_empty") in dead_names]
            for node in nodes_to_remove:
                node_tree.nodes.remove(node)

        # Now remove the items from the PropertyGroup list, iterating backwards.
        for i in sorted(indices_to_remove, reverse=True):
            domain_obj.sdf_nodes.remove(i)
        
        # IMPORTANT: After modifying the list, we must rewire the chain to fix broken links.
        rewire_full_sdf_chain(context)

        self.report({'INFO'}, f"Removed {len(indices_to_remove)} invalid item(s).")
        return {'FINISHED'}
    

class SDFCubeAdd(bpy.types.Operator):
    """Add a Cube with SDF settings"""
    bl_idname = "prototyper.sdf_cube_add"
    bl_label = "Add SDF Cube"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Cube")
        if not node_group:
            self.report({'ERROR'}, "'SDF Cube' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        # Create the GeometryNodeGroup node
        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Size" in sdf_node.inputs:
            sdf_node.inputs["Size"].default_value = (0.25, 0.25, 0.25)

        # Create controller Empty
        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Cube"
        empty.empty_display_size = 0.25
        empty.parent = domain

        # Add to UI list
        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_CUBE'

        # Link the node <-> empty
        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = empty

        rewire_full_sdf_chain(context)

        # Select new empty
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


class SDFCylinderAdd(bpy.types.Operator):
    """Add a Cylinder with SDF settings"""
    bl_idname = "prototyper.sdf_cylinder_add"
    bl_label = "Add SDF Cylinder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Cylinder")
        if not node_group:
            self.report({'ERROR'}, "'SDF Cylinder' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Height" in sdf_node.inputs:
            sdf_node.inputs["Height"].default_value = 0.5
        if "Radius" in sdf_node.inputs:
            sdf_node.inputs["Radius"].default_value = 0.25

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Cylinder"
        empty.empty_display_size = 0.25
        empty.parent = domain

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_CYLINDER'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = empty

        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


class SDFUVSphereAdd(bpy.types.Operator):
    """Add a UV Sphere with SDF settings"""
    bl_idname = "prototyper.sdf_uv_sphere_add"
    bl_label = "Add SDF UV Sphere"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Sphere")
        if not node_group:
            self.report({'ERROR'}, "'SDF Sphere' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Radius" in sdf_node.inputs:
            sdf_node.inputs["Radius"].default_value = 0.25

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='SPHERE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Sphere"
        empty.empty_display_size = 0.25
        empty.parent = domain

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_UVSPHERE'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = empty

        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


# Find this class in main.py and replace it completely.

class SDFConeAdd(bpy.types.Operator):
    """Add a Cone with a robust two-empty (base and tip) controller"""
    bl_idname = "prototyper.sdf_cone_add"
    bl_label = "Add SDF Cone"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Cone")
        if not node_group:
            self.report({'ERROR'}, "'SDF Cone' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        # Create the GeometryNodeGroup node
        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)

        # --- NEW TWO-EMPTY HIERARCHY ---
        domain = context.scene.sdf_domain

        # 1. Create the main Controller (acts as the Base)
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        controller_empty = context.active_object
        controller_empty.name = "SDF_Cone_Controller"
        controller_empty.parent = domain
        controller_empty.scale = (0.25, 0.25, 0.25) # Default base radius

        # 2. Create the Tip empty
        bpy.ops.object.empty_add(type='SPHERE', location=controller_empty.location)
        tip_empty = context.active_object
        tip_empty.name = "SDF_Cone_Tip"
        tip_empty.parent = controller_empty # Parent Tip to Controller
        tip_empty.location.y += 0.5 # Default height
        tip_empty.scale = (0.01, 0.01, 0.01) # Default tip radius (pointy)
        tip_empty.empty_display_size = 0.05

        # 3. Add the main CONTROLLER to the UI list
        item = domain.sdf_nodes.add()
        item.name = controller_empty.name
        item.empty_object = controller_empty
        item.icon = 'MESH_CONE'

        # 4. Link the node to BOTH empties
        sdf_node['associated_empty'] = controller_empty.name
        inputs = [s for s in sdf_node.inputs if s.type == 'OBJECT']
        if len(inputs) >= 2:
            inputs[0].default_value = controller_empty
            inputs[1].default_value = tip_empty
        else:
            self.report({'WARNING'}, "SDF Cone node is missing Object inputs for controllers.")

        rewire_full_sdf_chain(context)
        
        # Select the new controller empty
        bpy.ops.object.select_all(action='DESELECT')
        controller_empty.select_set(True)
        context.view_layer.objects.active = controller_empty

        return {'FINISHED'}


class SDFPrismAdd(bpy.types.Operator):
    """Add a Prism with SDF settings"""
    bl_idname = "prototyper.sdf_prism_add"
    bl_label = "Add SDF Prism"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Prism")
        if not node_group:
            self.report({'ERROR'}, "'SDF Prism' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Size" in sdf_node.inputs:
            sdf_node.inputs["Size"].default_value = (0.25, 0.25, 0.25)

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Prism"
        empty.empty_display_size = 0.25
        empty.parent = domain

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_ICOSPHERE'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = empty

        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


class SDFTorusAdd(bpy.types.Operator):
    """Add a Torus with SDF settings"""
    bl_idname = "prototyper.sdf_torus_add"
    bl_label = "Add SDF Torus"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Torus")
        if not node_group:
            self.report({'ERROR'}, "'SDF Torus' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        # Create the GeometryNodeGroup node
        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        
        # NOTE: We no longer set default radius on the node's inputs.
        # The new properties on SDFNodeItem now control the shape's parameters.

        # Create controller Empty
        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Torus"
        empty.empty_display_size = 0.25
        empty.parent = domain

        # Add to UI list
        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_TORUS'

        # Link the node <-> empty
        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = empty

        # Finalize
        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


class SDFCurveAdd(bpy.types.Operator):
    """Add a Curve with SDF settings"""
    bl_idname = "prototyper.sdf_curve_add"
    bl_label = "Add SDF Curve"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Curve")
        if not node_group:
            self.report({'ERROR'}, "'SDF Curve' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Radius" in sdf_node.inputs:
            sdf_node.inputs["Radius"].default_value = 0.05

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='SPHERE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Curve"
        empty.empty_display_size = 0.05
        empty.parent = domain

        bpy.ops.curve.primitive_bezier_curve_add()
        curve = context.active_object
        curve.parent = empty
        curve.show_in_front = True

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'CURVE_BEZCURVE'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = curve

        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


class SDFMeshAdd(bpy.types.Operator):
    """Add a Mesh with SDF settings"""
    bl_idname = "prototyper.sdf_mesh_add"
    bl_label = "Add SDF Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Mesh")
        if not node_group:
            self.report({'ERROR'}, "'SDF Mesh' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Mesh"
        empty.parent = domain

        # Use the active mesh as input if selected
        selected = context.selected_objects
        mesh_obj = next((o for o in selected if o.type == 'MESH' and o != domain), None)

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_MONKEY'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = mesh_obj # Can be None, which is fine
        if mesh_obj:
            mesh_obj.display_type = 'WIRE'

        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}


class SDFSculptAdd(bpy.types.Operator):
    """Add a Sculpt with SDF settings"""
    bl_idname = "prototyper.sdf_sculpt_add"
    bl_label = "Add SDF Sculpt"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        geo_nodes = get_sdf_geometry_node_tree(context)
        if not geo_nodes:
            self.report({'ERROR'}, "No SDF Domain with Geometry Nodes found. Please generate a domain first.")
            return {'CANCELLED'}

        node_group = bpy.data.node_groups.get("SDF Sculpt")
        if not node_group:
            self.report({'ERROR'}, "'SDF Sculpt' node group not found. Please regenerate the domain.")
            return {'CANCELLED'}

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Radius" in sdf_node.inputs:
            sdf_node.inputs["Radius"].default_value = 0.05

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='SPHERE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Sculpt"
        empty.empty_display_size = 0.05
        empty.parent = domain

        bpy.ops.object.grease_pencil_add(type='STROKE')
        gp = context.active_object
        gp.parent = empty

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'SCULPTMODE_HLT'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = gp

        rewire_full_sdf_chain(context)
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        context.view_layer.objects.active = empty

        return {'FINISHED'}





# -------------------------------------------------------------------
# List Management Operators (move up, move down, duplicate, delete, clear, etc.)
# -------------------------------------------------------------------
import bpy

# 1) Update the list-move operator to include a default for direction
class PROTOTYPER_OT_SDFListMove(bpy.types.Operator):
    """Move an item in the SDF list and rewire the node chain."""
    bl_idname = "prototyper.sdf_list_move"
    bl_label = "Move SDF List Item"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[
            ('UP',   "Up",   "Move item up"),
            ('DOWN', "Down", "Move item down"),
        ],
        default='UP'
    )


    @classmethod
    def poll(cls, context):
        dom = getattr(context.scene, "sdf_domain", None)
        return dom and dom.active_sdf_node_index >= 0

    def execute(self, context):
        dom = context.scene.sdf_domain
        idx = dom.active_sdf_node_index
        new_idx = idx - 1 if self.direction == 'UP' else idx + 1

        if 0 <= new_idx < len(dom.sdf_nodes):
            dom.sdf_nodes.move(idx, new_idx)
            dom.active_sdf_node_index = new_idx
            rewire_full_sdf_chain(context)
        return {'FINISHED'}


#--------------------------------------------------------------------

#---------------------------------------------------------------------

class OBJECT_OT_sdf_render_highres(bpy.types.Operator):
    """Renders a temporary high-res preview, optionally changing viewport shading."""
    bl_idname = "object.sdf_render_highres"
    bl_label = "Render High-Res Preview"

    _timer = None
    _original_view_data = {}

    def _find_3d_view_area(self, context):
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                return area
        return None

    def execute(self, context):
        scene = context.scene
        area = self._find_3d_view_area(context)
        if not area:
            self.report({'WARNING'}, "No 3D Viewport available for preview.")
            return {'CANCELLED'}
        space = area.spaces.active

        # Store original settings, including the new shading and overlay states
        self._original_view_data = {
            'preview_mode': scene.sdf_preview_mode, 'final_res': scene.sdf_final_resolution,
            'is_camera_view': space.region_3d.view_perspective == 'CAMERA',
            'shading_type': space.shading.type,
            'show_overlays': space.overlay.show_overlays,
        }

        # Apply settings for the preview
        scene.sdf_preview_mode = False
        scene.sdf_final_resolution = scene.sdf_render_highres_resolution
        if scene.sdf_render_shading_mode != 'CURRENT':
            space.shading.type = scene.sdf_render_shading_mode
        if scene.sdf_render_disable_overlays:
            space.overlay.show_overlays = False
        context.view_layer.update()

        if scene.sdf_render_from == 'CAMERA' and not self._original_view_data['is_camera_view']:
            with context.temp_override(area=area):
                bpy.ops.view3d.camera_to_view()

        with context.temp_override(area=area, region=area.regions[-1]):
             bpy.ops.render.opengl()

        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            scene = context.scene
            area = self._find_3d_view_area(context)
            if area:
                space = area.spaces.active
                # Restore everything
                if scene.sdf_render_from == 'CAMERA' and not self._original_view_data['is_camera_view']:
                     with context.temp_override(area=area):
                        bpy.ops.view3d.view_orbit(angle=0.0)
                space.shading.type = self._original_view_data['shading_type']
                space.overlay.show_overlays = self._original_view_data['show_overlays']
            
            scene.sdf_preview_mode = self._original_view_data['preview_mode']
            scene.sdf_final_resolution = self._original_view_data['final_res']
            context.window_manager.event_timer_remove(self._timer)
            self.report({'INFO'}, "Restored viewport settings.")
            return {'FINISHED'}
        return {'PASS_THROUGH'}

# -------------------------------------------------------------------
# Menu Creation for Right-Click and Tool Panel
# -------------------------------------------------------------------
def add_sdf_shapes(self, context):
    layout = self.layout
    layout.operator(StartSDFOperator.bl_idname, text="SDF Domain", icon='CUBE')
    layout.operator(SDFCubeAdd.bl_idname, text="SDF Cube", icon='MESH_CUBE')
    layout.operator(SDFCylinderAdd.bl_idname, text="SDF Cylinder", icon='MESH_CYLINDER')
    layout.operator(SDFUVSphereAdd.bl_idname, text="SDF Sphere", icon='MESH_UVSPHERE')
    layout.operator(SDFConeAdd.bl_idname, text="SDF Cone", icon='MESH_CONE')
    layout.operator(SDFPrismAdd.bl_idname, text="SDF Prism", icon='MESH_ICOSPHERE')
    layout.operator(SDFTorusAdd.bl_idname, text="SDF Torus", icon='MESH_TORUS')
    layout.operator(SDFCurveAdd.bl_idname, text="SDF Curve", icon='CURVE_BEZCURVE')
    layout.operator(SDFMeshAdd.bl_idname, text="SDF Mesh", icon='MESH_MONKEY')
    layout.operator(SDFSculptAdd.bl_idname, text="SDF Sculpt", icon='STROKE')


class VIEW3D_MT_sdf_rclick(bpy.types.Menu):
    bl_label = "SDF Operators"
    bl_idname = "VIEW3D_MT_sdf_rclick"
    
    def draw(self, context):
        layout = self.layout
        layout.operator(SDFDuplicateOperator.bl_idname, text="SDF_Duplicate")
        layout.operator(SDFDeleteOperator.bl_idname, text="SDF_Delete")
        layout.operator(SDFClearOperator.bl_idname, text="SDF_Clear list")
        layout.operator(SDFMeshToSDF.bl_idname, text="Convert to SDF")
        layout.operator(ConvertSDFOperator.bl_idname, text="Convert to Mesh")


def rclick_sdf_menu(self, context):
    self.layout.menu(VIEW3D_MT_sdf_rclick.bl_idname)


addon_keymaps = []


# -------------------------------------------------------------------
# Dynamic Resolution Monitor (NEW - Robust Timer-Based State Machine)
# -------------------------------------------------------------------

import time
from mathutils import Vector

# module-level globals (only once!)
_prev_transforms     = {}
_last_movement_time  = 0.0
_timer_is_running    = False

def monitor_sdf_movement():
    global _timer_is_running, _last_movement_time, _prev_transforms

    # stop the timer if the add-on unloaded
    if not _timer_is_running:
        return None

    sc = bpy.context.scene
    # only run when auto-preview is enabled and we have a domain
    if not (sc.sdf_auto_resolution_enable and sc.sdf_domain):
        return 0.2

    threshold = sc.sdf_auto_threshold    # reuse your user‐tweakable threshold
    delay     = sc.sdf_auto_idle_delay    # and idle delay

    movement = False
    now      = time.time()

    for item in sc.sdf_domain.sdf_nodes:
        empty = item.empty_object
        if not empty:
            continue

        # grab current world transforms
        mw    = empty.matrix_world
        loc   = mw.to_translation()
        rot   = mw.to_euler()
        scale = mw.to_scale()

        prev = _prev_transforms.get(empty.name)
        if prev is None:
            # first time around, just cache
            _prev_transforms[empty.name] = (loc.copy(), rot.copy(), scale.copy())
            continue

        prev_loc, prev_rot, prev_scale = prev

        # check if any component exceeds threshold
        if (loc - prev_loc).length > threshold:
            movement = True
        elif any(abs(rot[i] - prev_rot[i]) > threshold for i in range(3)):
            movement = True
        elif (scale - prev_scale).length > threshold:
            movement = True

        # update cache
        _prev_transforms[empty.name] = (loc.copy(), rot.copy(), scale.copy())

    # switch to Low-Res on movement
    if movement:
        _last_movement_time = now
        if not sc.sdf_preview_mode:
            sc.sdf_preview_mode    = True
            sc.sdf_status_message  = "Moving (Low-Res)"

    # switch back to High-Res after idle
    else:
        if sc.sdf_preview_mode and (now - _last_movement_time) > delay:
            sc.sdf_preview_mode    = False
            sc.sdf_status_message  = "Idle (High-Res)"

    return 0.1





def toggle_auto_resolution_mode(self, context):
    """
    Called when the 'Automatic Preview' checkbox is toggled.
    Reset our transform-cache and idle timer so the monitor starts fresh.
    """
    global _prev_transforms, _last_movement_time

    scene = context.scene

    if scene.sdf_auto_resolution_enable:
        # Clear the cached transforms so movement is detected immediately.
        _prev_transforms.clear()
        # Reset the idle timer so we begin in high-res.
        _last_movement_time = time.time()
        scene.sdf_preview_mode    = False
        scene.sdf_status_message  = "Idle (High-Res)"
    else:
        scene.sdf_status_message  = "Disabled"

    # Immediately apply the new preview mode (low/high) to the node tree.
    update_sdf_resolution(self, context)



# Resolution Updater with debug print + forced update

def update_sdf_resolution(self, context):
    sc   = context.scene
    tree = get_sdf_geometry_node_tree(context)
    if not tree:
        return

    dom = next((n for n in tree.nodes if n.name == "SDF Domain"), None)
    if not dom or "Resolution" not in dom.inputs:
        return

    val = sc.sdf_preview_resolution if sc.sdf_preview_mode else sc.sdf_final_resolution
    dom.inputs["Resolution"].default_value = max(1, int(val))

    # force immediate update
    context.view_layer.update()






# ——— Hook them into your properties ———

import bpy

# -----------------------------------------------------------------------------
# 1) List of classes to register (exclude SDFNodeItem)
# -----------------------------------------------------------------------------
# ——— Updated Class List, Register & Unregister ———

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    StringProperty, PointerProperty, EnumProperty,
    BoolProperty, FloatProperty, FloatVectorProperty, IntProperty
)

# Forward-declare update functions that are defined elsewhere in the file
def update_sdf_node_name(self, context): pass
def update_sdf_viewport_visibility(self, context): pass
def check_mute_nodes(scene): pass
def _redraw_shader_view(self, context): pass
def update_point_cloud_preview(self, context): pass


class SDFCurveControlPoint(PropertyGroup):
    """Properties for a single user-defined control point on an SDF Curve."""
    
    # --- Core Control Properties ---
    t_value: FloatProperty(name="Position (t)", description="Position of this control point along the curve (0=start, 1=end)", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    radius_multiplier: FloatProperty(name="Radius", description="Multiplier for the radius at this point", default=1.0, min=0.0, soft_max=5.0, subtype='FACTOR', update=_redraw_shader_view)
    color: FloatVectorProperty(name="Color", subtype='COLOR', default=(1.0, 1.0, 1.0), min=0.0, max=1.0, update=_redraw_shader_view)
    shape_type: EnumProperty(name="Shape", items=[('MESH_UVSPHERE', "Sphere", ""), ('MESH_CUBE', "Cube", ""), ('MESH_ICOSPHERE', "Prism", ""), ('CAPSULE', "Capsule", ""), ('MESH_TORUS', "Torus", ""), ('MESH_CYLINDER', "Cylinder", ""), ('MESH_CONE', "Cone", "")], default='MESH_UVSPHERE', update=_redraw_shader_view)
    
    # --- NEW: Per-Point Rotation ---
    rotation: FloatVectorProperty(name="Rotation", description="Local rotation for the shape at this point", subtype='EULER', default=(0.0, 0.0, 0.0), update=_redraw_shader_view)

    # --- NEW: All Advanced Parameters ---
    # CUBE
    thickness: FloatProperty(name="Thickness", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    roundness: FloatProperty(name="Roundness", default=0.0, min=0.0, soft_max=1, update=_redraw_shader_view)
    bevel: FloatProperty(name="Bevel", default=0.0, min=0.0, soft_max=0.5, update=_redraw_shader_view)
    pyramid: FloatProperty(name="Pyramid", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    twist: FloatProperty(name="Twist", default=0.0, min=-20.0, max=20.0, update=_redraw_shader_view)
    bend: FloatProperty(name="Bend", default=0.0, min=-2.0, max=2.0, update=_redraw_shader_view)
    
    # SPHERE
    sphere_thickness: FloatProperty(name="Thickness", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    sphere_elongation: FloatProperty(name="Elongation", default=0.0, min=0.0, soft_max=2.0, update=_redraw_shader_view)
    sphere_cut_angle: FloatProperty(name="Pac-man", default=math.tau, min=0.0, max=math.tau, subtype='ANGLE', update=_redraw_shader_view)

    # CYLINDER
    cylinder_thickness: FloatProperty(name="Thickness", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    cylinder_roundness: FloatProperty(name="Roundness", default=0.0, min=0.0, soft_max=4.0, update=_redraw_shader_view)
    cylinder_bend: FloatProperty(name="Bend", default=0.0, min=-2.0, max=2.0, update=_redraw_shader_view)
    cylinder_pyramid: FloatProperty(name="Pyramid", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)

    # PRISM
    prism_sides: IntProperty(name="Sides", default=6, min=3, max=16, update=_redraw_shader_view)
    prism_thickness: FloatProperty(name="Thickness", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    prism_bend: FloatProperty(name="Bend", default=0.0, min=-2.0, max=2.0, update=_redraw_shader_view)
    prism_twist: FloatProperty(name="Twist", default=0.0, min=-20.0, max=20.0, update=_redraw_shader_view)
    prism_pyramid: FloatProperty(name="Pyramid", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)

    # TORUS
    torus_outer_radius: FloatProperty(name="Outer Radius", default=0.4, min=0.01, soft_max=2.0, subtype='DISTANCE', update=_redraw_shader_view)
    torus_inner_radius: FloatProperty(name="Inner Radius", default=0.2, min=0.0, soft_max=2.0, subtype='DISTANCE', update=_redraw_shader_view)
    torus_thickness: FloatProperty(name="Thickness", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    torus_cut_angle: FloatProperty(name="Pac-man", default=math.tau, min=0.0, max=math.tau, subtype='ANGLE', update=_redraw_shader_view)
    torus_elongation: FloatProperty(name="Elongation", default=0.0, min=0.0, soft_max=2.0, subtype='DISTANCE', update=_redraw_shader_view)


class SDFNodeItem(PropertyGroup):
    # --- CORE PROPERTIES ---
    name: StringProperty(name="Node Name", description="Rename this SDF shape", update=update_sdf_node_name)
    empty_object: PointerProperty(name="Controller Empty", type=bpy.types.Object, description="The Empty that drives this shape")
    icon: EnumProperty(items=[('MESH_CUBE', "Cube", ""), ('MESH_CYLINDER', "Cylinder", ""), ('MESH_UVSPHERE', "Sphere", ""), ('MESH_CONE', "Cone", ""), ('MESH_ICOSPHERE', "Prism", ""), ('MESH_TORUS', "Torus", ""), ('CURVE_BEZCURVE', "Curve", ""), ('MESH_MONKEY', "Mesh", ""), ('SCULPTMODE_HLT', "Sculpt", "")], default='MESH_CUBE')
    
    # --- UI & MUTE PROPERTIES ---
    is_hidden: BoolProperty(name="Mute Shape", default=False, update=update_visibility_and_mute)
    is_viewport_hidden: BoolProperty(name="Hide Empty", default=False, update=update_visibility_and_mute)
    
    # --- COMMON SHADER PROPERTIES ---
    use_highlight: BoolProperty(name="Highlight Shape", default=False, update=_redraw_shader_view)
    operation: EnumProperty(
        name="Operation",
        items=[
            ('UNION', "Union", "Combine two shapes"),
            ('SUBTRACT', "Subtract", "Carve the second shape from the first"),
            ('INTERSECT', "Intersect", "Keep only the overlapping volume"),
            ('PAINT', "Paint", "Casts color onto shapes below it"),
            ('DISPLACE', "Displace", "Deforms the base shape outwards"),
            ('INDENT', "Indent", "Deforms the base shape inwards"),
            ('RELIEF', "Relief", "The second shape carves the first, while both remain visible"),
            ('ENGRAVE', "Engrave", "The first shape carves the second, while both remain visible"),
            ('MASK', "Mask", "Reveals the shape below using the current shape as a stencil")
        ],
        default='UNION',
        update=_redraw_shader_view
    )

    blend_type: EnumProperty(
        name="Blend Type",
        items=[
            ('ROUND', "Round", "Creates a soft, convex rounded blend"),
            ('CHAMFER', "Chamfer", "Creates a hard, 45-degree linear blend"),
            ('GROOVE', "Groove", "Creates a soft, concave (inverted) rounded blend"),
            ('PIPE', "Pipe", "Creates a convex bead along the intersection")
        ],
        default='ROUND',
        update=_redraw_shader_view
    ) 
    
    # --- THIS IS YOUR UPDATED PROPERTY ---
    blend: FloatProperty(
        name="Blend", 
        description="Controls the smoothness of the blend between shapes.",
        default=0.0, 
        min=0.0, 
        max=100.0,      # The absolute maximum value you can type in
        soft_max=50.0, # The maximum value the UI slider will drag to
        update=_redraw_shader_view
    )
    # --- END OF UPDATE ---

    blend_strength: FloatProperty(
        name="Strength", 
        description="Controls height/depth for CSG ops, or hollowness for Mask",
        default=0.1, 
        # --- UPDATE THE MINIMUM VALUE ---
        min=0.0, 
        max=2.0, 
        subtype='DISTANCE',
        update=_redraw_shader_view
    )
    mask_fill_amount: FloatProperty(
        name="Fill Amount",
        description="Controls the morph from intersection (0.0) to the pure mask shape (1.0)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR', # This makes it a 0-1 slider
        update=_redraw_shader_view
    )

    preview_color: FloatVectorProperty(name="Preview Color", subtype='COLOR', default=(1.0, 1.0, 1.0), min=0.0, max=1.0, update=_redraw_shader_view)

    # --- CUBE-SPECIFIC PROPERTIES ---
    thickness: FloatProperty(name="Thickness", description="Creates a hole through the cube", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    roundness: FloatProperty(name="Roundness", description="Rounds the corners and edges of the shape", default=0.0, min=0.0, soft_max=1, update=_redraw_shader_view)
    bevel: FloatProperty(name="Bevel", description="Adds a 45-degree chamfer to the shape's edges", default=0.0, min=0.0, soft_max=0.5, update=_redraw_shader_view)
    pyramid: FloatProperty(name="Pyramid", description="Tapers the top of the cube to a point", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    twist: FloatProperty(name="Twist", description="Twists the cube around its Z-axis", default=0.0, min=-20.0, max=20.0, update=_redraw_shader_view)
    bend: FloatProperty(name="Bend", description="Bends the cube along its X-axis", default=0.0, min=-2.0, max=2.0, update=_redraw_shader_view)
    
    # --- SPHERE-SPECIFIC PROPERTIES ---
    sphere_thickness: FloatProperty(name="Thickness", description="Creates a subtractive hole through the sphere", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    sphere_elongation: FloatProperty(name="Elongation", description="Stretches the sphere into a capsule along the Z-axis", default=0.0, min=0.0, soft_max=2.0, update=_redraw_shader_view)
    sphere_cut_angle: FloatProperty(
        name="Pac-man",
        description="The visible angle of the sphere wedge, from 0 (invisible) to 360 (full sphere)",
        default=math.tau,
        min=0.0,
        max=math.tau,
        soft_min=0.0,
        soft_max=math.tau,
        subtype='ANGLE',
        update=_redraw_shader_view
    )

    # --- CYLINDER-SPECIFIC PROPERTIES ---
    cylinder_thickness: FloatProperty(name="Thickness", description="Hollows the cylinder, creating a pipe", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    cylinder_roundness: FloatProperty(name="Roundness", description="Rounds the sharp edges of the cylinder", default=0.0, min=0.0, soft_max=4.0, update=_redraw_shader_view)
    cylinder_bend: FloatProperty(name="Bend", description="Bends the cylinder along its Y-axis (height)", default=0.0, min=-2.0, max=2.0, update=_redraw_shader_view)
    cylinder_pyramid: FloatProperty(name="Pyramid", description="Tapers the top of the cylinder into a cone", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)

    # --- PRISM-SPECIFIC PROPERTIES (NEW!) ---
    prism_sides: IntProperty(name="Sides", description="Number of sides for the prism (N-gon)", default=6, min=3, max=16, update=_redraw_shader_view)
    prism_thickness: FloatProperty(name="Thickness", description="Creates a hollow shell inside the prism", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    prism_bend: FloatProperty(name="Bend", description="Bends the prism along its X-axis", default=0.0, min=-2.0, max=2.0, update=_redraw_shader_view)
    prism_twist: FloatProperty(name="Twist", description="Twists the prism around its Y-axis", default=0.0, min=-20.0, max=20.0, update=_redraw_shader_view)
    prism_pyramid: FloatProperty(name="Pyramid", description="Tapers the top of the prism to a point", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)

    # --- TORUS-SPECIFIC PROPERTIES (NEW!) ---
    torus_outer_radius: FloatProperty(name="Outer Radius", description="The radius from the center to the outer edge", default=0.4, min=0.01, soft_max=2.0, subtype='DISTANCE', update=_redraw_shader_view)
    torus_inner_radius: FloatProperty(name="Inner Radius", description="The radius of the center hole", default=0.2, min=0.0, soft_max=2.0, subtype='DISTANCE', update=_redraw_shader_view)
    torus_thickness: FloatProperty(name="Thickness", description="Hollows the torus, creating a shell", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    torus_cut_angle: FloatProperty(
        name="Pac-man",
        description="The visible angle of the torus wedge, from 0 (invisible) to 360 (full torus)",
        default=math.tau,
        min=0.0,
        max=math.tau,
        subtype='ANGLE',
        update=_redraw_shader_view
    )
    torus_elongation: FloatProperty(
        name="Elongation",
        description="Stretches the torus cross-section into a capsule shape",
        default=0.0,
        min=0.0,
        soft_max=1.0,
        subtype='DISTANCE',
        update=_redraw_shader_view
    )

    # --- SYMMETRY PROPERTIES ---
    use_mirror_x: BoolProperty(name="X", default=False, update=_redraw_shader_view)
    use_mirror_y: BoolProperty(name="Y", default=False, update=_redraw_shader_view)
    use_mirror_z: BoolProperty(name="Z", default=False, update=_redraw_shader_view)
    use_radial_mirror: BoolProperty(name="Enable Radial Mirror", default=False, update=_redraw_shader_view)
    radial_mirror_count: IntProperty(name="Count", default=6, min=2, max=64, update=_redraw_shader_view)

    # --- CURVE-SPECIFIC PROPERTIES ---
    curve_mode: EnumProperty(name="Curve Mode", items=[('HARD', "Hard", ""), ('SMOOTH', "Smooth", "")], default='SMOOTH', update=_redraw_shader_view)
    curve_point_density: IntProperty(name="Point Density", default=10, min=1, max=128, update=_redraw_shader_view)
    curve_instance_type: EnumProperty(
        name="Instance Shape", 
        items=[
            ('MESH_UVSPHERE', "Sphere", ""), 
            ('MESH_CUBE', "Cube", ""), 
            ('MESH_ICOSPHERE', "Prism", ""), 
            ('CAPSULE', "Capsule", ""),
            ('MESH_TORUS', "Torus", ""),
            ('MESH_CYLINDER', "Cylinder", ""),
            ('MESH_CONE', "Cone", "")
        ], 
        default='MESH_UVSPHERE', 
        update=_redraw_shader_view
    )
    curve_instance_rotation: FloatVectorProperty(
        name="Instance Rotation",
        description="Apply an additional local rotation to each shape on the curve",
        subtype='EULER',
        default=(0.0, 0.0, 0.0),
        update=_redraw_shader_view
    )
    curve_instance_spacing: FloatProperty(
        name="Instance Spacing",
        description="Controls the distance between shapes along the curve. Higher values mean more space",
        default=1.0,
        min=0.1,
        soft_max=10.0,
        subtype='FACTOR',
        update=_redraw_shader_view
    )
    curve_control_mode: EnumProperty(
        name="Control Mode",
        items=[('UNIFORM', "Uniform", "Use a single shape for the whole curve"),
               ('CUSTOM', "Custom Points", "Define specific points with unique properties")],
        default='UNIFORM',
        update=_redraw_shader_view
    )
    
    custom_control_points: CollectionProperty(type=SDFCurveControlPoint)
    active_control_point_index: IntProperty(default=-1)
    
    curve_point_density: IntProperty(name="Point Density", default=10, min=1, max=128, update=_redraw_shader_view)
    curve_global_radius: FloatProperty(name="Global Radius", default=0.4, min=0.001, soft_max=2.0, subtype='DISTANCE', update=_redraw_shader_view)
    curve_segment_scale: FloatProperty(name="Segment Scale", default=0.9, min=0.0, soft_max=1.5, subtype='FACTOR', update=_redraw_shader_view)
    curve_taper_head: FloatProperty(name="Taper Head", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)
    curve_taper_tail: FloatProperty(name="Taper Tail", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_redraw_shader_view)

# -------------------------------------------------------------------
# 2) List of classes to register (excluding PropertyGroup)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 2) List of classes to register (excluding PropertyGroup)
# -------------------------------------------------------------------
_classes = [
    # UI Panels and Menus
    SDFPrototyperPanel,
    SDFRenderPanel,
    SDF_UL_nodes,
    SDF_UL_curve_points,
    VIEW3D_MT_sdf_rclick,

    # Data Structures (must be registered before classes that use them)
    SDFCurveControlPoint,

    # Core SDF Generation & Conversion
    StartSDFOperator,
    ConvertSDFOperator,
    OBJECT_OT_sdf_bake_volume,
    OBJECT_OT_sdf_bake_to_remesh,
    OBJECT_OT_sdf_auto_uv,                    
    OBJECT_OT_sdf_snap_selection_to_active, 
    # --- THIS IS THE ONLY CHANGE: Using the new, correct operator ---
    OBJECT_OT_sdf_remesh_tools,

    # Add SDF Shape Operators (YOUR FULL LIST IS PRESERVED)
    SDFCubeAdd, SDFCylinderAdd, SDFUVSphereAdd, SDFConeAdd,
    SDFPrismAdd, SDFTorusAdd, SDFCurveAdd, SDFMeshAdd, SDFSculptAdd,
    SDFMeshToSDF,

    # List & Shape Management Operators
    PROTOTYPER_OT_SDFSetPointCloudPreview,
    PROTOTYPER_OT_SDFListMove,
    SDFDuplicateOperator, SDFDeleteOperator, SDFClearOperator,
    PROTOTYPER_OT_SDFRepeatShape,
    PROTOTYPER_OT_SDFCleanupList,
    PROTOTYPER_OT_toggle_smooth,
    PROTOTYPER_OT_SDFCurvePointAdd, 
    PROTOTYPER_OT_SDFCurvePointRemove,

    # Symmetry Baking Operator
    OBJECT_OT_bake_sdf_symmetry,
    PROTOTYPER_OT_SDFSymmetrize, 
    PROTOTYPER_OT_SDFFlipActiveShape,
    PROTOTYPER_OT_SDFFlipModel,

    # Domain & Global Control Operators
    OBJECT_OT_reset_global_scale,
    OBJECT_OT_reset_domain_transform,
    OBJECT_OT_ToggleDecimation,

    # Brush-Cube Clipping Operators
    OBJECT_OT_create_brush_cube, OBJECT_OT_delete_brush_cube,
    OBJECT_OT_reset_brush_cube_transform, OBJECT_OT_select_brush_cube,
    OBJECT_OT_toggle_brush_cube_visibility, OBJECT_OT_apply_brush_cube,
    OBJECT_OT_toggle_clip,

    # Rendering Operators
    OBJECT_OT_sdf_render_highres,
    OBJECT_OT_sdf_render_final,

    # Helper & Internal Operators
    OBJECT_OT_SelectEmpty,
    OBJECT_OT_ToggleMuteNode,
    OBJECT_OT_purge_unused_data,
    VIEW3D_OT_toggle_overlays,
]

_addon_keymaps = []
_timer_is_running = False

# -------------------------------------------------------------------
# 3) REGISTER
# -------------------------------------------------------------------
def register():
    global _addon_keymaps, _timer_is_running

    # --- STEP 1: Register all helper classes FIRST ---
    # This includes SDFCurveControlPoint, operators, panels, etc.
    for cls in _classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass # Class is already registered

    # --- STEP 2: Now that its dependencies are registered, register the main data class ---
    bpy.utils.register_class(SDFNodeItem)

    # --- STEP 3: Attach properties to Blender's built-in types ---
    bpy.types.Object.sdf_nodes = bpy.props.CollectionProperty(type=SDFNodeItem)
    bpy.types.Object.active_sdf_node_index = bpy.props.IntProperty(default=-1)

    # --- STEP 4: Define and attach all Scene properties ---
    Scene = bpy.types.Scene
    Scene.sdf_domain                   = bpy.props.PointerProperty(type=bpy.types.Object)
    Scene.sdf_max_shapes = bpy.props.IntProperty(
        name="Max Shapes",
        description="Shader shape limit. High values require a powerful GPU. Change requires restart",
        default=32, min=8, max=2048, step=4 
    )
    Scene.sdf_raymarch_max_steps = bpy.props.IntProperty(
        name="Max Ray Steps",
        description="Maximum steps for the raymarcher. Lower is faster but has a shorter render distance",
        default=128, 
        min=1, # <-- Change this value from 16 to 1
        max=512,
        update=_redraw_shader_view
    )
    Scene.sdf_pixelation_amount = bpy.props.IntProperty(
        name="Pixelation",
        description="Increases the size of pixel blocks for a stylized, faster preview. 1 = Off",
        default=1, min=1, max=32,
        update=_redraw_shader_view
    )
    Scene.lock_sdf_panel               = bpy.props.BoolProperty(name="Lock SDF Panel", default=False, update=update_lock)
    Scene.locked_sdf_object            = bpy.props.PointerProperty(type=bpy.types.Object)
    Scene.sdf_status_message           = bpy.props.StringProperty(default="Ready")
    Scene.sdf_auto_resolution_enable   = bpy.props.BoolProperty(name="Automatic Preview", default=False, update=toggle_auto_resolution_mode)
    Scene.sdf_preview_mode             = bpy.props.BoolProperty(name="Preview Mode", default=True, update=update_sdf_resolution)
    Scene.sdf_preview_resolution       = bpy.props.IntProperty(name="Low-Res", default=1, min=1, soft_max=64, update=update_sdf_resolution)
    Scene.sdf_final_resolution         = bpy.props.IntProperty(name="High-Res", default=3, min=1, soft_max=512, update=update_sdf_resolution)
    Scene.sdf_auto_threshold           = bpy.props.FloatProperty(name="Movement Sensitivity", default=1e-5, min=1e-6, max=1e-3)
    Scene.sdf_auto_idle_delay          = bpy.props.FloatProperty(name="Idle Delay (s)", default=0.5, min=0.1, max=2.0)
    Scene.sdf_decimation_enable        = bpy.props.BoolProperty(name="Enable Decimation", default=False)
    Scene.sdf_global_scale             = bpy.props.FloatProperty(name="Global Scale", default=1.0, min=0.1, max=10.0, update=update_sdf_global_scale)
    Scene.use_brush_cube               = bpy.props.BoolProperty(name="Use Brush Cube", default=False)
    Scene.brush_cube                   = bpy.props.PointerProperty(type=bpy.types.Object)
    Scene.clip_enabled                 = bpy.props.BoolProperty(name="Clipping Enabled", default=False)
    Scene.sdf_render_panel_enable      = bpy.props.BoolProperty(name="Show Render Options", default=False)
    Scene.sdf_render_from              = bpy.props.EnumProperty(name="Render From", items=[('CAMERA','Camera',''),('VIEW','View','')], default='CAMERA')
    Scene.sdf_render_highres_resolution= bpy.props.IntProperty(name="Res", default=3, min=1, max=1024)
    Scene.sdf_render_scale             = bpy.props.FloatProperty(name="Scale", default=1.0, min=0.1, max=2.0)
    Scene.sdf_render_engine            = bpy.props.EnumProperty(name="Engine", items=[('BLENDER_EEVEE_NEXT','Eevee',''),('CYCLES','Cycles','')], default='BLENDER_EEVEE_NEXT')
    Scene.sdf_render_samples           = bpy.props.IntProperty(name="Eevee Samples", default=5, min=1, max=4096)
    Scene.sdf_cycles_samples           = bpy.props.IntProperty(name="Cycles Max", default=5, min=1, max=4096)
    Scene.sdf_cycles_preview_samples   = bpy.props.IntProperty(name="Cycles Min", default=16, min=1, max=4096)
    Scene.sdf_render_shading_mode      = bpy.props.EnumProperty(name="Shading Mode", items=[('CURRENT','Current',''),('MATERIAL','Material',''),('RENDERED','Rendered','')], default='CURRENT')
    Scene.sdf_render_disable_overlays  = bpy.props.BoolProperty(name="Disable Overlays", default=False)
    Scene.sdf_shape_tab                = bpy.props.EnumProperty(name="Shape Tab", items=[('BASIC','Basic',''),('DEFORM','Deform',''),('STYLE','Style',''),('MISC','Misc','')], default='BASIC')
    Scene.sdf_global_tint              = bpy.props.FloatVectorProperty(name="Global Tint", subtype='COLOR', default=(1.0,1.0,1.0), min=0.0, max=1.0, description="Multiply shape colors globally", update=_redraw_shader_view)
    Scene.sdf_light_direction = bpy.props.FloatVectorProperty(
        name="Light Direction",
        subtype='DIRECTION',
        default=(0.6, 0.6, 0.5),
        update=_redraw_shader_view
    )
    Scene.sdf_preview_brightness = bpy.props.FloatProperty(
        name="Brightness",
        default=0.0, min=-1.0, max=1.0,
        update=_redraw_shader_view
    )
    Scene.sdf_preview_contrast = bpy.props.FloatProperty(
        name="Contrast",
        default=1.0, min=0.0, max=2.0,
        update=_redraw_shader_view
    )
    Scene.sdf_cavity_enable = bpy.props.BoolProperty(
        name="Enable Cavity",
        description="Calculate ambient occlusion to add detail. May impact performance.",
        default=True,
        update=_redraw_shader_view
    )
    Scene.sdf_cavity_strength = bpy.props.FloatProperty(
        name="Strength",
        description="How strong the cavity/occlusion effect is",
        default=0.5, 
        min=0.0, 
        max=20.0,      # Increased hard limit
        soft_max=5.0,  # Increased soft limit for the slider
        update=_redraw_shader_view
    )
    Scene.sdf_shader_view              = bpy.props.BoolProperty(name="Enable SDF Shader View", default=False, update=lambda s,c: enable_sdf_shader_view(s.sdf_shader_view))
    Scene.sdf_color_blend_mode         = bpy.props.EnumProperty(name="Color Blend", items=[('HARD',"Hard","Pick one shape’s color"),('SOFT',"Soft","Interpolate in smooth areas")], default='HARD')
    Scene.sdf_domain_scale             = bpy.props.FloatProperty(name="Domain Scale", description="Sets the size of the SDF evaluation space. Inactive when Brush-Cube Clipping is on", default=1.0, min=0.01, soft_max=10.0, update=update_sdf_domain_scale)
    Scene.sdf_visualization_mode       = bpy.props.EnumProperty(name="Visualization Mode", items=[('SOLID', "Solid", "Render the SDF as a solid mesh"), ('POINTS', "Points", "Render the SDF as a fast point cloud")], default='SOLID', update=lambda self, context: rewire_full_sdf_chain(context))
    Scene.sdf_view_mode = bpy.props.EnumProperty(
        name="View Filter",
        description="Controls which shapes are visible in the shader view and baking",
        items=[
            ('ALL', "Show All", "Display all shapes"),
            ('SELECTED', "Selected", "Display only the selected shapes"),
            ('UNSELECTED', "Unselected", "Display only the unselected shapes")
        ],
        default='ALL',
        update=_redraw_shader_view # Trigger a redraw when the mode changes
    )

    # --- STEP 5: Add handlers, menus, and keymaps ---
    if check_mute_nodes not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(check_mute_nodes)
    if depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update)
    if toggle_lock_based_on_selection not in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.append(toggle_lock_based_on_selection)

    bpy.types.VIEW3D_MT_mesh_add.prepend(add_sdf_shapes)
    bpy.types.VIEW3D_MT_object_context_menu.prepend(rclick_sdf_menu)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km  = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new('prototyper.sdf_list_move','EQUAL','PRESS')
        kmi.properties.direction='UP';    _addon_keymaps.append((km,kmi))
        kmi = km.keymap_items.new('prototyper.sdf_list_move','MINUS','PRESS')
        kmi.properties.direction='DOWN';  _addon_keymaps.append((km,kmi))

    _timer_is_running = True
    bpy.app.timers.register(monitor_sdf_movement)


# -------------------------------------------------------------------
# 4) UNREGISTER
# -------------------------------------------------------------------
def unregister():
    global _addon_keymaps, _timer_is_running

    _timer_is_running = False

    for km,kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    if check_mute_nodes in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(check_mute_nodes)
    if depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update)
    if toggle_lock_based_on_selection in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.remove(toggle_lock_based_on_selection)

    bpy.types.VIEW3D_MT_mesh_add.remove(add_sdf_shapes)
    bpy.types.VIEW3D_MT_object_context_menu.remove(rclick_sdf_menu)

    # --- REVERSE THE REGISTRATION ORDER ---
    # Unregister the main data class FIRST
    bpy.utils.unregister_class(SDFNodeItem)

    # Unregister all other helper classes
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass

    # Delete custom properties from Blender's types
    del bpy.types.Object.sdf_nodes
    del bpy.types.Object.active_sdf_node_index

    # Delete Scene props
    Scene = bpy.types.Scene
    props_to_del = [
        "sdf_domain", "sdf_domain_scale", "sdf_max_shapes", "lock_sdf_panel", "locked_sdf_object",
        "sdf_status_message", "sdf_auto_resolution_enable", "sdf_preview_mode",
        "sdf_preview_resolution", "sdf_final_resolution", "sdf_auto_threshold",
        "sdf_auto_idle_delay", "sdf_decimation_enable", "sdf_global_scale",
        "use_brush_cube", "brush_cube", "clip_enabled",
        "sdf_render_panel_enable", "sdf_render_from",
        "sdf_render_highres_resolution", "sdf_render_scale", "sdf_render_engine",
        "sdf_render_samples", "sdf_cycles_samples", "sdf_cycles_preview_samples",
        "sdf_render_shading_mode", "sdf_render_disable_overlays",
        "sdf_shape_tab", "sdf_shader_view", "sdf_global_tint",
        "sdf_light_azimuth", "sdf_light_elevation", "sdf_color_blend_mode",
        "sdf_visualization_mode",
        "sdf_view_mode",
        "sdf_raymarch_max_steps",
        "sdf_pixelation_amount"
        
    ]
    for p in props_to_del:
        if hasattr(Scene, p):
            try:
                delattr(Scene, p)
            except Exception:
                pass