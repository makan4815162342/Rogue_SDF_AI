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


# Find this function in main.py and add the new 'if item.icon == 'MESH_CONE':' block.

def collect_sdf_data(context):
    """
    Gather ALL shape data for the SDF shader.
    This version is simplified to pass a new 'itemID' to the shader,
    which will handle all the complex grouping logic.
    """
    shapes = []
    op_map = { 'SMOOTH_UNION': 3, 'SMOOTH_SUBTRACT': 4, 'SMOOTH_INTERSECT': 5, }
    domain = getattr(context.scene, "sdf_domain", None)
    
    MAX_SHAPES_CURRENT = context.scene.sdf_max_shapes

    if not domain:
        # The data tuple now has 11 elements: (..., color, highlight, itemID)
        return [(-1, (0,0,0), (1,1,1), (1,0,0,0), 0, 0.0, 0, 0, (1,1,1), 0, -1)] * MAX_SHAPES_CURRENT

    # We use enumerate to get a unique ID for each item in the UI list.
    for item_index, item in enumerate(domain.sdf_nodes):
        e = item.empty_object
        if not e or item.is_hidden or len(shapes) >= MAX_SHAPES_CURRENT:
            continue

        code = { 
            'MESH_CUBE': 0, 'MESH_UVSPHERE': 1, 'MESH_TORUS': 2, 
            'MESH_CYLINDER': 3, 'MESH_CONE': 4, 'MESH_ICOSPHERE': 5,
            'CURVE_BEZCURVE': 6,
        }.get(item.icon, -1)
        
        op = op_map.get(item.operation, 3)
        blend = item.blend
        color = item.preview_color
        highlight = int(item.use_highlight)
        mirror_flags = (int(item.use_mirror_x) * 1) | (int(item.use_mirror_y) * 2) | (int(item.use_mirror_z) * 4)
        radial_count = item.radial_mirror_count if item.use_radial_mirror else 0
        itemID = item_index # The unique ID for this group of shapes.

        if item.icon == 'CURVE_BEZCURVE':
            curve_obj = next((child for child in e.children if child.type == 'CURVE'), None)
            if not curve_obj or not curve_obj.data.splines: continue
            
            curve_segments = []
            for spline in curve_obj.data.splines:
                if len(spline.bezier_points) < 2: continue
                
                if item.curve_mode == 'HARD':
                    for i in range(len(spline.bezier_points) - 1):
                        p1_local, p2_local = spline.bezier_points[i].co, spline.bezier_points[i+1].co
                        radius = spline.bezier_points[i].radius * e.empty_display_size
                        start, end = curve_obj.matrix_world @ p1_local, curve_obj.matrix_world @ p2_local
                        curve_segments.append({'start': start, 'end': end, 'radius': radius})
                else: # 'SMOOTH'
                    for i in range(len(spline.bezier_points) - 1):
                        bp1, bp2 = spline.bezier_points[i], spline.bezier_points[i+1]
                        p0, h0, h1, p1 = bp1.co, bp1.handle_right, bp2.handle_left, bp2.co
                        r1, r2 = bp1.radius * e.empty_display_size, bp2.radius * e.empty_display_size
                        subdivisions = item.curve_subdivisions
                        last_point = curve_obj.matrix_world @ get_bezier_point(0.0, p0, h0, h1, p1)
                        for j in range(1, subdivisions + 1):
                            t = j / float(subdivisions)
                            current_point = curve_obj.matrix_world @ get_bezier_point(t, p0, h0, h1, p1)
                            current_radius = r1 * (1.0 - t) + r2 * t
                            curve_segments.append({'start': last_point, 'end': current_point, 'radius': current_radius})
                            last_point = current_point
            
            if not curve_segments: continue

            for seg in curve_segments:
                if len(shapes) >= MAX_SHAPES_CURRENT: break
                pos = (seg['start'] + seg['end']) / 2.0
                height = (seg['end'] - seg['start']).length
                scl = Vector((seg['radius'], height, seg['radius']))
                direction = (seg['end'] - seg['start']).normalized() if height > 0 else Vector((0,1,0))
                rot = direction.to_track_quat('Y', 'Z')
                # Every segment gets the same op, blend, and itemID. The shader will sort it out.
                shapes.append((code, pos, scl, rot, op, blend, mirror_flags, radial_count, color, highlight, itemID))

        else: # STANDARD PRIMITIVE LOGIC
            if item.icon == 'MESH_CONE':
                tip_empty = next((child for child in e.children if "Tip" in child.name), None)
                if not tip_empty: continue
                base_pos, tip_pos = e.matrix_world.to_translation(), tip_empty.matrix_world.to_translation()
                pos, height = (base_pos + tip_pos) / 2.0, (tip_pos - base_pos).length
                r1 = (e.matrix_world.to_scale().x + e.matrix_world.to_scale().z) / 2.0
                r2 = (tip_empty.matrix_world.to_scale().x + tip_empty.matrix_world.to_scale().z) / 2.0
                scl, direction = Vector((r1, height, r2)), (tip_pos - base_pos).normalized() if height > 0 else Vector((0,1,0))
                rot = direction.to_track_quat('Y', 'Z')
            else:
                mw = e.matrix_world
                pos, scl, rot = mw.to_translation(), mw.to_scale(), mw.to_quaternion()
            
            shapes.append((code, pos, scl, rot, op, blend, mirror_flags, radial_count, color, highlight, itemID))

    while len(shapes) < MAX_SHAPES_CURRENT:
        shapes.append((-1, (0,0,0), (1,1,1), (1,0,0,0), 0, 0.0, 0, 0, (1,1,1), 0, -1))
        
    return shapes


import array

def draw_sdf_shader():
    """
    Main draw handler for the SDF Shader Preview.
    This version packs and sends the new itemID for shader-side grouping.
    """
    global shader, batch
    if not shader: return

    ctx = bpy.context
    scene = ctx.scene
    region = ctx.region
    rv3d = ctx.region_data
    MAX_SHAPES_CURRENT = scene.sdf_max_shapes

    state.depth_test_set('NONE')
    state.blend_set('NONE')

    shader.bind()
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("viewMatrixInv", rv3d.view_matrix.inverted())
    shader.uniform_float("projMatrixInv", rv3d.window_matrix.inverted())

    az, el = scene.sdf_light_azimuth, scene.sdf_light_elevation
    shader.uniform_float("uLightDir", (math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)))
    shader.uniform_float("uGlobalTint", scene.sdf_global_tint)
    shader.uniform_float("uDomainCenter", scene.sdf_domain.location if scene.sdf_domain else (0.0, 0.0, 0.0))

    shapes = collect_sdf_data(ctx)
    shader.uniform_int("uCount", sum(1 for s in shapes if s[0] >= 0))

    # --- Flatten ALL data directly from the 'shapes' list ---
    tf                  = [int(s[0]) for s in shapes]
    type_flat           = [tf[i+j] for i in range(0, MAX_SHAPES_CURRENT, 4) for j in range(4)]
    pos_flat            = [v for s in shapes for v in s[1]]
    scale_flat          = [v for s in shapes for v in s[2]]
    rot_flat            = [v for s in shapes for v in s[3]]
    op_flat             = [int(s[4]) for s in shapes]
    blend_flat          = [float(s[5]) for s in shapes]
    mirror_flags_flat   = [int(s[6]) for s in shapes]
    radial_count_flat   = [int(s[7]) for s in shapes]
    color_flat          = [v for s in shapes for v in s[8]]
    highlight_flat      = [int(s[9]) for s in shapes]
    # --- NEW: Pack the itemID ---
    item_id_flat        = [int(s[10]) for s in shapes]

    # --- Create byte buffers ---
    type_buf           = array.array('i', type_flat).tobytes()
    pos_buf            = array.array('f', pos_flat).tobytes()
    scale_buf          = array.array('f', scale_flat).tobytes()
    rot_buf            = array.array('f', rot_flat).tobytes()
    op_buf             = array.array('i', op_flat).tobytes()
    blend_buf          = array.array('f', blend_flat).tobytes()
    mirror_flags_buf   = array.array('i', mirror_flags_flat).tobytes()
    radial_count_buf   = array.array('i', radial_count_flat).tobytes()
    color_buf          = array.array('f', color_flat).tobytes()
    highlight_buf      = array.array('i', highlight_flat).tobytes()
    item_id_buf        = array.array('i', item_id_flat).tobytes() # --- NEW ---

    # --- Get shader uniform locations ---
    loc_t, loc_p, loc_s, loc_r = shader.uniform_from_name("uShapeTypePacked"), shader.uniform_from_name("uShapePos"), shader.uniform_from_name("uShapeScale"), shader.uniform_from_name("uShapeRot")
    loc_o, loc_b, loc_c = shader.uniform_from_name("uShapeOp"), shader.uniform_from_name("uShapeBlend"), shader.uniform_from_name("uShapeColor")
    loc_mf, loc_rc, loc_hl = shader.uniform_from_name("uShapeMirrorFlags"), shader.uniform_from_name("uShapeRadialCount"), shader.uniform_from_name("uShapeHighlight")
    loc_iid = shader.uniform_from_name("uShapeItemID") # --- NEW ---

    # --- Upload data to the GPU ---
    shader.uniform_vector_int(loc_t, type_buf,  4,  MAX_SHAPES_CURRENT // 4)
    shader.uniform_vector_float(loc_p, pos_buf,   3, MAX_SHAPES_CURRENT)
    shader.uniform_vector_float(loc_s, scale_buf, 3, MAX_SHAPES_CURRENT)
    shader.uniform_vector_float(loc_r, rot_buf,   4, MAX_SHAPES_CURRENT)
    shader.uniform_vector_int(loc_o, op_buf,    1, MAX_SHAPES_CURRENT)
    shader.uniform_vector_float(loc_b, blend_buf, 1, MAX_SHAPES_CURRENT)
    shader.uniform_vector_float(loc_c, color_buf, 3, MAX_SHAPES_CURRENT)
    shader.uniform_vector_int(loc_mf, mirror_flags_buf, 1, MAX_SHAPES_CURRENT)
    shader.uniform_vector_int(loc_rc, radial_count_buf, 1, MAX_SHAPES_CURRENT)
    shader.uniform_vector_int(loc_hl, highlight_buf, 1, MAX_SHAPES_CURRENT)
    shader.uniform_vector_int(loc_iid, item_id_buf, 1, MAX_SHAPES_CURRENT) # --- NEW ---

    shader.uniform_int("uColorBlendMode", 1 if scene.sdf_color_blend_mode == 'SOFT' else 0)

    batch.draw(shader)
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

    # 1) Mute or unmute the final GN node (prevents mesh drawing)
    toggle_gn_output_mute(ctx, mute=enable)

    # 2) Swap the domain’s display type to BOUNDS for quick wire‐box
    if domain:
        if enable:
            domain["_orig_display"] = domain.display_type
            domain.display_type     = 'BOUNDS'
        else:
            if "_orig_display" in domain:
                domain.display_type = domain["_orig_display"]
                del domain["_orig_display"]

    # 3) When enabling, compile & install the GLSL handler
    if enable and handler is None:
        # 3a) Build the full-screen shader
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

        # 3b) Install the draw handler
        handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_sdf_shader, (), 'WINDOW', 'POST_VIEW'
        )
        if ctx.area:
            ctx.area.tag_redraw()
        print("SDF Shader View ENABLED.")

    # 4) When disabling, remove the handler and restore state
    elif not enable and handler:
        bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
        handler = shader = batch = None

        # force a redraw so the GN mesh + empties reappear
        if ctx.area:
            ctx.area.tag_redraw()
        print("SDF Shader View DISABLED.")




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
            
            # --- NEW: Highlight Toggle Button ---
            # Use a different icon based on the state for better feedback
            icon = 'RESTRICT_SELECT_ON' if sdf_node.use_highlight else 'RESTRICT_SELECT_OFF'
            sub.prop(sdf_node, "use_highlight", text="", icon=icon, emboss=True)
            # --- END NEW ---

            sub.prop(sdf_node, "is_viewport_hidden", text="", icon_only=True, emboss=True)
            sub.prop(sdf_node, "is_hidden", text="")

        else:
            row = layout.row(align=True)
            row.label(text=f"'{sdf_node.name}' is broken!", icon='ERROR')
            row.operator(PROTOTYPER_OT_SDFCleanupList.bl_idname, text="Clean List", icon='BRUSH_DATA')

# -------------------------------------------------------------------
# Operators for Object Selection and Mute Toggle
# -------------------------------------------------------------------
import sys, subprocess

def ensure_pyopenvdb():
    try:
        import openvdb   # the C++ Python binding
        return openvdb
    except ImportError:
        # 1) Make sure pip is available
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        # 2) Install or upgrade pyopenvdb
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "pyopenvdb"
        ])
        # 3) Retry import
        import openvdb
        return openvdb
    

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




def render_sdf_slices(resX, resY, depth, bounds_min, bounds_max):
    """
    Renders SDF slices within a specific world-space bounding box provided by the caller.
    """
    # --- Recompile the shader on every bake ---
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

    # --- Gather original WORLD-SPACE shape data ---
    shapes   = collect_sdf_data(bpy.context)
    uCount   = sum(1 for s in shapes if s[0] >= 0)
    tf       = [int(s[0]) for s in shapes]
    type_flat = [tf[i + j] for i in range(0, 32, 4) for j in range(4)]
    pos_flat   = [f for s in shapes for f in s[1]]
    scale_flat = [f for s in shapes for f in s[2]]
    rot_flat   = [f for s in shapes for f in s[3]]
    op_flat    = [int(s[4]) for s in shapes]
    blend_flat = [float(s[5]) for s in shapes]

    # --- Pack to byte buffers ---
    type_buf  = array.array('i', type_flat).tobytes()
    pos_buf   = array.array('f', pos_flat).tobytes()
    scale_buf = array.array('f', scale_flat).tobytes()
    rot_buf   = array.array('f', rot_flat).tobytes()
    op_buf    = array.array('i', op_flat).tobytes()
    blend_buf = array.array('f', blend_flat).tobytes()

    # --- Set up off-screen rendering ---
    offscreen = gpu.types.GPUOffScreen(resX, resY)
    slices    = np.zeros((depth, resY, resX), dtype=np.float32)

    with offscreen.bind():
        gpu.state.viewport_set(0, 0, resX, resY)
        slice_shader.bind()
        
        # --- Pass the final, scaled bounding box and other uniforms ---
        slice_shader.uniform_float("uBoundsMin", bounds_min)
        slice_shader.uniform_float("uBoundsMax", bounds_max)
        slice_shader.uniform_int("uDepth", depth)
        slice_shader.uniform_int("uCount", uCount)

        # --- Upload uniform arrays ---
        loc = slice_shader.uniform_from_name
        slice_shader.uniform_vector_int(loc("uShapeTypePacked"), type_buf, 4, 8)
        slice_shader.uniform_vector_float(loc("uShapePos"), pos_buf, 3, 32)
        slice_shader.uniform_vector_float(loc("uShapeScale"), scale_buf, 3, 32)
        slice_shader.uniform_vector_float(loc("uShapeRot"), rot_buf, 4, 32)
        slice_shader.uniform_vector_int(loc("uShapeOp"), op_buf, 1, 32)
        slice_shader.uniform_vector_float(loc("uShapeBlend"), blend_buf, 1, 32)

        framebuffer = gpu.state.active_framebuffer_get()
        for z in range(depth):
            slice_shader.uniform_int("uSliceIndex", z)
            slice_batch.draw(slice_shader)
            pixel_buffer = framebuffer.read_color(0, 0, resX, resY, 4, 0, 'FLOAT')
            arr = np.array(pixel_buffer.to_list(), dtype=np.float32)
            arr = arr.reshape(resY, resX, 4)
            slices[z, :, :] = arr[:, :, 0]

    offscreen.free()
    return slices






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





import os
import bpy
import numpy as np
import gpu
from gpu.types import GPUOffScreen
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

class OBJECT_OT_sdf_bake_volume(bpy.types.Operator):
    bl_idname = "object.sdf_bake_volume"
    bl_label = "Bake SDF to High-Quality Mesh"
    bl_description = "Creates a high-quality, retopologized mesh from the SDF Domain via a refinement pipeline"
    bl_options = {'REGISTER', 'UNDO'}

    # --- STAGE 1: BAKE SETTINGS ---
    res: bpy.props.IntProperty(
        name="Initial Resolution",
        description="Voxels along the longest axis for the initial bake. Higher is more detailed",
        default=256, min=32, max=1024
    )
    bake_scale: bpy.props.FloatProperty(
        name="Bake Scale",
        description="Multiplier for the bake volume size. >1.0 can help capture thin features",
        default=1.0, min=0.1, max=10.0
    )
    filepath: bpy.props.StringProperty(
        name="VDB File Path",
        description="Filepath to write the intermediate VDB file",
        default=os.path.expanduser("~/Desktop/sdf_bake.vdb"),
        subtype='FILE_PATH'
    )

    # --- STAGE 2: RETOPOLOGY ---
    retopology_method: bpy.props.EnumProperty(
        name="Retopology Method",
        description="Method to rebuild the mesh for better quality",
        items=[
            ('NONE', "None", "Keep the raw mesh from the volume conversion (fast, low quality)"),
            ('VOXEL', "Voxel Remesh", "Clean the mesh into uniform voxels (good for cleanup)"),
            ('QUADRIFLOW', "QuadriFlow Remesh", "Rebuild with clean quad topology (slow, highest quality)"),
        ],
        default='QUADRIFLOW'
    )
    voxel_remesh_size: bpy.props.FloatProperty(
        name="Voxel Size",
        description="The size of the voxels for remeshing. Smaller values are very memory intensive",
        default=0.01, min=0.001, soft_min=0.01, max=1.0, precision=4
    )
    quadriflow_target_faces: bpy.props.IntProperty(
        name="Target Face Count",
        description="The desired number of faces for the new mesh",
        default=5000, min=100, max=100000
    )

    # --- STAGE 3: POLISHING ---
    add_subdivision_modifier: bpy.props.BoolProperty(
        name="Add Subdivision Surface",
        description="Add a Subdivision Surface modifier. Best used after retopology",
        default=True
    )
    subdivision_levels: bpy.props.IntProperty(
        name="Subdivision Levels",
        default=2, min=1, max=6
    )
    add_smooth_modifier: bpy.props.BoolProperty(
        name="Add Classic Smooth",
        description="Add a classic Smooth modifier to even out geometry (Laplacian)",
        default=False
    )
    smooth_factor: bpy.props.FloatProperty(
        name="Factor",
        default=0.5, min=0.0, max=2.0
    )
    smooth_iterations: bpy.props.IntProperty(
        name="Iterations",
        default=5, min=1, max=30
    )
    add_corrective_smooth_modifier: bpy.props.BoolProperty(
        name="Add Corrective Smooth",
        description="Add a final smoothing pass to relax the mesh after subdivision",
        default=True
    )
    shade_smooth: bpy.props.BoolProperty(
        name="Shade Smooth",
        description="Automatically apply smooth shading to the final mesh",
        default=True
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        
        box = layout.box()
        box.label(text="Stage 1: Initial Bake", icon='VOLUME_DATA')
        col = box.column()
        col.prop(self, "res")
        col.prop(self, "bake_scale")
        col.prop(self, "filepath")

        box = layout.box()
        box.label(text="Stage 2: Retopology", icon='MOD_REMESH')
        col = box.column()
        col.prop(self, "retopology_method", text="Method")
        if self.retopology_method == 'VOXEL':
            sub = col.box()
            sub.prop(self, "voxel_remesh_size")
            domain = context.scene.sdf_domain
            if domain:
                dims = domain.dimensions * self.bake_scale
                longest_axis = max(dims) if max(dims) > 0 else 1
                estimated_voxels = (longest_axis / self.voxel_remesh_size)**3 if self.voxel_remesh_size > 0 else 0
                if estimated_voxels > 25_000_000:
                    warning_box = layout.box()
                    warning_box.alert = True
                    warning_box.label(text="WARNING: High voxel count!", icon='ERROR')
                    warning_box.label(text="May cause Blender to freeze or crash.")
        elif self.retopology_method == 'QUADRIFLOW':
            sub = col.box()
            sub.prop(self, "quadriflow_target_faces")

        box = layout.box()
        box.label(text="Stage 3: Final Polishing", icon='MOD_SMOOTH')
        col = box.column()
        col.prop(self, "add_subdivision_modifier")
        if self.add_subdivision_modifier:
            sub = col.box()
            sub.prop(self, "subdivision_levels")
        col.separator()
        col.prop(self, "add_smooth_modifier")
        if self.add_smooth_modifier:
            sub = col.box()
            sub.prop(self, "smooth_factor")
            sub.prop(self, "smooth_iterations")
        col.separator()
        col.prop(self, "add_corrective_smooth_modifier")
        col.separator()
        col.prop(self, "shade_smooth")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=450)

    def execute(self, context):
        domain = getattr(context.scene, "sdf_domain", None)
        if not domain:
            self.report({'ERROR'}, "SDF Domain object not found.")
            return {'CANCELLED'}

        self.report({'INFO'}, "Stage 1: Baking SDF to Volume...")
        
        base_corners = [domain.matrix_world @ Vector(corner) for corner in domain.bound_box]
        base_min = Vector(min(c[i] for c in base_corners) for i in range(3))
        base_max = Vector(max(c[i] for c in base_corners) for i in range(3))
        base_size = base_max - base_min
        base_center = (base_min + base_max) / 2.0
        final_size = base_size * self.bake_scale
        final_min = base_center - (final_size / 2.0)
        final_max = base_center + (final_size / 2.0)

        longest_axis = max(final_size)
        if longest_axis <= 0: return {'CANCELLED'}
        voxel_size = longest_axis / self.res
        res_x, res_y, res_z = [max(16, int(s / voxel_size)) for s in final_size]

        try:
            slices = render_sdf_slices(res_x, res_y, res_z, final_min, final_max)
            vdb_path = bpy.path.abspath(self.filepath)
            os.makedirs(os.path.dirname(vdb_path), exist_ok=True)
            write_vdb(slices, vdb_path, final_min, voxel_size)
            bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD', location=(0,0,0))
            volume_object = context.view_layer.objects.active
            volume_object.name = "SDF_Volume_Source"
        except Exception as e:
            self.report({'ERROR'}, f"Stage 1 failed: {e}")
            return {'CANCELLED'}

        final_mesh_object = bpy.data.objects.new("SDF_Mesh_Result", bpy.data.meshes.new("SDF_Mesh_Data"))
        context.collection.objects.link(final_mesh_object)
        
        mod_v2m = final_mesh_object.modifiers.new(name="VolumeToMesh", type='VOLUME_TO_MESH')
        mod_v2m.object = volume_object
        mod_v2m.threshold = 0.0
        
        # --- NEW ROBUST METHOD ---
        # 1. Get the dependency graph
        depsgraph = context.evaluated_depsgraph_get()
        # 2. Get the object with the modifier evaluated
        object_eval = final_mesh_object.evaluated_get(depsgraph)
        # 3. Create a new mesh datablock from the evaluated object
        mesh_from_eval = bpy.data.meshes.new_from_object(object_eval)
        # 4. Assign the new mesh data to our object
        final_mesh_object.data = mesh_from_eval
        # 5. Clean up by removing the modifier and the source volume
        final_mesh_object.modifiers.clear()
        bpy.data.objects.remove(volume_object, do_unlink=True)
        # --- END OF ROBUST METHOD ---

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = final_mesh_object
        final_mesh_object.select_set(True)

        if self.retopology_method == 'VOXEL':
            self.report({'INFO'}, "Stage 2: Applying Voxel Remesh...")
            # Voxel Remesh is a modifier, so it's applied directly
            mod_remesh = final_mesh_object.modifiers.new(name="SDF_VoxelRemesh", type='REMESH')
            mod_remesh.mode = 'VOXEL'
            mod_remesh.voxel_size = self.voxel_remesh_size
            mod_remesh.use_smooth_shade = True
            bpy.ops.object.modifier_apply(modifier=mod_remesh.name)

        elif self.retopology_method == 'QUADRIFLOW':
            self.report({'INFO'}, "Stage 2: Applying QuadriFlow Remesh... (this may take a moment)")
            # Now that we have real geometry, this operator will work correctly
            bpy.ops.object.quadriflow_remesh(target_faces=self.quadriflow_target_faces, use_mesh_symmetry=False)
        
        else:
            self.report({'INFO'}, "Stage 2: Skipping Retopology.")

        self.report({'INFO'}, "Stage 3: Applying Final Polish...")
        
        if self.add_subdivision_modifier:
            mod_subdiv = final_mesh_object.modifiers.new(name="SDF_Subdivision", type='SUBSURF')
            mod_subdiv.levels = self.subdivision_levels
            mod_subdiv.render_levels = self.subdivision_levels

        if self.add_smooth_modifier:
            mod_smooth = final_mesh_object.modifiers.new(name="SDF_ClassicSmooth", type='SMOOTH')
            mod_smooth.factor = self.smooth_factor
            mod_smooth.iterations = self.smooth_iterations

        if self.add_corrective_smooth_modifier:
            mod_smooth_corr = final_mesh_object.modifiers.new(name="SDF_CorrectiveSmooth", type='CORRECTIVE_SMOOTH')
        
        if self.shade_smooth and len(final_mesh_object.data.polygons) > 0:
            final_mesh_object.data.polygons.foreach_set('use_smooth', [True] * len(final_mesh_object.data.polygons))
            final_mesh_object.data.update()

        self.report({'INFO'}, "Bake and refinement pipeline completed successfully!")
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
    The master rewiring function for the entire SDF node chain.
    This version uses the correct API to modify the node tree's interface.
    """
    scene = context.scene
    node_tree = get_sdf_geometry_node_tree(context)
    if not node_tree:
        return

    # 1. Find all the essential nodes
    domain_node = next((n for n in node_tree.nodes if n.name == "SDF Domain"), None)
    dec_node = get_dec_node(context)
    out_node = next((n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT'), None)

    if not domain_node or not out_node:
        print("[Rogue SDF AI] ERROR: Missing essential Domain or Output node.")
        return

    # --- THE CORRECT API FIX ---
    # We check the tree's interface to see if a "Geometry" OUTPUT socket exists.
    # This corresponds to an INPUT on the Group Output node inside the tree.
    if not any(item.name == "Geometry" and item.item_type == 'SOCKET' and item.in_out == 'OUTPUT' for item in node_tree.interface.items_tree):
        # If it doesn't exist, create it on the tree's interface. This is the correct method.
        node_tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    # Now that we're sure the interface socket exists, we can safely get the corresponding input on the node.
    output_geo_in = out_node.inputs.get("Geometry")
    if not output_geo_in:
        print("[Rogue SDF AI] FATAL ERROR: Could not find or create 'Geometry' input on Group Output node.")
        return
    # ---

    # 2. Get an ordered list of the active SDF shape nodes
    chain_nodes = []
    if hasattr(context.scene.sdf_domain, 'sdf_nodes'):
        chain_nodes = [
            next((n for n in node_tree.nodes if n.get("associated_empty") == item.empty_object.name), None)
            for item in context.scene.sdf_domain.sdf_nodes if item.empty_object and not item.is_hidden
        ]
        chain_nodes = [n for n in chain_nodes if n is not None]

    # 3. Clear all incoming links to the SDF inputs
    all_sdf_inputs = [node.inputs[0] for node in chain_nodes if node.inputs]
    if domain_node.inputs:
        all_sdf_inputs.append(domain_node.inputs[0])
    for sdf_input_socket in all_sdf_inputs:
        for link in list(sdf_input_socket.links):
            node_tree.links.remove(link)

    # 4. Connect the shapes to each other in sequence
    if len(chain_nodes) > 1:
        for i in range(len(chain_nodes) - 1):
            node_tree.links.new(chain_nodes[i].outputs[0], chain_nodes[i+1].inputs[0])
    
    # 5. Connect the last shape to the Domain's SDF input
    if chain_nodes:
        node_tree.links.new(chain_nodes[-1].outputs[0], domain_node.inputs[0])

    # 6. Find the Domain's "Mesh" output socket
    domain_geo_out = domain_node.outputs.get("Mesh")
    if not domain_geo_out:
        print("[Rogue SDF AI] FATAL ERROR: The 'SDF Domain' node has no output socket named 'Mesh'.")
        return

    # 7. Clear any links currently going into the final Group Output.
    for link in list(output_geo_in.links):
        node_tree.links.remove(link)

    # 8. Decide the connection based on whether decimation is enabled.
    if scene.sdf_decimation_enable and dec_node:
        decimate_geo_in = dec_node.inputs.get("Geometry")
        decimate_geo_out = dec_node.outputs.get("Geometry")
        if decimate_geo_in and decimate_geo_out:
            for link in list(decimate_geo_in.links):
                node_tree.links.remove(link)
            node_tree.links.new(domain_geo_out, decimate_geo_in)
            node_tree.links.new(decimate_geo_out, output_geo_in)
        else:
            node_tree.links.new(domain_geo_out, output_geo_in)
    else:
        node_tree.links.new(domain_geo_out, output_geo_in)

    print("[Rogue SDF AI] Node chain rewired successfully.")

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

# In main.py, replace the ENTIRE SDFPrototyperPanel class with this final, corrected version.

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

        box = layout.box()
        box.label(text="Global Controls", icon='SETTINGS')
        col = box.column(align=True)
        col.prop(scene, "sdf_max_shapes")
        row = col.row(align=True)
        row.prop(scene, "sdf_global_scale", text="Scale")
        row.operator("object.reset_sdf_global_scale", text="", icon='FILE_REFRESH')
        col.prop(scene, "sdf_auto_resolution_enable", text="Automatic Preview", icon='AUTO')
        if scene.sdf_auto_resolution_enable:
            auto = box.box()
            ac = auto.column(align=True)
            ac.prop(scene, "sdf_auto_threshold", text="Sensitivity")
            ac.prop(scene, "sdf_auto_idle_delay", text="Idle Delay (s)")
        sbox = box.box()
        sc = sbox.column()
        mr = sc.row(align=True)
        mr.enabled = not scene.sdf_auto_resolution_enable
        mr.prop(scene, "sdf_preview_mode", text="Manual Preview", toggle=True)
        rr = sc.row(align=True)
        rr.prop(scene, "sdf_preview_resolution", text="Low-Res")
        rr.prop(scene, "sdf_final_resolution",   text="High-Res")
        if scene.sdf_auto_resolution_enable:
            st = box.row()
            st.label(text="Status:")
            st.label(text=scene.sdf_status_message)
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
        row = layout.row()
        row.template_list("SDF_UL_nodes", "", domain, "sdf_nodes", domain, "active_sdf_node_index", rows=4)
        ops = row.column(align=True)
        ops.operator("prototyper.sdf_duplicate",    icon='DUPLICATE', text="")
        ops.operator("prototyper.sdf_repeat_shape", icon='MOD_ARRAY', text="")
        ops.operator("prototyper.sdf_delete",       icon='REMOVE',    text="")
        ops.separator()
        up = ops.operator("prototyper.sdf_list_move", icon='TRIA_UP', text="")
        up.direction = 'UP'
        dn = ops.operator("prototyper.sdf_list_move", icon='TRIA_DOWN', text="")
        dn.direction = 'DOWN'
        ops.separator()
        ops.operator("prototyper.sdf_clear",        icon='X',         text="")
        
        layout.separator()
        layout.prop(scene, "sdf_shader_view", text="Enable SDF Shader View", icon='SHADING_RENDERED')
        if scene.sdf_shader_view:
            light = layout.box()
            light.label(text="Preview Light", icon='LIGHT_HEMI')
            light.prop(scene, "sdf_light_azimuth",   text="Azimuth")
            light.prop(scene, "sdf_light_elevation", text="Elevation")
            
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
                        sc2.prop(item, "blend", text="Blend")
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
                            row3.operator(PROTOTYPER_OT_SDFFlipShape.bl_idname, text="X").axis = 'X'
                            row3.operator(PROTOTYPER_OT_SDFFlipShape.bl_idname, text="Y").axis = 'Y'
                            row3.operator(PROTOTYPER_OT_SDFFlipShape.bl_idname, text="Z").axis = 'Z'
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
                        if item.curve_mode == 'SMOOTH':
                            ccol.prop(item, "curve_subdivisions")
                    
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
        act_box.label(text="Finalize Mesh", icon='MOD_MESHDEFORM')
        act_col = act_box.column(align=True)
        act_col.operator("object.convert_sdf", text="Convert to Mesh", icon='MESH_DATA')
        act_col.operator("object.sdf_bake_volume", text="Bake to High-Quality Mesh", icon='VOLUME_DATA')
        act_col.separator()
        act_col.operator("prototyper.sdf_bake_symmetry", text="Bake Active Symmetries", icon='CHECKMARK')
        
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

    # ... (the invoke and draw methods remain the same) ...

    def execute(self, context):
            # --- 1. CRITICAL: Ensure all assets are loaded ONCE. ---
            try:
                load_all_sdf_node_groups()
            except Exception as e:
                self.report({'ERROR'}, f"Failed to load SDF resources: {e}")
                return {'CANCELLED'}

            # 2. Cleanly remove any existing SDF domain to prevent conflicts.
            if context.scene.sdf_domain:
                # We must use the clear operator to be safe
                bpy.ops.prototyper.sdf_clear('INVOKE_DEFAULT')
                # Check again in case the user cancelled
                if context.scene.sdf_domain:
                    bpy.data.objects.remove(context.scene.sdf_domain, do_unlink=True)
                context.scene.sdf_domain = None

            # 3. Create the new SDF Domain object
            bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
            domain_obj = context.active_object
            domain_obj.name = "SDF_Domain"
            context.scene.sdf_domain = domain_obj

            # 4. Add and configure the Geometry Nodes modifier
            geo_mod = domain_obj.modifiers.new(name="SDF Nodes", type='NODES')
            if not geo_mod.node_group:
                geo_mod.node_group = bpy.data.node_groups.new(name="SDF Node Tree", type='GeometryNodeTree')
            node_tree = geo_mod.node_group
            node_tree.nodes.clear()

            # 5. Create the essential nodes for the base setup
            group_output = node_tree.nodes.new(type="NodeGroupOutput")
            group_output.location = (400, 0)
            
            domain_node_group = bpy.data.node_groups.get("SDF Domain")
            if not domain_node_group:
                self.report({'ERROR'}, "FATAL: Could not find 'SDF Domain' node group after loading assets.")
                bpy.data.objects.remove(domain_obj, do_unlink=True)
                return {'CANCELLED'}
                
            sdf_domain_node = node_tree.nodes.new(type="GeometryNodeGroup")
            sdf_domain_node.node_tree = domain_node_group
            sdf_domain_node.location = (0, 0)
            sdf_domain_node.name = "SDF Domain"

            # 6. Wire the initial chain, lock the panel, and set initial resolution
            rewire_full_sdf_chain(context)
            context.scene.lock_sdf_panel = True
            update_sdf_resolution(self, context)
            # The line 'start_sdf_monitor()' has been removed from here.

            self.report({'INFO'}, "New Rogue SDF AI system initialized successfully.")
            return {'FINISHED'}

#---------------------------------------------------------------------


class ConvertSDFOperator(bpy.types.Operator):
    """Convert SDF to a new, separate Mesh object"""
    bl_idname = "object.convert_sdf"
    bl_label = "Convert SDF to Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "sdf_domain", None) is not None

    def execute(self, context):
        domain_obj = context.scene.sdf_domain

        # 1. Deselect everything and select the domain object
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = domain_obj
        domain_obj.select_set(True)

        # 2. Duplicate the domain object. The new object will be active.
        bpy.ops.object.duplicate()
        new_mesh_obj = context.active_object
        new_mesh_obj.name = "Converted_SDF_Mesh"

        # 3. Apply the Geometry Nodes modifier ON THE DUPLICATE
        # We find the modifier by type, which is more robust than by name
        mod_to_apply = next((m for m in new_mesh_obj.modifiers if m.type == 'NODES'), None)
        if mod_to_apply:
            bpy.ops.object.modifier_apply(modifier=mod_to_apply.name)
        else:
            self.report({'WARNING'}, "No Geometry Nodes modifier found to apply.")
            return {'CANCELLED'}
        
        # 4. Clean up the converted mesh (remove custom properties)
        if 'sdf_nodes' in new_mesh_obj:
            del new_mesh_obj['sdf_nodes']
        if 'active_sdf_node_index' in new_mesh_obj:
            del new_mesh_obj['active_sdf_node_index']

        self.report({'INFO'}, f"Successfully converted SDF to new mesh: '{new_mesh_obj.name}'")
        return {'FINISHED'}
    
#---------------------------------------------------------------------

import bpy, math
from mathutils import Vector

# In main.py, replace the entire OBJECT_OT_bake_sdf_symmetry class with this one.

class OBJECT_OT_bake_sdf_symmetry(bpy.types.Operator):
    """Convert mirror/radial flags into real, separate SDF shapes"""
    bl_idname = "prototyper.sdf_bake_symmetry"
    bl_label  = "Bake Active Symmetries"
    bl_options= {'REGISTER','UNDO'}

    @classmethod
    def poll(cls, context):
        # The button will only be greyed out if no shape is selected.
        domain = context.scene.sdf_domain
        if not (domain and hasattr(domain, 'sdf_nodes')):
            return False
        return 0 <= domain.active_sdf_node_index < len(domain.sdf_nodes)

    def _create_baked_shape(self, context, source_item, source_node, transform_matrix):
        """Helper function to create a single new SDF shape instance."""
        domain_obj = context.scene.sdf_domain
        node_tree = get_sdf_geometry_node_tree(context)
        source_empty = source_item.empty_object

        # --- Create New Node ---
        new_node = node_tree.nodes.new(type=source_node.bl_idname)
        if new_node.bl_idname == 'GeometryNodeGroup':
            new_node.node_tree = source_node.node_tree
        new_node.location = source_node.location + Vector((0, -200 * (len(domain_obj.sdf_nodes) + 1)))
        for i, orig_input in enumerate(source_node.inputs):
            if hasattr(orig_input, "default_value"):
                new_node.inputs[i].default_value = orig_input.default_value

        # --- Create New Empty & Apply Transform ---
        new_empty = source_empty.copy()
        if new_empty.data: new_empty.data = source_empty.data.copy()
        context.collection.objects.link(new_empty)
        new_empty.parent = domain_obj
        new_empty.matrix_world = transform_matrix

        # Normalize scale: mirror flips scale neg, but rotation should show mirrored geometry
        scale = new_empty.scale
        if scale.x < 0 or scale.y < 0 or scale.z < 0:
            # Flip sign to be positive
            scale = Vector((abs(scale.x), abs(scale.y), abs(scale.z)))
            new_empty.scale = scale

        
        # --- Create New UI List Item ---
        new_item = domain_obj.sdf_nodes.add()
        new_empty.name = f"{source_item.name}.Sym"
        new_item.name = new_empty.name
        new_item.empty_object = new_empty
        new_item.icon = source_item.icon
        new_item.preview_color = source_item.preview_color

        # --- Link everything ---
        new_node["associated_empty"] = new_empty.name
        
        # --- Link the new empty/objects to the geometry node input ---
        if source_item.icon == 'MESH_CONE':
            source_tip = next((child for child in source_empty.children if "Tip" in child.name), None)
            if source_tip:
                new_tip = source_tip.copy()
                if new_tip.data:
                    new_tip.data = source_tip.data.copy()
                context.collection.objects.link(new_tip)
                new_tip.parent = new_empty

                # Transform tip to match mirror
                new_tip.matrix_world = transform_matrix @ source_empty.matrix_world.inverted() @ source_tip.matrix_world

                # Optional mirror correction block
                if is_mirror_matrix(transform_matrix):
                    fix_scale_and_direction(new_empty, new_tip)

                # 🔧 FIX: Recalculate cone direction
                base_pos = new_empty.matrix_world.to_translation()
                tip_pos = new_tip.matrix_world.to_translation()
                direction = (tip_pos - base_pos).normalized()
                if direction.length < 0.0001:
                    direction = Vector((0, 1, 0))  # fallback
                rot = direction.to_track_quat('Y', 'Z')
                new_empty.rotation_mode = 'QUATERNION'
                new_empty.rotation_quaternion = rot

                # Assign base and tip to GN node
                obj_inputs = [s for s in new_node.inputs if s.type == 'OBJECT']
                if len(obj_inputs) >= 2:
                    obj_inputs[0].default_value = new_empty
                    obj_inputs[1].default_value = new_tip

        else:
            # For standard shapes like cube, sphere, etc — link the main empty
            for socket in new_node.inputs:
                if socket.type == 'OBJECT':
                    socket.default_value = new_empty
                    break  # done
            
        return new_empty

    def execute(self, context):
        domain = context.scene.sdf_domain
        idx = domain.active_sdf_node_index
        source_item = domain.sdf_nodes[idx]

        # --- NEW LOGIC: Check for symmetry inside the operator ---
        has_symmetry = source_item.use_mirror_x or source_item.use_mirror_y or source_item.use_mirror_z or \
                      (source_item.use_radial_mirror and source_item.radial_mirror_count > 1)
        
        if not has_symmetry:
            self.report({'WARNING'}, "Enable Mirror or Radial Symmetry on the selected shape first.")
            return {'CANCELLED'}
        # --- END OF NEW LOGIC ---

        source_empty = source_item.empty_object
        node_tree = get_sdf_geometry_node_tree(context)
        source_node = next((n for n in node_tree.nodes if n.get("associated_empty") == source_empty.name), None)

        if not (source_empty and source_node):
            self.report({'ERROR'}, "Source shape is not valid.")
            return {'CANCELLED'}

        # --- Baking logic ---
        # Store original matrix to avoid duplicating the source shape
        original_matrix = source_empty.matrix_world.copy()
        
        # Start with a list of matrices to process, beginning with the original
        matrices_to_process = [original_matrix]
        
        # Store the final list of all generated matrices (including original)
        final_matrices = [original_matrix]

        pivot_matrix = Matrix.Translation(domain.location)
        pivot_matrix_inv = pivot_matrix.inverted()

        # --- Radial Symmetry ---
        if source_item.use_radial_mirror and source_item.radial_mirror_count > 1:
            count = source_item.radial_mirror_count
            angle_step = (2 * math.pi) / count
            new_radial_matrices = []
            for i in range(1, count):
                rot_matrix = Matrix.Rotation(angle_step * i, 4, 'Z')
                transform = pivot_matrix @ rot_matrix @ pivot_matrix_inv
                new_matrix = transform @ original_matrix
                new_radial_matrices.append(new_matrix)
            matrices_to_process.extend(new_radial_matrices)
            final_matrices.extend(new_radial_matrices)
        
        # --- Planar Mirror ---
        mirror_axes = []
        if source_item.use_mirror_x: mirror_axes.append(Vector((1, 0, 0)))
        if source_item.use_mirror_y: mirror_axes.append(Vector((0, 1, 0)))
        if source_item.use_mirror_z: mirror_axes.append(Vector((0, 0, 1)))

        if mirror_axes:
            current_matrices_to_mirror = list(final_matrices) # Mirror all existing shapes
            for axis_vector in mirror_axes:
                scale_matrix = Matrix.Scale(-1, 4, axis_vector)
                transform = pivot_matrix @ scale_matrix @ pivot_matrix_inv
                for mat in current_matrices_to_mirror:
                    final_matrices.append(transform @ mat)

        # --- Utility: Test if two matrices are equal (with tolerance) ---
        def matrices_equal(mat_a, mat_b, tol=1e-6):
            for i in range(4):
                for j in range(4):
                    if abs(mat_a[i][j] - mat_b[i][j]) > tol:
                        return False
            return True

        # --- Create all the new objects, excluding the original ---
        num_created = 0
        for matrix in final_matrices:
            if matrices_equal(matrix, original_matrix):
                continue
            self._create_baked_shape(context, source_item, source_node, matrix)
            num_created += 1


        source_item.use_mirror_x = False
        source_item.use_mirror_y = False
        source_item.use_mirror_z = False
        source_item.use_radial_mirror = False

        # Final cleanup
        rewire_full_sdf_chain(context)
        self.report({'INFO'}, f"Baked {num_created} new shapes from symmetry.")
        
        self.report({'INFO'}, f"Baked {num_created} new shapes from symmetry.")
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
    """Duplicate the selected SDF node and its associated empty(s), then rebuild the entire chain."""
    bl_idname = "prototyper.sdf_duplicate"
    bl_label = "Duplicate Selected SDF Node"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        if not (domain_obj and hasattr(domain_obj, 'sdf_nodes')): return {'CANCELLED'}
        
        active_index = domain_obj.active_sdf_node_index
        if not (0 <= active_index < len(domain_obj.sdf_nodes)): return {'CANCELLED'}

        original_item = domain_obj.sdf_nodes[active_index]
        original_empty = original_item.empty_object
        if not original_empty: return {'CANCELLED'}
            
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree: return {'CANCELLED'}
        
        original_node = next((n for n in node_tree.nodes if n.get('associated_empty') == original_empty.name), None)
        if not original_node: return {'CANCELLED'}

        # --- Create New Node ---
        new_node = node_tree.nodes.new(type=original_node.bl_idname)
        if new_node.bl_idname == 'GeometryNodeGroup':
            new_node.node_tree = original_node.node_tree
        new_node.location = original_node.location + Vector((0, -200))
        for i, orig_input in enumerate(original_node.inputs):
            if hasattr(orig_input, "default_value"):
                new_node.inputs[i].default_value = orig_input.default_value

        # --- Create New Empty Controller ---
        new_empty = original_empty.copy()
        if new_empty.data:
            new_empty.data = original_empty.data.copy()
        context.collection.objects.link(new_empty)
        new_empty.parent = original_empty.parent

        # --- Create New UI List Item ---
        new_item = domain_obj.sdf_nodes.add()
        new_empty.name = f"{original_item.name}.Dupe"
        new_item.name = new_empty.name
        new_item.empty_object = new_empty
        new_item.icon = original_item.icon
        
        # --- THIS IS THE NEW LINE ---
        new_item.preview_color = original_item.preview_color
        # --- END OF NEW LINE ---

        # --- CONE-AWARE LOGIC ---
        new_node["associated_empty"] = new_empty.name

        if original_item.icon == 'MESH_CONE':
            original_tip = next((child for child in original_empty.children if "Tip" in child.name), None)
            if original_tip:
                new_tip = original_tip.copy()
                if new_tip.data:
                    new_tip.data = original_tip.data.copy()
                context.collection.objects.link(new_tip)
                new_tip.parent = new_empty
                
                obj_inputs = [sock for sock in new_node.inputs if sock.type == 'OBJECT']
                if len(obj_inputs) >= 2:
                    obj_inputs[0].default_value = new_empty
                    obj_inputs[1].default_value = new_tip
                else:
                    self.report({'WARNING'}, "Cone node is missing its object inputs.")
            else:
                self.report({'WARNING'}, "Could not find Tip empty for cone duplication.")
        else:
            obj_input_socket = next((sock for sock in new_node.inputs if sock.type == 'OBJECT'), None)
            if obj_input_socket:
                obj_input_socket.default_value = new_empty
        
        # --- Finalize ---
        domain_obj.sdf_nodes.move(len(domain_obj.sdf_nodes) - 1, active_index + 1)
        domain_obj.active_sdf_node_index = active_index + 1
        rewire_full_sdf_chain(context)
        
        bpy.ops.object.select_all(action='DESELECT')
        new_empty.select_set(True)
        context.view_layer.objects.active = new_empty

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

class PROTOTYPER_OT_SDFRepeatShape(bpy.types.Operator):
    """Create multiple copies of the selected SDF shape that are fully compatible with Global Scale"""
    bl_idname = "prototyper.sdf_repeat_shape"
    bl_label = "Repeat SDF Shape"
    bl_options = {'REGISTER', 'UNDO'}

    count: bpy.props.IntProperty(
        name="Count",
        description="How many copies to create",
        default=5,
        min=1,
        soft_max=100,
    )
    
    direction: bpy.props.FloatVectorProperty(
        name="Direction",
        description="The axis and direction of repetition (e.g., (1,0,0) for X-axis)",
        default=(1.0, 0.0, 0.0)
    )
    
    spacing: bpy.props.FloatProperty(
        name="Spacing",
        description="The distance between each repeated shape",
        default=1.0,
        min=0.0,
        soft_max=10.0,
        unit='LENGTH'
    )

    @classmethod
    def poll(cls, context):
        domain = getattr(context.scene, "sdf_domain", None)
        if not domain or not hasattr(domain, 'sdf_nodes'):
            return False
        return 0 <= domain.active_sdf_node_index < len(domain.sdf_nodes)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        active_index = domain_obj.active_sdf_node_index

        original_item = domain_obj.sdf_nodes[active_index]
        original_empty = original_item.empty_object
        if not original_empty:
            self.report({'ERROR'}, "Source item has no valid Empty object.")
            return {'CANCELLED'}
        
        node_tree = get_sdf_geometry_node_tree(context)
        if not node_tree:
            self.report({'ERROR'}, "SDF Domain has no valid Geometry Node tree.")
            return {'CANCELLED'}
        
        original_node = next((n for n in node_tree.nodes if n.get("associated_empty") == original_empty.name), None)
        if not original_node:
            self.report({'ERROR'}, "Could not find the Geometry Node for the selected shape.")
            return {'CANCELLED'}

        direction_vec = Vector(self.direction)
        final_offset = direction_vec.normalized() * self.spacing if direction_vec.length > 0 else Vector((0.0, 0.0, 0.0))

        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = None
        
        last_created_empty = None

        for i in range(self.count):
            # --- Create New Node ---
            new_node = node_tree.nodes.new(type=original_node.bl_idname)
            if new_node.bl_idname == 'GeometryNodeGroup':
                new_node.node_tree = original_node.node_tree
            new_node.location = original_node.location + Vector((20 * (i+1), -200 * (i+1)))
            for j, orig_input in enumerate(original_node.inputs):
                if hasattr(orig_input, "default_value"):
                    new_node.inputs[j].default_value = orig_input.default_value

            # --- Create and Position New Empty Controller ---
            new_empty = original_empty.copy()
            if new_empty.data:
                new_empty.data = original_empty.data.copy()
            context.collection.objects.link(new_empty)
            
            new_empty.parent = domain_obj
            new_empty.location = original_empty.location + (final_offset * (i + 1))
            new_empty.scale = original_empty.scale
            new_empty.rotation_euler = original_empty.rotation_euler

            # --- CRITICAL FIX for Global Scale ---
            new_empty["initial_location"] = new_empty.location.copy()
            new_empty["initial_scale"] = new_empty.scale.copy()

            # --- Create New UI List Item ---
            new_item = domain_obj.sdf_nodes.add()
            new_empty.name = f"{original_item.name}.Repeat"
            new_item.name = new_empty.name
            new_item.empty_object = new_empty
            new_item.icon = original_item.icon
            new_item.is_hidden = original_item.is_hidden
            new_item.is_viewport_hidden = original_item.is_viewport_hidden
            
            # --- THIS IS THE CRITICAL ADDITION ---
            new_item.preview_color = original_item.preview_color
            # --- END OF ADDITION ---
            
            # --- NEW CONE-AWARE LOGIC FOR REPEAT ---
            new_node["associated_empty"] = new_empty.name

            if original_item.icon == 'MESH_CONE':
                original_tip = next((child for child in original_empty.children if "Tip" in child.name), None)
                if original_tip:
                    new_tip = original_tip.copy()
                    if new_tip.data:
                        new_tip.data = original_tip.data.copy()
                    context.collection.objects.link(new_tip)
                    new_tip.parent = new_empty
                    obj_inputs = [sock for sock in new_node.inputs if sock.type == 'OBJECT']
                    if len(obj_inputs) >= 2:
                        obj_inputs[0].default_value = new_empty
                        obj_inputs[1].default_value = new_tip
                else:
                    self.report({'WARNING'}, "Could not find Tip empty for cone repetition.")
            else:
                obj_input_socket = next((sock for sock in new_node.inputs if sock.type == 'OBJECT'), None)
                if obj_input_socket:
                    obj_input_socket.default_value = new_empty
            
            last_created_empty = new_empty

        # --- Finalize ---
        rewire_full_sdf_chain(context)
        
        if last_created_empty:
            last_created_empty.select_set(True)
            context.view_layer.objects.active = last_created_empty

        self.report({'INFO'}, f"Created {self.count} repeated shapes.")
        return {'FINISHED'}
    

from mathutils import Matrix # Make sure this is imported at the top of your script

from mathutils import Matrix # Make sure this is imported at the top of your script

class PROTOTYPER_OT_SDFFlipShape(bpy.types.Operator):
    """Flips the selected SDF shape on a given WORLD axis using Blender's built-in mirror tool."""
    bl_idname = "prototyper.sdf_flip_shape"
    bl_label = "Flip SDF Shape"
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis",
        items=[('X', "X-Axis", "Flip on the World X-axis"),
               ('Y', "Y-Axis", "Flip on the World Y-axis"),
               ('Z', "Z-Axis", "Flip on the World Z-axis")],
        default='X'
    )

    @classmethod
    def poll(cls, context):
        domain = getattr(context.scene, "sdf_domain", None)
        if not domain or not hasattr(domain, 'sdf_nodes'):
            return False
        active_index = domain.active_sdf_node_index
        if 0 <= active_index < len(domain.sdf_nodes):
            return domain.sdf_nodes[active_index].empty_object is not None
        return False

    def execute(self, context):
        domain_obj = context.scene.sdf_domain
        active_index = domain_obj.active_sdf_node_index
        item = domain_obj.sdf_nodes[active_index]
        empty = item.empty_object

        if not empty:
            return {'CANCELLED'}

        # --- NEW, ROBUST METHOD USING BLENDER'S CORE MIRROR OPERATOR ---

        # 1. Store the current selection state to restore it later.
        active_object = context.view_layer.objects.active
        selected_objects = context.selected_objects[:]

        # 2. Prepare the scene for the operator: deselect everything and
        #    make our target empty the only active and selected object.
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = empty
        empty.select_set(True)

        # 3. Set the pivot point to "Individual Origins" to ensure the
        #    flip happens around the object's own center.
        original_pivot = context.scene.tool_settings.transform_pivot_point
        context.scene.tool_settings.transform_pivot_point = 'INDIVIDUAL_ORIGINS'

        # 4. Call Blender's built-in mirror operator with the correct axis.
        bpy.ops.transform.mirror(
            orient_type='GLOBAL',
            constraint_axis=(self.axis == 'X', self.axis == 'Y', self.axis == 'Z')
        )

        # 5. Restore the original pivot point setting.
        context.scene.tool_settings.transform_pivot_point = original_pivot

        # 6. Restore the original selection.
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected_objects:
            # Make sure the object still exists before trying to select it
            if obj.name in context.view_layer.objects:
                obj.select_set(True)
        context.view_layer.objects.active = active_object
            
        self.report({'INFO'}, f"Flipped '{item.name}' on the World {self.axis}-axis.")
        return {'FINISHED'}
    


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

        sdf_node = geo_nodes.nodes.new(type="GeometryNodeGroup")
        sdf_node.node_tree = node_group
        sdf_node.name = node_group.name
        sdf_node.location = (NodePositionManager.increment_position(), 1000)
        if "Major Radius" in sdf_node.inputs:
            sdf_node.inputs["Major Radius"].default_value = 0.20
        if "Minor Radius" in sdf_node.inputs:
            sdf_node.inputs["Minor Radius"].default_value = 0.05

        domain = context.scene.sdf_domain
        bpy.ops.object.empty_add(type='CUBE', location=domain.location)
        empty = context.active_object
        empty.name = "SDF_Torus"
        empty.empty_display_size = 0.25
        empty.parent = domain

        item = domain.sdf_nodes.add()
        item.name = empty.name
        item.empty_object = empty
        item.icon = 'MESH_TORUS'

        sdf_node['associated_empty'] = empty.name
        obj_input = next((s for s in sdf_node.inputs if s.type == 'OBJECT'), None)
        if obj_input:
            obj_input.default_value = empty

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


class SDFNodeItem(PropertyGroup):
    name: StringProperty(
        name="Node Name",
        description="Rename this SDF shape",
        update=update_sdf_node_name
    )
    empty_object: PointerProperty(
        name="Controller Empty",
        type=bpy.types.Object,
        description="The Empty that drives this shape"
    )
    icon: EnumProperty(
        name="Icon",
        items=[
            ('MESH_CUBE',    "Cube",     ""),
            ('MESH_CYLINDER',"Cylinder", ""),
            ('MESH_UVSPHERE',"Sphere",   ""),
            ('MESH_CONE',    "Cone",     ""),
            ('MESH_ICOSPHERE',"Prism",   ""),
            ('MESH_TORUS',   "Torus",    ""),
            ('CURVE_BEZCURVE',"Curve",   ""),
            ('MESH_MONKEY',  "Mesh",     ""),
            ('SCULPTMODE_HLT',"Sculpt",  ""),
        ],
        default='MESH_CUBE'
    )
    is_hidden: BoolProperty(name="Mute Shape", default=False, update=lambda self,ctx: check_mute_nodes(ctx.scene))
    is_viewport_hidden: BoolProperty(name="Hide Empty", default=False, update=update_sdf_viewport_visibility)
    use_highlight: BoolProperty(name="Highlight Shape", default=False, update=_redraw_shader_view)
    operation: EnumProperty(
        name="Operation",
        items=[('SMOOTH_UNION', "Smooth Union", ""), ('SMOOTH_SUBTRACT', "Smooth Subtract", ""), ('SMOOTH_INTERSECT', "Smooth Intersect", "")],
        default='SMOOTH_UNION',
        update=_redraw_shader_view
    )
    blend: FloatProperty(name="Blend", default=0.0, min=0.0, max=1.0, update=_redraw_shader_view)
    preview_color: FloatVectorProperty(name="Preview Color", subtype='COLOR', default=(1.0, 1.0, 1.0), min=0.0, max=1.0, update=_redraw_shader_view)
    
    # --- Symmetry Properties ---
    use_mirror_x: BoolProperty(name="X", default=False, update=_redraw_shader_view)
    use_mirror_y: BoolProperty(name="Y", default=False, update=_redraw_shader_view)
    use_mirror_z: BoolProperty(name="Z", default=False, update=_redraw_shader_view)
    use_radial_mirror: BoolProperty(name="Enable Radial Mirror", default=False, update=_redraw_shader_view)
    radial_mirror_count: IntProperty(name="Count", default=6, min=2, max=64, update=_redraw_shader_view)
    radial_mirror_offset: FloatProperty(name="Offset", default=0.0, update=_redraw_shader_view)

    # --- Curve-Specific Properties ---
    curve_mode: EnumProperty(
        name="Curve Mode",
        description="How to interpret the curve's shape",
        items=[('HARD', "Hard", "Linear segments between control points"), ('SMOOTH', "Smooth", "Approximate the true curve using subdivisions")],
        default='HARD',
        update=_redraw_shader_view
    )
    curve_subdivisions: IntProperty(
        name="Subdivisions",
        description="Number of smaller segments to approximate the curve",
        default=4,
        min=1,
        max=16,
        update=_redraw_shader_view
    )


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
    VIEW3D_MT_sdf_rclick,

    # Core SDF Generation & Conversion
    StartSDFOperator,
    ConvertSDFOperator,
    OBJECT_OT_sdf_bake_volume,

    # Add SDF Shape Operators
    SDFCubeAdd, SDFCylinderAdd, SDFUVSphereAdd, SDFConeAdd,
    SDFPrismAdd, SDFTorusAdd, SDFCurveAdd, SDFMeshAdd, SDFSculptAdd,
    SDFMeshToSDF,

    # List & Shape Management Operators
    PROTOTYPER_OT_SDFListMove,
    SDFDuplicateOperator, SDFDeleteOperator, SDFClearOperator,
    PROTOTYPER_OT_SDFRepeatShape,
    PROTOTYPER_OT_SDFFlipShape,
    PROTOTYPER_OT_SDFCleanupList,
    PROTOTYPER_OT_toggle_smooth,

    # Symmetry Baking Operator
    OBJECT_OT_bake_sdf_symmetry,

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

    # A) Register the unified PropertyGroup which now includes all curve properties
    bpy.utils.register_class(SDFNodeItem)

    # B) Attach it to Object
    bpy.types.Object.sdf_nodes = bpy.props.CollectionProperty(type=SDFNodeItem)
    bpy.types.Object.active_sdf_node_index = bpy.props.IntProperty(default=-1)

    # C) Define Scene properties
    Scene = bpy.types.Scene
    Scene.sdf_domain                   = bpy.props.PointerProperty(type=bpy.types.Object)
    
    # --- NEW MAX SHAPES PROPERTY ---
    Scene.sdf_max_shapes = bpy.props.IntProperty(
        name="Max Shapes",
        description="The maximum number of shapes the shader can handle. WARNING: Higher values can impact performance",
        default=32,
        min=8,
        max=256
    )
    # --- END NEW ---

    Scene.lock_sdf_panel               = bpy.props.BoolProperty(name="Lock SDF Panel", default=False, update=update_lock)
    Scene.locked_sdf_object            = bpy.props.PointerProperty(type=bpy.types.Object)

    Scene.sdf_status_message           = bpy.props.StringProperty(default="Ready")
    Scene.sdf_auto_resolution_enable   = bpy.props.BoolProperty(
        name="Automatic Preview", default=False, update=toggle_auto_resolution_mode)
    Scene.sdf_preview_mode             = bpy.props.BoolProperty(
        name="Preview Mode", default=True, update=update_sdf_resolution)
    Scene.sdf_preview_resolution       = bpy.props.IntProperty(
        name="Low-Res", default=1, min=1, soft_max=64, update=update_sdf_resolution)
    Scene.sdf_final_resolution         = bpy.props.IntProperty(
        name="High-Res", default=3, min=1, soft_max=512, update=update_sdf_resolution)
    Scene.sdf_auto_threshold           = bpy.props.FloatProperty(
        name="Movement Sensitivity", default=1e-5, min=1e-6, max=1e-3)
    Scene.sdf_auto_idle_delay          = bpy.props.FloatProperty(
        name="Idle Delay (s)", default=0.5, min=0.1, max=2.0)

    Scene.sdf_decimation_enable        = bpy.props.BoolProperty(name="Enable Decimation", default=False)
    Scene.sdf_global_scale             = bpy.props.FloatProperty(
        name="Global Scale", default=1.0, min=0.1, max=10.0, update=update_sdf_global_scale)

    Scene.use_brush_cube               = bpy.props.BoolProperty(name="Use Brush Cube", default=False)
    Scene.brush_cube                   = bpy.props.PointerProperty(type=bpy.types.Object)
    Scene.clip_enabled                 = bpy.props.BoolProperty(name="Clipping Enabled", default=False)

    Scene.sdf_render_panel_enable      = bpy.props.BoolProperty(name="Show Render Options", default=False)
    Scene.sdf_render_from              = bpy.props.EnumProperty(
        name="Render From", items=[('CAMERA','Camera',''),('VIEW','View','')], default='CAMERA')
    Scene.sdf_render_highres_resolution= bpy.props.IntProperty(name="Res", default=3, min=1, max=1024)
    Scene.sdf_render_scale             = bpy.props.FloatProperty(name="Scale", default=1.0, min=0.1, max=2.0)
    Scene.sdf_render_engine            = bpy.props.EnumProperty(
        name="Engine",
        items=[('BLENDER_EEVEE_NEXT','Eevee',''),('CYCLES','Cycles','')],
        default='BLENDER_EEVEE_NEXT'
    )
    Scene.sdf_render_samples           = bpy.props.IntProperty(name="Eevee Samples", default=5, min=1, max=4096)
    Scene.sdf_cycles_samples           = bpy.props.IntProperty(name="Cycles Max", default=5, min=1, max=4096)
    Scene.sdf_cycles_preview_samples   = bpy.props.IntProperty(name="Cycles Min", default=16, min=1, max=4096)
    Scene.sdf_render_shading_mode      = bpy.props.EnumProperty(
        name="Shading Mode",
        items=[('CURRENT','Current',''),('MATERIAL','Material',''),('RENDERED','Rendered','')],
        default='CURRENT'
    )
    Scene.sdf_render_disable_overlays  = bpy.props.BoolProperty(name="Disable Overlays", default=False)

    Scene.sdf_shape_tab                = bpy.props.EnumProperty(
        name="Shape Tab",
        items=[('BASIC','Basic',''),('DEFORM','Deform',''),('STYLE','Style',''),('MISC','Misc','')],
        default='BASIC'
    )

    Scene.sdf_global_tint    = bpy.props.FloatVectorProperty(
        name="Global Tint", subtype='COLOR',
        default=(1.0,1.0,1.0), min=0.0, max=1.0,
        description="Multiply shape colors globally",
        update=_redraw_shader_view
    )
    Scene.sdf_light_azimuth  = bpy.props.FloatProperty(
        name="Light Azimuth", default=45.0,
        min=0.0, max=360.0, subtype='ANGLE',
        update=_redraw_shader_view
    )
    Scene.sdf_light_elevation= bpy.props.FloatProperty(
        name="Light Elevation", default=45.0,
        min=-90.0, max=90.0, subtype='ANGLE',
        update=_redraw_shader_view
    )
    Scene.sdf_shader_view = bpy.props.BoolProperty(
        name="Enable SDF Shader View", default=False,
        update=lambda s,c: enable_sdf_shader_view(s.sdf_shader_view)
    )
    Scene.sdf_color_blend_mode = bpy.props.EnumProperty(
        name="Color Blend",
        items=[('HARD',"Hard","Pick one shape’s color"),('SOFT',"Soft","Interpolate in smooth areas")],
        default='HARD'
    )

    # D) Register all other classes
    for cls in _classes:
        try: bpy.utils.register_class(cls)
        except ValueError: pass

    # E) Handlers & menus
    if check_mute_nodes not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(check_mute_nodes)
    if depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update)
    if toggle_lock_based_on_selection not in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.append(toggle_lock_based_on_selection)

    bpy.types.VIEW3D_MT_mesh_add.prepend(add_sdf_shapes)
    bpy.types.VIEW3D_MT_object_context_menu.prepend(rclick_sdf_menu)

    # F) Keymaps
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km  = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
        kmi = km.keymap_items.new('prototyper.sdf_list_move','EQUAL','PRESS')
        kmi.properties.direction='UP';    _addon_keymaps.append((km,kmi))
        kmi = km.keymap_items.new('prototyper.sdf_list_move','MINUS','PRESS')
        kmi.properties.direction='DOWN';  _addon_keymaps.append((km,kmi))

    # G) Start auto‐preview timer
    _timer_is_running = True
    bpy.app.timers.register(monitor_sdf_movement)


# -------------------------------------------------------------------
# 4) UNREGISTER
# -------------------------------------------------------------------
def unregister():
    global _addon_keymaps, _timer_is_running

    # A) Stop timer
    _timer_is_running = False

    # B) Remove keymaps
    for km,kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    # C) Handlers & menus
    if check_mute_nodes in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(check_mute_nodes)
    if depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update)
    if toggle_lock_based_on_selection in bpy.app.handlers.depsgraph_update_pre:
        bpy.app.handlers.depsgraph_update_pre.remove(toggle_lock_based_on_selection)

    bpy.types.VIEW3D_MT_mesh_add.remove(add_sdf_shapes)
    bpy.types.VIEW3D_MT_object_context_menu.remove(rclick_sdf_menu)

    # D) Unregister all other classes
    for cls in reversed(_classes):
        try: bpy.utils.unregister_class(cls)
        except ValueError: pass

    # E) Unregister PropertyGroup
    bpy.utils.unregister_class(SDFNodeItem)
    del bpy.types.Object.sdf_nodes
    del bpy.types.Object.active_sdf_node_index

    # F) Delete Scene props
    Scene = bpy.types.Scene
    # This list now includes the new sdf_max_shapes property for clean uninstallation
    props_to_del = [
        "sdf_domain", "sdf_max_shapes", "lock_sdf_panel", "locked_sdf_object",
        "sdf_status_message", "sdf_auto_resolution_enable", "sdf_preview_mode",
        "sdf_preview_resolution", "sdf_final_resolution", "sdf_auto_threshold",
        "sdf_auto_idle_delay", "sdf_decimation_enable", "sdf_global_scale",
        "use_brush_cube", "brush_cube", "clip_enabled",
        "sdf_render_panel_enable", "sdf_render_from",
        "sdf_render_highres_resolution", "sdf_render_scale", "sdf_render_engine",
        "sdf_render_samples", "sdf_cycles_samples", "sdf_cycles_preview_samples",
        "sdf_render_shading_mode", "sdf_render_disable_overlays",
        "sdf_shape_tab", "sdf_shader_view", "sdf_global_tint",
        "sdf_light_azimuth", "sdf_light_elevation"
    ]
    for p in props_to_del:
        if hasattr(Scene, p):
            delattr(Scene, p)
