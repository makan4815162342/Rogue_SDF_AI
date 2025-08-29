// Rogue SDF AI — Volume Baking Fragment Shader
// Version: 2025-08-17 - Added Color Baking Support

in vec2 uv;
out vec4 fragColor;

// ── UNIFORMS ──────────────────────────────────────────────────────────────
uniform vec3 uBoundsMin;
uniform vec3 uBoundsMax;
uniform int uSliceIndex;
uniform int uDepth;

uniform int    uCount;
uniform ivec4  uShapeTypePacked[16];
uniform vec3   uShapePos[64];
uniform vec3   uShapeScale[64];
uniform vec4   uShapeRot[64];
uniform int    uShapeOp[64];
uniform float  uShapeBlend[64];
uniform float  uShapeBlendStrength[64];
uniform float  uShapeMaskFill[64];
uniform int    uShapeMirrorFlags[64];
uniform int    uShapeRadialCount[64];
uniform int    uShapeItemID[64];

uniform vec4   uShapeParams1[64];
uniform vec4   uShapeParams2[64];

uniform vec3   uDomainCenter;
uniform vec3   uShapeColor[64];
uniform int    uColorBlendMode;  // NEW: 0 for HARD, 1 for SOFT (from scene.sdf_color_blend_mode)

// ── CONSTANTS & HELPERS ───────────────────────────────────────────────────
const float PI = 3.14159265359;
const float TAU = 6.28318530718;
vec3 rotateByQuat(vec3 p, vec4 q) { return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p); }

// ── PRIMITIVES ────────────────────────────────────────────────────────────
float sdfBox(vec3 p, vec3 b, float r, float bevel, float thickness) {
    float d_outer;
    if (bevel > 0.0) {
        float f = bevel * 0.70710678;
        vec3 d = abs(p) - b;
        d_outer = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - f;
    } else {
        vec3 d = abs(p) - b;
        d_outer = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - r;
    }
    if (thickness > 0.0) {
        vec3 b_inner = b - thickness;
        b_inner.z = b.z + 0.1;
        vec3 di = abs(p) - b_inner;
        float d_inner = length(max(di, 0.0)) + min(max(di.x, max(di.y, di.z)), 0.0);
        return max(d_outer, -d_inner);
    }
    return d_outer;
}
float sdfTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz)-t.x,p.y); return length(q)-t.y; }
float sdfCylinder(vec3 p, float h, float r) { vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h); return min(max(d.x,d.y),0.0) + length(max(d,0.0)); }
float sdfCylinderZ(vec3 p, float h, float r) { vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }
float sdfCone(vec3 p, float h, float r1, float r2) { p.y -= h * 0.5; vec2 q = vec2(length(p.xz), p.y); vec2 k1 = vec2(r2, h); vec2 k2 = vec2(r2 - r1, 2.0 * h); vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h); vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0); float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0; return s * sqrt(min(dot(ca, ca), dot(cb, cb))); }
float sdfCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }

// --- ADVANCED SHAPE FUNCTIONS ---
float sdfSphereAdvanced(vec3 p, float r, float thickness_ratio, float elongation, float cut_angle_rad) {
    float d;
    if (elongation > 0.0) { d = sdfCapsule(p, vec3(0,0,-elongation), vec3(0,0,elongation), r); }
    else { d = length(p) - r; }
    if (thickness_ratio > 0.0) { float hole_radius = r * (1.0 - thickness_ratio); float hole_half_height = (r + elongation) * 1.05; float d_hole = sdfCylinderZ(p, hole_half_height, hole_radius); d = max(d, -d_hole); }
    if (cut_angle_rad < 0.001) return 1e6;
    if (cut_angle_rad > TAU - 0.001) return d;
    vec2 n1 = vec2(0.0, -1.0);
    vec2 n2 = vec2(sin(cut_angle_rad), -cos(cut_angle_rad));
    if (cut_angle_rad > PI) { n1 = -n1; n2 = -n2; }
    float d_plane1 = dot(p.xy, n1);
    float d_plane2 = dot(p.xy, n2);
    float d_wedge = (cut_angle_rad > PI) ? min(d_plane1, d_plane2) : max(d_plane1, d_plane2);
    return max(d, d_wedge);
}

float sdfCylinderAdvanced(vec3 p, float h, float r, float thickness_ratio, float roundness, float pyramid, float bend) {
    float deform_factor = 1.0;
    if (abs(bend) > 0.001 && h > 0.0) {
        float bend_factor = (h*h - p.y*p.y) / (h*h);
        p.x -= bend * bend_factor;
        deform_factor = mix(0.8, 0.2, smoothstep(0.0, 2.0, abs(bend)));
    }
    float tapered_radius = r;
    if (pyramid > 0.0) {
        float taper_t = clamp((p.y / h) * 0.5 + 0.5, 0.0, 1.0);
        tapered_radius = mix(r, r * (1.0 - pyramid), taper_t);
        deform_factor = min(deform_factor, 0.6);
    }
    vec2 d_abs = abs(vec2(length(p.xz), p.y)) - vec2(tapered_radius, h);
    float sd;
    if (roundness > 0.0) {
        d_abs += roundness;
        sd = min(max(d_abs.x, d_abs.y), 0.0) + length(max(d_abs, 0.0)) - roundness;
    } else {
        sd = min(max(d_abs.x, d_abs.y), 0.0) + length(max(d_abs, 0.0));
    }
    if (thickness_ratio > 0.0) {
        float inner_radius = tapered_radius * (1.0 - thickness_ratio);
        float d_hole;
        if (pyramid > 0.0) {
            float inner_r2 = r * (1.0 - thickness_ratio) * (1.0 - pyramid);
            d_hole = sdfCone(p, h * 2.05, inner_radius, inner_r2);
        } else {
            d_hole = sdfCylinder(p, h * 1.05, inner_radius);
        }
        sd = max(sd, -d_hole);
    }
    return sd * deform_factor;
}

float sdfPrismAdvanced(vec3 p, float h, float r, float n_sides, float thickness_ratio) {
    float ang = TAU / n_sides;
    float a = atan(p.z, p.x) + ang/2.0;
    a = mod(a, ang) - ang/2.0;
    vec2 p_folded = length(p.xz) * vec2(cos(a), sin(a));
    float d_solid_2d = p_folded.x - r * cos(ang/2.0);
    vec2 d_prism = vec2(d_solid_2d, abs(p.y) - h);
    float final_d = min(max(d_prism.x, d_prism.y), 0.0) + length(max(d_prism, 0.0));
    if (thickness_ratio > 0.0) {
        float inner_r = r * (1.0 - thickness_ratio);
        float d_inner_2d = p_folded.x - inner_r * cos(ang/2.0);
        vec2 d_prism_inner = vec2(d_inner_2d, abs(p.y) - h * 1.05);
        float d_inner = min(max(d_prism_inner.x, d_prism_inner.y), 0.0) + length(max(d_prism_inner, 0.0));
        return max(final_d, -d_inner);
    }
    return final_d;
}

float sdfTorusAdvanced(vec3 p, float outer_r, float inner_r, float cut_angle_rad, float thickness_ratio, float elongation) {
    if (outer_r <= inner_r) return 1e6;
    float major_r = (outer_r + inner_r) * 0.5;
    float minor_r = (outer_r - inner_r) * 0.5;
    vec2 p_xz = p.xz;
    float dist_from_segment;
    if (elongation > 0.0) {
        vec2 a = vec2(0, -elongation);
        vec2 b = vec2(0, elongation);
        vec2 pa = p_xz - a;
        vec2 ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        dist_from_segment = length(pa - ba * h);
    } else {
        dist_from_segment = length(p_xz);
    }
    vec2 q = vec2(dist_from_segment - major_r, p.y);
    float d = length(q) - minor_r;
    if (cut_angle_rad > 0.001 && cut_angle_rad < TAU - 0.001) {
        float angle = atan(p.z, p.x);
        if (angle < 0.0) angle += TAU;
        if (angle > cut_angle_rad) d = 1e6;
    }
    if (thickness_ratio > 0.0) {
        float hole_minor_r = minor_r * (1.0 - thickness_ratio);
        vec2 q_hole = vec2(dist_from_segment - major_r, p.y);
        float d_hole = length(q_hole) - hole_minor_r;
        d = max(d, -d_hole);
    }
    return d;
}

// --- UNIVERSAL EVALUATION FUNCTION ---
struct Hit { float d; vec3 col; int id; };

Hit evalPrim(int i, vec3 pW) {
    int vi = i >> 2, ci = i & 3;
    int st = (ci == 0 ? uShapeTypePacked[vi].x : ci == 1 ? uShapeTypePacked[vi].y : ci == 2 ? uShapeTypePacked[vi].z : uShapeTypePacked[vi].w);
    if (st < 0) return Hit(1e6, vec3(0), -1);

    vec3 p = pW;
    int rc = uShapeRadialCount[i];
    if (rc > 1) { vec3 q = p - uDomainCenter; float ang = atan(q.y, q.x), seg = 2.0 * PI / float(rc), r = length(q.xy); ang = mod(ang, seg) - seg * 0.5; q.xy = vec2(cos(ang), sin(ang)) * r; p = q + uDomainCenter; }
    int f = uShapeMirrorFlags[i];
    if ((f & 1) != 0) p.x = uDomainCenter.x - abs(p.x - uDomainCenter.x);
    if ((f & 2) != 0) p.y = uDomainCenter.y - abs(p.y - uDomainCenter.y);
    if ((f & 4) != 0) p.z = uDomainCenter.z - abs(p.z - uDomainCenter.z);

    vec3 lp = p - uShapePos[i];
    vec4 rq = uShapeRot[i];
    lp = rotateByQuat(lp, vec4(rq.w, -rq.xyz));
    vec3 S = uShapeScale[i];
    float sd = 1e6;

    if (st == 0) { // Cube
        vec3 box_half_size = abs(S) * 0.5;
        vec4 params1 = uShapeParams1[i]; float thickness = params1.x, roundness = params1.y, bevel = params1.z, pyramid = params1.w;
        vec4 params2 = uShapeParams2[i]; float twist = params2.x, bend = params2.y;
        if (abs(bend) > 0.0) { float c = cos(bend * lp.z); float s = sin(bend * lp.z); mat2 rot = mat2(c, -s, s, c); lp.xz = rot * lp.xz; }
        if (abs(twist) > 0.0) { float c = cos(twist * PI * lp.z); float s = sin(twist * PI * lp.z); mat2 rot = mat2(c, -s, s, c); lp.xy = rot * lp.xy; }
        float xy_scale = 1.0;
        if (pyramid > 0.0) { float taper_t = clamp((lp.z / box_half_size.z) * 0.5 + 0.5, 0.0, 1.0); xy_scale = mix(1.0, 1.0 - pyramid, taper_t); lp.xy /= (xy_scale + 1e-6); }
        float max_dim = max(box_half_size.x, max(box_half_size.y, box_half_size.z));
        sd = sdfBox(lp, box_half_size, roundness * max_dim, bevel * max_dim, thickness * max_dim);
        if (pyramid > 0.0) { sd *= xy_scale; }
    }
    else if (st == 1) { // Sphere
        vec3 AS = abs(S);
        float scale_factor = min(AS.x, min(AS.y, AS.z));
        lp /= AS;
        vec4 params1 = uShapeParams1[i];
        float unit_r = 0.5;
        float thickness_ratio = params1.x;
        float rel_elong = params1.y;
        float cut_angle_rad = params1.z;
        float abs_elong = rel_elong * unit_r;
        sd = sdfSphereAdvanced(lp, unit_r, thickness_ratio, abs_elong, cut_angle_rad);
        sd *= scale_factor;
    }
    else if (st == 3) { // Cylinder
        float h = S.y * 0.5; float r = S.x * 0.5;
        vec4 params1 = uShapeParams1[i]; float thickness = params1.x, roundness = params1.y * min(h,r), pyramid = params1.z;
        vec4 params2 = uShapeParams2[i]; float bend = params2.y;
        sd = sdfCylinderAdvanced(lp, h, r, thickness, roundness, pyramid, bend);
    }
    else if (st == 5) { // Prism (N-gon)
        float h = S.y * 0.5; float r = S.x * 0.5;
        vec4 params1 = uShapeParams1[i]; float n_sides = params1.x, pyramid = params1.y, thickness = params1.z;
        vec4 params2 = uShapeParams2[i]; float bend = params2.x, twist = params2.y;
        if (abs(bend) > 0.0) { lp.x -= bend * lp.y * lp.y; }
        if (abs(twist) > 0.0) { float c = cos(twist * lp.y); float s = sin(twist * lp.y); mat2 rot = mat2(c, -s, s, c); lp.xz = rot * lp.xz; }
        float taper_scale = 1.0;
        if (pyramid > 0.0) { float taper_t = clamp((lp.y / h) * 0.5 + 0.5, 0.0, 1.0); taper_scale = mix(1.0, 1.0 - pyramid, taper_t); lp.xz /= taper_scale; }
        sd = sdfPrismAdvanced(lp, h, r, n_sides, thickness);
        if (pyramid > 0.0) { sd *= taper_scale; }
        if (abs(bend) > 0.0 || abs(twist) > 0.0 || pyramid > 0.0) { sd *= 0.6; }
    }
    else if (st == 4) { // Cone
        sd = sdfCone(lp, S.y, S.x, S.z);
    }
    else if (st == 2) { // Advanced Torus
        vec3 AS = abs(S);
        float scale_factor = min(AS.x, min(AS.y, AS.z));
        lp /= AS;
        vec4 params1 = uShapeParams1[i];
        vec4 params2 = uShapeParams2[i];
        float outer_r = params1.x;
        float inner_r = params1.y;
        float cut_angle_rad = params1.z;
        float thickness_ratio = params1.w;
        float elongation = params2.x;
        sd = sdfTorusAdvanced(lp, outer_r, inner_r, cut_angle_rad, thickness_ratio, elongation);
        sd *= scale_factor;
    }
    else { // Fallback for simple shapes (now only Capsule)
        lp /= S;
        if (st == 6) sd = sdfCapsule(lp, vec3(0, -0.5, 0), vec3(0, 0.5, 0), 0.5);
        sd *= min(abs(S.x), min(abs(S.y), abs(S.z)));
    }

    return Hit(sd, uShapeColor[i], uShapeItemID[i]);
}

// ── SMOOTH BLENDS & COMBINE ───────────────────────────────────────────────
float opRoundU(float A, float B, float k) { if (k <= 0.) return min(A, B); float h = clamp(0.5 + 0.5 * (B - A) / k, 0., 1.); return mix(B, A, h) - k * h * (1. - h); }
float opRoundSub(float A, float B, float k) { if (k <= 0.) return max(A, -B); float h = clamp(0.5 - 0.5 * (A + B) / k, 0., 1.); return mix(A, -B, h) + k * h * (1. - h); }
float opRoundI(float A, float B, float k) { if (k <= 0.) return max(A, B); float h = clamp(0.5 - 0.5 * (B - A) / k, 0., 1.); return mix(B, A, h) + k * h * (1. - h); }

float opChamferU(float a, float b, float r) { if (r <= 0.0) return min(a, b); return min(min(a, b), (a - r + b)*0.70710678); }
float opChamferSub(float a, float b, float r) { if (r <= 0.0) return max(a, -b); return max(a, -b) + r*0.70710678; }
float opChamferI(float a, float b, float r) { if (r <= 0.0) return max(a, b); return max(a, b) - r*0.70710678; }

// FINAL: Groove (Concave/Inverted Round) Blending Functions
float opGrooveU(float A, float B, float k) { if (k <= 0.) return min(A, B); float h = clamp(0.5 + 0.5 * (B - A) / k, 0., 1.); return mix(B, A, h) + k * h * (1. - h); }
float opGrooveSub(float A, float B, float k) { if (k <= 0.) return max(A, -B); float h = clamp(0.5 - 0.5 * (A + B) / k, 0., 1.); return mix(A, -B, h) - k * h * (1. - h); }
float opGrooveI(float A, float B, float k) { if (k <= 0.) return max(A, B); float h = clamp(0.5 - 0.5 * (B - A) / k, 0., 1.); return mix(B, A, h) - k * h * (1. - h); }

// FINAL: Pipe (Exterior Bead) Blending Functions
float opPipe(float a, float b, float r) {
    // This helper function defines the pipe geometry itself at the intersection.
    return opRoundI(a, b, r * 0.5) - r;
}

float opPipeU(float a, float b, float r) {
    // Union: The result is the union of the original shapes AND the pipe.
    return min(min(a, b), opPipe(a, b, r));
}

float opPipeSub(float a, float b, float r) {
    // Subtract: The result is the subtraction, with the pipe added along the seam.
    return min(max(a, -b), opPipe(a, b, r));
}

float opPipeI(float a, float b, float r) {
    // Intersect: The result is just the pipe itself, as it's defined by the intersection.
    return opPipe(a, b, r);
}


// 'k' is 'Smoothness'/'Chamfer', 'strength' is 'Strength', 'fill' is 'Fill Amount'
Hit combine(Hit A, Hit B, int op, float k, float strength, float fill) {
    float da = A.d, db = B.d;
    float d;
    vec3 c;
    int id;
    float h_col = 0.0;

    // --- Special Operations that return immediately ---
    if (op == 6) { /* Paint */ d = A.d; id = A.id; c = mix(A.col, B.col, clamp(1.0 - db / max(strength, 0.001), 0.0, 1.0)); return Hit(d, c, id); }
    else if (op == 7) { /* Displace */ float displacement = strength * (1.0 - smoothstep(0.0, k * strength + 1e-6, db)); d = da - displacement; h_col = clamp(displacement / (abs(strength) + 1e-6), 0.0, 1.0); c = mix(A.col, B.col, h_col); id = A.id; d *= 0.7; return Hit(d, c, id); }
    else if (op == 8) { /* Indent */ float displacement = strength * (1.0 - smoothstep(0.0, k * strength + 1e-6, db)); d = da + displacement; h_col = clamp(displacement / (abs(strength) + 1e-6), 0.0, 1.0); c = mix(A.col, B.col, h_col); id = A.id; d *= 0.7; return Hit(d, c, id); }
    else if (op == 9) { /* Relief */ float carved_base = opRoundSub(da, db - strength, k); d = min(carved_base, db); if (carved_base < db) { c = A.col; id = A.id; } else { c = B.col; id = B.id; } return Hit(d, c, id); }
    else if (op == 10) { /* Engrave */ float carved_tool = opRoundSub(db, da - strength, k); d = min(carved_tool, da); if (carved_tool < da) { c = B.col; id = B.id; } else { c = A.col; id = A.id; } return Hit(d, c, id); }
    else if (op == 11) { // Mask
        float d_intersect = opRoundI(da, db, k);
        float fill_amount = clamp(fill, 0.0, 1.0);
        d = mix(d_intersect, db, fill_amount);
        c = mix(A.col, B.col, fill_amount);
        id = fill_amount < 0.5 ? A.id : B.id;
        return Hit(d, c, id);
    }

    // --- Standard & Chamfer CSG Operations (Fall-through) ---
    else if (op == 3) { d = opRoundU(da, db, k); }
    else if (op == 4) { d = opRoundSub(da, db, k); }
    else if (op == 5) { d = opRoundI(da, db, k); }
    else if (op == 13) { d = opChamferU(da, db, k); }
    else if (op == 14) { d = opChamferSub(da, db, k); }
    else if (op == 15) { d = opChamferI(da, db, k); }
    else if (op == 23) { d = opGrooveU(da, db, k); }
    else if (op == 24) { d = opGrooveSub(da, db, k); }
    else if (op == 25) { d = opGrooveI(da, db, k); }
    else if (op == 33) { d = opPipeU(da, db, k); }
    else if (op == 34) { d = opPipeSub(da, db, k); }
    else if (op == 35) { d = opPipeI(da, db, k); }
    else { d = min(da, db); } // Hard Union

    // --- Unified Color Blending for Standard & Chamfer Ops ---
    bool isSmoothBlend = (op >= 3 && op <= 5) || (op >= 23 && op <= 35); // Updated to include Groove and Pipe
    if (uColorBlendMode == 1 && k > 0.0 && isSmoothBlend) { // Soft Color Blending
        if (op == 3 || op == 23 || op == 33) { h_col = clamp(0.5 + 0.5 * (db - da) / k, 0., 1.); c = mix(B.col, A.col, h_col); id = h_col < 0.5 ? B.id : A.id; }
        else if (op == 4 || op == 24 || op == 34) { h_col = clamp(0.5 - 0.5 * (da + db) / k, 0., 1.); c = mix(A.col, B.col, h_col); id = h_col < 0.5 ? A.id : B.id; }
        else { h_col = clamp(0.5 - 0.5 * (db - da) / k, 0., 1.); c = mix(B.col, A.col, h_col); id = h_col < 0.5 ? B.id : A.id; }
    } else { // Hard Color Blending
        if (op == 4 || op == 14 || op == 24 || op == 34) { // Subtract
             if (da > -db) { c = A.col; id = A.id; } else { c = B.col; id = B.id; }
        } else if (op == 5 || op == 15 || op == 25 || op == 35) { // Intersect
            if (da > db) { c = A.col; id = A.id; } else { c = B.col; id = B.id; }
        } else { // Union
            if (da < db) { c = A.col; id = A.id; } else { c = B.col; id = B.id; }
        }
    }
    return Hit(d, c, id);
}

// ── SCENE SDF (with Color) ────────────────────────────────────────────────
Hit sceneSDF(vec3 p) {
    Hit accum = Hit(1e6, vec3(1), -1);
    if (uCount == 0) return accum;

    Hit group = Hit(1e6, vec3(0), -1);
    int currentID = -1;
    int lastOp = 0;
    float lastBlend = 0.0;
    float lastStrength = 0.0;
    float lastFill = 0.0; // <-- ADD THIS

    for (int i = 0; i < uCount; ++i) {
        Hit h = evalPrim(i, p);
        if (h.id < 0) continue;

        if (h.id != currentID && currentID != -1) {
            accum = combine(accum, group, lastOp, lastBlend, lastStrength, lastFill); // <-- UPDATE THIS
            group = h;
        } else {
            group = (group.id < 0) ? h : combine(group, h, 0, 0.0, 0.0, 0.0);  // <-- UPDATE THIS
        }
        currentID = h.id;
        lastOp = uShapeOp[i];
        lastBlend = uShapeBlend[i];
        lastStrength = uShapeBlendStrength[i];
        lastFill = uShapeMaskFill[i]; // <-- ADD THIS
    }

    if (currentID != -1) {
        accum = combine(accum, group, lastOp, lastBlend, lastStrength, lastFill); // <-- UPDATE THIS
    }
    return accum;
}

// ── MAIN ──────────────────────────────────────────────────────────────────
void main() {
    float nz = (float(uSliceIndex) + 0.5) / float(uDepth);
    vec3 uvw = vec3(uv, nz);
    vec3 p = mix(uBoundsMin, uBoundsMax, uvw);
    Hit h = sceneSDF(p);
    fragColor = vec4(h.d, h.col);
}