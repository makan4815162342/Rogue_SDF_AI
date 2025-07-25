// Rogue SDF AI — Volume Baking Fragment Shader
// This version uses itemID grouping for robust booleans.

in vec2 uv;
out vec4 fragColor;

// ... (Uniforms for bounds and slice are unchanged) ...
uniform vec3 uBoundsMin;
uniform vec3 uBoundsMax;
uniform int uSliceIndex;
uniform int uDepth;

// SDF shape data
uniform int    uCount;
uniform ivec4  uShapeTypePacked[16];
uniform vec3   uShapePos[64];
uniform vec3   uShapeScale[64];
uniform vec4   uShapeRot[64];
uniform int    uShapeOp[64];
uniform float  uShapeBlend[64];
uniform int    uShapeMirrorFlags[64];
uniform int    uShapeRadialCount[64];
uniform int    uShapeItemID[64]; // NEW: For grouping

uniform vec3   uDomainCenter;

// ... (Constants, Helpers, Primitives, and Smooth Ops are unchanged) ...
const float PI = 3.14159265359;
vec3 rotateByQuat(vec3 p, vec4 q) { return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p); }
float sdfSphere(vec3 p, float r) { return length(p) - r; }
float sdfBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0); }
float sdfTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz)-t.x,p.y); return length(q)-t.y; }
float sdfCylinder(vec3 p, float h, float r) { vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h); return min(max(d.x,d.y),0.0) + length(max(d,0.0)); }
float sdfPrism(vec3 p, vec2 h) { vec3 q = abs(p); return max(q.z - h.x, max(q.x*0.866025+q.y*0.5, q.y) - h.y); }
float sdfCone(vec3 p, float h, float r1, float r2) { p.y -= h * 0.5; vec2 q = vec2(length(p.xz), p.y); vec2 k1 = vec2(r2, h); vec2 k2 = vec2(r2 - r1, 2.0 * h); vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h); vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0); float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0; return s * sqrt(min(dot(ca, ca), dot(cb, cb))); }
float sdfCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }
float opSmoothUnion(float a,float b,float k){ if(k<=0.)return min(a,b); float h=clamp(0.5+0.5*(b-a)/k,0.,1.); return mix(b,a,h)-k*h*(1.-h); }
float opSmoothSubtract(float a,float b,float k){ if(k<=0.)return max(a,-b);float h=clamp(0.5-0.5*(a+b)/k,0.,1.);return mix(a,-b,h)+k*h*(1.-h);}
float opSmoothIntersection(float a,float b,float k){ if(k<=0.)return max(a,b); float h=clamp(0.5-0.5*(b-a)/k,0.,1.); return mix(b,a,h)+k*h*(1.0-h); }

// --- NEW: Combine function for geometry only ---
float combine(float d1, float d2, int op, float k) {
    if      (op == 0) return min(d1, d2);
    else if (op == 1) return max(d1, -d2);
    else if (op == 2) return max(d1, d2);
    else if (op == 3) return opSmoothUnion(d1, d2, k);
    else if (op == 4) return opSmoothSubtract(d1, d2, k);
    else if (op == 5) return opSmoothIntersection(d1, d2, k);
    return d1;
}

// --- DEFINITIVE sceneSDF with Grouping Logic ---
float sceneSDF(vec3 p) {
    float scene_d = 1e6;
    float group_d = 1e6;
    int last_itemID = -1;

    if (uCount == 0) return scene_d;

    for (int i = 0; i < uCount; ++i) {
        int vi = i >> 2, ci = i & 3;
        int st = (ci==0? uShapeTypePacked[vi].x : ci==1? uShapeTypePacked[vi].y : ci==2? uShapeTypePacked[vi].z : uShapeTypePacked[vi].w);
        if (st < 0) continue;

        vec3 p_sym = p;
        int radial_count = uShapeRadialCount[i];
        if (radial_count > 1) { vec3 p_centered = p - uDomainCenter; float angle = atan(p_centered.y, p_centered.x); float radius = length(p_centered.xy); float segmentAngle = 2.0 * PI / float(radial_count); angle = mod(angle, segmentAngle) - segmentAngle * 0.5; p_centered.xy = vec2(cos(angle), sin(angle)) * radius; p_sym = p_centered + uDomainCenter; }
        int flags = uShapeMirrorFlags[i];
        if ((flags & 1) > 0) p_sym.x = uDomainCenter.x - abs(p_sym.x - uDomainCenter.x);
        if ((flags & 2) > 0) p_sym.y = uDomainCenter.y - abs(p_sym.y - uDomainCenter.y);
        if ((flags & 4) > 0) p_sym.z = uDomainCenter.z - abs(p_sym.z - uDomainCenter.z);
        
        vec3 pl = p_sym - uShapePos[i];
        vec4 rq = uShapeRot[i];
        vec4 iq = vec4(rq.w, -rq.x, -rq.y, -rq.z);
        pl = rotateByQuat(pl, iq);

        float sd = 1e6;
        if (st == 0) sd = sdfBox(pl, uShapeScale[i] * 0.5); else if (st == 1) sd = sdfSphere(pl, uShapeScale[i].x * 0.5); else if (st == 2) sd = sdfTorus(pl, vec2(uShapeScale[i].x * 0.4, uShapeScale[i].x * 0.1)); else if (st == 3) sd = sdfCylinder(pl, uShapeScale[i].y * 0.5, uShapeScale[i].x * 0.5); else if (st == 4) sd = sdfCone(pl, uShapeScale[i].y, uShapeScale[i].x, uShapeScale[i].z); else if (st == 5) sd = sdfPrism(pl, uShapeScale[i].xy * 0.5); else if (st == 6) { float height = uShapeScale[i].y; float radius = uShapeScale[i].x; vec3 p1 = vec3(0, -height * 0.5, 0); vec3 p2 = vec3(0,  height * 0.5, 0); sd = sdfCapsule(pl, p1, p2, radius); }

        int current_itemID = uShapeItemID[i];
        if (current_itemID != last_itemID && last_itemID != -1) {
            int final_op = uShapeOp[i-1];
            float final_blend = uShapeBlend[i-1];
            scene_d = combine(scene_d, group_d, final_op, final_blend);
            group_d = sd;
        } else {
            group_d = combine(group_d, sd, 0, 0.0);
        }
        last_itemID = current_itemID;
    }
    
    int final_op = uShapeOp[uCount-1];
    float final_blend = uShapeBlend[uCount-1];
    scene_d = combine(scene_d, group_d, final_op, final_blend);

    return scene_d;
}

void main() {
    float nz = (float(uSliceIndex)+0.5)/float(uDepth);
    vec3 uvw = vec3(uv, nz);
    vec3 p = mix(uBoundsMin, uBoundsMax, uvw);
    float d = sceneSDF(p);
    fragColor = vec4(d, 0.0, 0.0, 1.0);
}