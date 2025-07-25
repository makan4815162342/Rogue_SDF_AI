// Rogue SDF AI — Final Ray-March & Shading
// This version uses itemID grouping for robust booleans.

// ——— Inputs & Outputs ———
in vec2  uv;
out vec4 fragColor;

// ——— Uniforms ———
uniform vec2  viewportSize;
uniform mat4  viewMatrixInv;
uniform mat4  projMatrixInv;
uniform vec3  uLightDir;

uniform int    uCount;
uniform ivec4  uShapeTypePacked[16];
uniform vec3   uShapePos[64];
uniform vec3   uShapeScale[64];
uniform vec4   uShapeRot[64];
uniform int    uShapeOp[64];
uniform float  uShapeBlend[64];
uniform vec3   uShapeColor[64];
uniform int    uShapeMirrorFlags[64];
uniform int    uShapeRadialCount[64];
uniform int    uShapeHighlight[64];
uniform int    uShapeItemID[64]; // NEW: For grouping

uniform vec3   uDomainCenter;
uniform vec3   uGlobalTint;
uniform int    uColorBlendMode;

// ... (Lighting constants, Helpers, and Primitives are unchanged) ...
const float AMBIENT    = 0.2;
const float DIFFUSE_M  = 0.8;
const float SPEC_POWER = 32.0;
const float SPEC_SCALE = 0.2;
const vec3  BG_COLOR   = vec3(0.05);
const float PI = 3.14159265359;
const vec3  HIGHLIGHT_COLOR = vec3(1.0, 0.5, 0.0);
vec3 rotateByQuat(vec3 p, vec4 q) { return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p); }
float sdfSphere(vec3 p, float r) { return length(p) - r; }
float sdfBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0); }
float sdfTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }
float sdfCylinder(vec3 p, float h, float r) { vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h); return min(max(d.x,d.y), 0.0) + length(max(d,0.0)); }
float sdfPrism(vec3 p, vec2 h) { vec3 q = abs(p); return max(q.z - h.x, max(q.x*0.866025 + q.y*0.5, q.y) - h.y); }
float sdfCone(vec3 p, float h, float r1, float r2) { p.y -= h * 0.5; vec2 q = vec2(length(p.xz), p.y); vec2 k1 = vec2(r2, h); vec2 k2 = vec2(r2 - r1, 2.0 * h); vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h); vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0); float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0; return s * sqrt(min(dot(ca, ca), dot(cb, cb))); }
float sdfCapsule(vec3 p, vec3 a, vec3 b, float r) { vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }
float opSmoothUnion(float a, float b, float k) { if(k<=0.) return min(a,b); float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b,a,h) - k*h*(1.0 - h); }
float opSmoothSubtract(float a, float b, float k) { if(k<=0.) return max(a,-b); float h = clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0); return mix(a,-b,h) + k*h*(1.0 - h); }
float opSmoothIntersect(float a, float b, float k) { if(k<=0.) return max(a,b); float h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) + k*h*(1.0-h); }
struct SDFRes { float d; vec3 c; int id; };
SDFRes evalPrim(int i, vec3 p) {
    int vi = i >> 2, ci = i & 3;
    int st = (ci==0?uShapeTypePacked[vi].x :ci==1?uShapeTypePacked[vi].y :ci==2?uShapeTypePacked[vi].z :uShapeTypePacked[vi].w);
    if (st < 0) return SDFRes(1e6, vec3(0), -1);
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
    if (st==0) sd = sdfBox(pl, uShapeScale[i] * 0.5); else if (st==1) sd = sdfSphere(pl, uShapeScale[i].x * 0.5); else if (st==2) sd = sdfTorus(pl, vec2(uShapeScale[i].x * 0.4, uShapeScale[i].x * 0.1)); else if (st==3) sd = sdfCylinder(pl, uShapeScale[i].y * 0.5, uShapeScale[i].x * 0.5); else if (st==4) sd = sdfCone(pl, uShapeScale[i].y, uShapeScale[i].x, uShapeScale[i].z); else if (st==5) sd = sdfPrism(pl, uShapeScale[i].xy * 0.5); else if (st==6) { float height = uShapeScale[i].y; float radius = uShapeScale[i].x; vec3 p1 = vec3(0, -height * 0.5, 0); vec3 p2 = vec3(0,  height * 0.5, 0); sd = sdfCapsule(pl, p1, p2, radius); }
    return SDFRes(sd, uShapeColor[i], i);
}

// --- NEW: Combine function for clarity ---
SDFRes combine(SDFRes res1, SDFRes res2, int op, float k) {
    float a = res1.d;
    float b = res2.d;
    float newd = a;
    vec3 newc = res1.c;
    float h = 0.0;

    if (op == 0) { newd = min(a, b); } else if (op == 1) { newd = max(a, -b); } else if (op == 2) { newd = max(a, b); } else if (op == 3) { h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); newd = mix(b, a, h) - k * h * (1.0 - h); } else if (op == 4) { h = clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0); newd = mix(a, -b, h) + k * h * (1.0 - h); } else if (op == 5) { h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0); newd = mix(b, a, h) + k * h * (1.0 - h); }
    
    if (uColorBlendMode == 1) { if (op == 3 || op == 5) { newc = mix(res1.c, res2.c, 1.0 - h); } else if (op == 4) { newc = mix(res1.c, res2.c, h); } else { if (b < a) { newc = res2.c; } }
    } else { bool contributed = false; if (op == 0 || op == 3) { contributed = (b < a); } else if (op == 2 || op == 5) { contributed = (b > a); } else { contributed = (b < 0.0); } if (contributed) { newc = res2.c; } }
    
    SDFRes final_res;
    final_res.d = newd;
    final_res.c = newc;
    final_res.id = (newd < res1.d) ? res2.id : res1.id;
    return final_res;
}

// --- DEFINITIVE sceneSDFColor with Grouping Logic ---
SDFRes sceneSDFColor(vec3 p) {
    SDFRes scene_acc; scene_acc.d = 1e6; scene_acc.c = vec3(1.0); scene_acc.id = -1;
    SDFRes group_acc; group_acc.d = 1e6; group_acc.c = vec3(1.0); group_acc.id = -1;
    int last_itemID = -1;

    if (uCount == 0) return scene_acc;

    for (int i = 0; i < uCount; ++i) {
        SDFRes pr = evalPrim(i, p);
        int current_itemID = uShapeItemID[i];

        if (current_itemID != last_itemID && last_itemID != -1) {
            int final_op = uShapeOp[i-1];
            float final_blend = uShapeBlend[i-1];
            scene_acc = combine(scene_acc, group_acc, final_op, final_blend);
            group_acc = pr;
        } else {
            group_acc = combine(group_acc, pr, 0, 0.0); // Always Hard Union inside a group
        }
        last_itemID = current_itemID;
    }
    
    // Combine the very last group with the scene
    int final_op = uShapeOp[uCount-1];
    float final_blend = uShapeBlend[uCount-1];
    scene_acc = combine(scene_acc, group_acc, final_op, final_blend);

    return scene_acc;
}

// ... (getNormal and main functions are unchanged) ...
float sceneSDF(vec3 p) { return sceneSDFColor(p).d; }
vec3 getNormal(vec3 p) { float e = 0.0005; vec2 h = vec2(e, 0); return normalize(vec3( sceneSDF(p + h.xyy) - sceneSDF(p - h.xyy), sceneSDF(p + h.yxy) - sceneSDF(p - h.yxy), sceneSDF(p + h.yyx) - sceneSDF(p - h.yyx) )); }
void main() {
    vec2 uvn  = (gl_FragCoord.xy / viewportSize)*2.0 - 1.0;
    vec4 clip = vec4(uvn, -1.0, 1.0);
    vec4 eye  = projMatrixInv * clip; eye /= eye.w;
    vec3 ro   = (viewMatrixInv * vec4(0,0,0,1)).xyz;
    vec3 rd   = normalize((viewMatrixInv * eye).xyz - ro);
    float t = 0.0;
    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd * t; 
        SDFRes res = sceneSDFColor(p);
        float dist = res.d;
        if (dist < 0.001) {
            vec3 N = getNormal(p); vec3 L = normalize(uLightDir); vec3 V = normalize(-rd); vec3 R = reflect(-L, N);
            float diff = max(dot(N, L), 0.0); float spec = pow(max(dot(R, V), 0.0), SPEC_POWER) * SPEC_SCALE;
            vec3 base = res.c * uGlobalTint; vec3 col = base * (AMBIENT + DIFFUSE_M * diff) + spec;
            if (res.id >= 0 && uShapeHighlight[res.id] == 1) { float rim = 1.0 - max(dot(N, V), 0.0); float highlight = smoothstep(0.0, 1.0, pow(rim, 1.5)); col += HIGHLIGHT_COLOR * highlight; }
            fragColor = vec4(col, 1.0); return;
        }
        t += dist; if (t > 200.0) break;
    }
    fragColor = vec4(BG_COLOR, 1.0);
}