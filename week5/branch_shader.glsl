// GLES2 / GLSL ES 1.00
// -*- mode: c -*-

#extension GL_OES_standard_derivatives : enable
precision mediump float;

varying vec3 v_worldPos;
varying vec3 v_normal;
varying float v_radius01; // 1 = trunk, 0 = small branch (optional)
varying vec2 v_uv;

uniform vec3 u_lightDir;   // normalized, world space (direction TO light)
uniform vec3 u_camPos;     // optional, for subtle fresnel


// Artistic controls
uniform float u_barkScale;     // e.g. 0.25
uniform float u_detailScale;   // e.g. 2.0
uniform float u_crackStrength; // e.g. 0.7
uniform float u_normalStrength;// e.g. 0.35

// -------------------- tiny hash/noise --------------------
float hash31(vec3 p) {
  // cheap-ish hash for GLES2
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float noise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  // smoothstep curve
  f = f * f * (3.0 - 2.0 * f);

  float n000 = hash31(i + vec3(0,0,0));
  float n100 = hash31(i + vec3(1,0,0));
  float n010 = hash31(i + vec3(0,1,0));
  float n110 = hash31(i + vec3(1,1,0));
  float n001 = hash31(i + vec3(0,0,1));
  float n101 = hash31(i + vec3(1,0,1));
  float n011 = hash31(i + vec3(0,1,1));
  float n111 = hash31(i + vec3(1,1,1));

  float nx00 = mix(n000, n100, f.x);
  float nx10 = mix(n010, n110, f.x);
  float nx01 = mix(n001, n101, f.x);
  float nx11 = mix(n011, n111, f.x);

  float nxy0 = mix(nx00, nx10, f.y);
  float nxy1 = mix(nx01, nx11, f.y);

  return mix(nxy0, nxy1, f.z);
}

float fbm(vec3 p) {
  float s = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    s += a * noise3(p);
    p *= 2.02;
    a *= 0.5;
  }
  return s;
}

// Ridged noise for “cracks”
float ridged(vec3 p) {
  float n = 0.0;
  float a = 0.55;
  for (int i = 0; i < 4; i++) {
    float v = noise3(p);
    v = 1.0 - abs(2.0 * v - 1.0); // ridge
    n += a * v;
    p *= 2.1;
    a *= 0.5;
  }
  return n;
}

// -------------------- triplanar helpers --------------------
vec3 triplanarWeights(vec3 n) {
  vec3 an = abs(n);
  // sharpen to reduce mushy blending
  an = pow(an, vec3(4.0));
  return an / (an.x + an.y + an.z + 1e-5);
}

// Use planar coords for each axis
vec2 uvX(vec3 p) { return p.zy; } // projection onto YZ
vec2 uvY(vec3 p) { return p.xz; } // projection onto XZ
vec2 uvZ(vec3 p) { return p.xy; } // projection onto XY

// -------------------- main bark function --------------------
void main() {
  vec3 N = normalize(v_normal);

  // Height 0..1 for trunk darkening / lichen
  float u_treeMaxY = 30.0;
  float u_treeMinY = 0.0;
  float h = (v_worldPos.y) / max(0.001, (u_treeMaxY - u_treeMinY));
  h = clamp(h, 0.0, 1.0);

  // Thickness (if not provided, treat as trunk-ish)
  float thick = clamp(v_radius01, 0.0, 1.0);

  // Triplanar weights
  vec3 w = triplanarWeights(N);

  // World-space bark coordinates: scale changes with thickness so small branches don’t look too “wide”
  float scale = mix(u_detailScale, u_barkScale, thick);
  vec3 p = v_worldPos * scale;

  // Per-axis procedural “albedo” noise via planar coords
  float nX = fbm(vec3(uvX(p), 7.1));
  float nY = fbm(vec3(uvY(p), 3.7));
  float nZ = fbm(vec3(uvZ(p), 1.9));
  float baseNoise = nX * w.x + nY * w.y + nZ * w.z;

  // Cracks: more pronounced on thick trunk, fade out on twigs
  vec3 pc = v_worldPos * (scale * 0.6);
  float cX = ridged(vec3(uvX(pc), 11.0));
  float cY = ridged(vec3(uvY(pc), 13.0));
  float cZ = ridged(vec3(uvZ(pc), 17.0));
  float cracks = (cX * w.x + cY * w.y + cZ * w.z);
  cracks = pow(cracks, 2.2);
  cracks *= mix(0.2, 1.0, thick) * u_crackStrength;

  // Base bark colors (oak-ish): darker at base, slightly lighter up high
  vec3 barkDark  = vec3(0.18, 0.14, 0.10);
  vec3 barkLight = vec3(0.32, 0.27, 0.22);
  vec3 bark = mix(barkDark, barkLight, 0.45 + 0.35 * h);

  // Add subtle variation
  bark *= 0.85 + 0.30 * baseNoise;
  bark *= 0.85;
  bark += 0.30 *baseNoise;

  // Carve cracks darker
  bark *= (1.0 - 0.55 * cracks);

  // Optional: faint lichen near base (and on flatter/up-facing-ish surfaces)
  float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
  float lichenMask = (1.0 - h);                // more near bottom
  lichenMask *= smoothstep(0.3, 0.9, up);      // prefers upward-ish normals
  lichenMask *= smoothstep(0.35, 1.0, thick);  // mostly trunk/large limbs
  float lichenNoise = fbm(v_worldPos * 0.8);
  lichenMask *= smoothstep(0.55, 0.85, lichenNoise);

  // Generate a lichen color based on noise, ranging from yellow to green to orange
  vec3 baseColor1 = vec3(0.6, 0.8, 0.2); // yellow-green
  vec3 baseColor2 = vec3(0.9, 0.6, 0.1); // orange
  float colorNoise = fbm(v_worldPos * 1.2 + 100.0); // Offset noise for color variation
  vec3 lichenCol = mix(baseColor1, baseColor2, colorNoise);

  bark = mix(bark, lichenCol, lichenMask);

  // --- Fake bump from cracks (cheap normal perturbation) ---
  // Approximate gradient using screen-space derivatives (OES_standard_derivatives).
  // If you can enable it, this is very handy. If not, just skip this block.
  #ifdef GL_OES_standard_derivatives
    // NOTE: you must enable the extension in your shader header if your pipeline requires it:
    // #extension GL_OES_standard_derivatives : enable
    float c = cracks;
    vec3 dx = dFdx(v_worldPos);
    vec3 dy = dFdy(v_worldPos);
    // approximate bump direction: push normal away from increasing crack intensity
    float dcx = dFdx(c);
    float dcy = dFdy(c);
    vec3 bump = normalize(N - u_normalStrength * (dcx * normalize(dx) + dcy * normalize(dy)));
    N = bump;
  #endif

  vec2 grainUV = vec2(v_uv.x, v_uv.y * 20.0); // stretch in y direction
  float g = fbm(vec3(grainUV, 0.0));
  float low = fbm(vec3(v_uv.x * 2.0, v_uv.y * 2.0, 0.0));
  float warp = (low - 0.5) * 0.15;
  grainUV = vec2(v_uv.x, (v_uv.y + warp) * 30.0);
  float grain = fbm(vec3(grainUV, 0.5));
  float bands = sin((v_uv.y + warp) * 80.0 + v_uv.x * 2.0);
  float lines = smoothstep(0.2, 0.8, bands*0.5 + 0.5);

  bark *= 0.85 + 0.30 * grain;
  vec3 bumpN = normalize(vec3(dFdx(lines), dFdy(lines), 1.0));
  N = normalize(N + bumpN * u_normalStrength);

  // --- Lighting (simple) ---
  vec3 L = normalize(u_lightDir);
  float ndl = max(0.0, dot(N, L));

  vec3 ambient = bark * 0.35;
  vec3 diffuse = bark * (0.75 * ndl);

  // Subtle fresnel darkening (helps silhouette)
  vec3 V = normalize(u_camPos - v_worldPos);
  float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
  vec3 color = ambient + diffuse;
  //  color *= (1.0 - 0.12 * fres);

  gl_FragColor = vec4(color, 1.0);
}
