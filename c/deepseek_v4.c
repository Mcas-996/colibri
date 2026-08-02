/* DeepSeek-V4-Flash CPU reference engine.
 *
 * The runtime side of this backend is intentionally boring: the converter
 * stores ordinary Colibri int4/int8 tensors and keeps routed experts in the
 * official MXFP4 (E2M1 + UE8M0/g32) representation.  Dense tensors are loaded
 * once, while an expert's three matrices are read and released when it is
 * selected.  That keeps the resident working set suitable for a 15 GB host;
 * the model files themselves remain on SSD.
 *
 * This is the first V4 backend.  It is a single-sequence, CLI-oriented path
 * (the Python launcher supplies PROMPT/SNAP/NGEN) and deliberately does not
 * load the optional DSpark/MTP sidecar.  The forward pass follows the public
 * reference implementation: MLA, learned KV compression, deterministic
 * sparse top-k compressed attention, SwiGLU MoE, and Hyper-Connections.  The
 * optional ratio-4 learned indexer is retained by the converter but is not
 * resident in this first 15-GB CPU path; its top-k is approximated with the
 * main attention query against compressed keys.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include "st.h"
#include "tok.h"
#include "quant.h"

typedef struct {
    int vocab, dim, n_layers;
    int n_heads, head_dim, rope_dim;
    int q_lora, o_groups, o_lora;
    int n_routed, n_shared, n_active, moe_inter, n_hash;
    int window, index_topk, index_heads, index_head_dim;
    int hc_mult, hc_iters;
    int max_ctx;
    float norm_eps, hc_eps, route_scale, swiglu_limit;
    float rope_theta, compress_rope_theta, rope_factor;
    int original_seq_len, beta_fast, beta_slow;
    int *compress_ratios;
    int eos[8], n_eos;
} V4Cfg;

typedef struct {
    int fmt;                    /* 0=f32, 1=int8, 4=int4-gN, 7=MXFP4 */
    int O, I, gs;
    float *f;
    void *q;
    float *scale;
    uint8_t *mscale;
} V4W;

typedef struct {
    float *local;                /* [window, head_dim], ring ordered by pos */
    int *local_pos;
    float *compressed;            /* [ceil(ctx/ratio), head_dim] */
    int *compressed_pos;
    int compressed_n, compressed_cap;
    int ratio, pending_n;
    float *pending, *pending_score;
    float *overlap, *overlap_score;
    float *next_overlap, *next_overlap_score;
} V4KV;

typedef struct {
    V4W wq_a, wq_b, wkv, wo_a, wo_b;
    float *q_norm, *kv_norm, *sink;

    V4W compressor_wkv, compressor_wgate;
    float *compressor_ape, *compressor_norm;
    int compressor_coff;

    V4W gate;
    float *gate_bias, *tid2eid;
    V4W shared_w1, shared_w2, shared_w3;

    float *attn_norm, *ffn_norm;
    float *hc_attn_fn, *hc_attn_base, *hc_attn_scale;
    float *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    int hc_ready;
    V4KV kv;
} V4Layer;

typedef struct {
    V4Cfg c;
    shards S;
    V4W embed, head;
    float *final_norm;
    float *hc_head_fn, *hc_head_base, *hc_head_scale;
    int hc_head_ready;
    V4Layer *layers;
} V4Model;

static int g_v4_drop_expert = 0;

static float *v4_alloc(int64_t n) {
    if (n <= 0) return NULL;
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "DeepSeek-V4: OOM for %lld floats\n", (long long)n); exit(1); }
    return p;
}

static void *v4_bytes(int64_t n) {
    if (n <= 0) return NULL;
    void *p = malloc((size_t)n);
    if (!p) { fprintf(stderr, "DeepSeek-V4: OOM for %lld bytes\n", (long long)n); exit(1); }
    return p;
}

static double v4_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static jval *v4_json_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "DeepSeek-V4: cannot open %s: %s\n", path, strerror(errno)); exit(1); }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); exit(1); }
    long n = ftell(f);
    if (n <= 0 || n > (64L << 20)) { fprintf(stderr, "%s: invalid config size\n", path); exit(1); }
    rewind(f);
    char *buf = (char *)v4_bytes((int64_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    fclose(f); buf[n] = 0;
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    free(buf);
    (void)arena;
    if (!root || root->t != J_OBJ) { fprintf(stderr, "%s: invalid JSON\n", path); exit(1); }
    return root;
}

static int v4_int(jval *o, const char *key, int fallback) {
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? (int)v->num : fallback;
}

static float v4_float(jval *o, const char *key, float fallback) {
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? (float)v->num : fallback;
}

static int v4_int_any(jval *o, const char *a, const char *b, int fallback) {
    int x = v4_int(o, a, fallback);
    return x == fallback ? v4_int(o, b, fallback) : x;
}

static void v4_load_config(V4Cfg *c, const char *snap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    jval *r = v4_json_file(path);
    memset(c, 0, sizeof(*c));
    c->vocab = v4_int_any(r, "vocab_size", "vocab", 129280);
    c->dim = v4_int_any(r, "dim", "hidden_size", 4096);
    c->n_layers = v4_int_any(r, "n_layers", "num_hidden_layers", 43);
    c->n_heads = v4_int_any(r, "n_heads", "num_attention_heads", 64);
    c->head_dim = v4_int(r, "head_dim", 512);
    c->rope_dim = v4_int_any(r, "rope_head_dim", "qk_rope_head_dim", 64);
    c->q_lora = v4_int(r, "q_lora_rank", 1024);
    c->o_groups = v4_int(r, "o_groups", 8);
    c->o_lora = v4_int(r, "o_lora_rank", 1024);
    c->n_routed = v4_int(r, "n_routed_experts", 256);
    c->n_shared = v4_int(r, "n_shared_experts", 1);
    c->n_active = v4_int(r, "n_activated_experts", 6);
    c->moe_inter = v4_int_any(r, "moe_inter_dim", "moe_intermediate_size", 2048);
    c->n_hash = v4_int(r, "n_hash_layers", 3);
    c->window = v4_int(r, "window_size", 128);
    c->index_topk = v4_int(r, "index_topk", 512);
    c->index_heads = v4_int(r, "index_n_heads", 64);
    c->index_head_dim = v4_int(r, "index_head_dim", 128);
    c->hc_mult = v4_int(r, "hc_mult", 4);
    c->hc_iters = v4_int(r, "hc_sinkhorn_iters", 20);
    c->norm_eps = v4_float(r, "norm_eps", v4_float(r, "norm_eps", 1e-6f));
    c->hc_eps = v4_float(r, "hc_eps", 1e-6f);
    c->route_scale = v4_float(r, "route_scale", 1.5f);
    c->swiglu_limit = v4_float(r, "swiglu_limit", 10.f);
    c->rope_theta = v4_float(r, "rope_theta", 10000.f);
    c->compress_rope_theta = v4_float(r, "compress_rope_theta", 160000.f);
    c->rope_factor = v4_float(r, "rope_factor", 16.f);
    c->original_seq_len = v4_int(r, "original_seq_len", 65536);
    c->beta_fast = v4_int(r, "beta_fast", 32);
    c->beta_slow = v4_int(r, "beta_slow", 1);
    c->max_ctx = 50000;
    jval *cr = json_get(r, "compress_ratios");
    c->compress_ratios = (int *)calloc((size_t)c->n_layers, sizeof(int));
    if (!c->compress_ratios) { fprintf(stderr, "DeepSeek-V4: OOM compression ratios\n"); exit(1); }
    for (int i = 0; i < c->n_layers; i++)
        c->compress_ratios[i] = (cr && cr->t == J_ARR && i < cr->len) ? (int)cr->kids[i]->num : (i < 2 ? 0 : 4);
    jval *e = json_get(r, "eos_token_id");
    if (e && e->t == J_NUM) c->eos[c->n_eos++] = (int)e->num;
    else if (e && e->t == J_ARR) for (int i = 0; i < e->len && c->n_eos < 8; i++)
        if (e->kids[i]->t == J_NUM) c->eos[c->n_eos++] = (int)e->kids[i]->num;
    if (c->vocab < 1 || c->vocab > (1 << 22) || c->dim < 1 || c->dim > 65536 ||
        c->n_layers < 1 || c->n_layers > 128 || c->n_heads < 1 || c->head_dim < 2 ||
        c->rope_dim < 2 || c->rope_dim > c->head_dim || c->n_active > c->n_routed ||
        c->hc_mult < 1 || c->hc_mult > 8 || c->window < 1 || c->window > 4096) {
        fprintf(stderr, "DeepSeek-V4: unsupported or corrupt config dimensions\n"); exit(1);
    }
}

static void v4_wfree(V4W *w) {
    if (!w) return;
    free(w->f); free(w->q); free(w->scale); free(w->mscale);
    memset(w, 0, sizeof(*w));
}

static void v4_missing(shards *s, const char *name, int optional) {
    if (!optional) st_die_missing(s, name);
}

static float *v4_f32_load(V4Model *m, const char *name, int64_t n, int optional) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { v4_missing(&m->S, name, optional); return NULL; }
    if (t->numel != n) {
        fprintf(stderr, "DeepSeek-V4: %s has %lld values, expected %lld\n", name,
                (long long)t->numel, (long long)n); exit(1);
    }
    float *out = v4_alloc(n);
    st_read_f32(&m->S, name, out, 0);
    return out;
}

static int v4_scale_groups(const st_tensor *t, int O, int I) {
    if (!t || t->numel <= 0 || t->numel % O) return 0;
    int ng = (int)(t->numel / O);
    if (ng <= 0 || ng > I) return 0;
    /* The converter pads the last input group when I is not a multiple of
     * --group.  The common V4 shapes produce exactly 64 here, but ceil keeps
     * the container valid for small regression fixtures too. */
    return (I + ng - 1) / ng;
}

static void v4_wload(V4Model *m, V4W *w, const char *name, int O, int I, int optional) {
    memset(w, 0, sizeof(*w)); w->O = O; w->I = I;
    st_tensor *t = st_find(&m->S, name);
    if (!t) { v4_missing(&m->S, name, optional); return; }
    if (t->numel == (int64_t)O * I && t->dtype != 3) {
        w->fmt = 0; w->f = v4_alloc((int64_t)O * I);
        st_read_f32(&m->S, name, w->f, 0); return;
    }
    if (t->dtype != 3) {
        fprintf(stderr, "DeepSeek-V4: %s is not a supported matrix dtype\n", name); exit(1);
    }
    char sn[640]; snprintf(sn, sizeof(sn), "%s.scale", name);
    st_tensor *s = st_find(&m->S, sn);
    if (!s) {
        /* Accept the compact converter spelling: foo.scale for foo.weight. */
        const char *suffix = ".weight";
        size_t n = strlen(name), slen = strlen(suffix);
        if (n > slen && strcmp(name + n - slen, suffix) == 0) {
            snprintf(sn, sizeof(sn), "%.*s.scale", (int)(n - slen), name);
            s = st_find(&m->S, sn);
        }
    }
    if (!s) { fprintf(stderr, "DeepSeek-V4: %s has no scale sidecar\n", name); exit(1); }
    int64_t int4_bytes = (int64_t)O * ((I + 1) / 2);
    if (t->nbytes == (int64_t)O * I && s->numel == O) {
        w->fmt = 1; w->q = v4_bytes((int64_t)O * I); w->scale = v4_alloc(O);
        st_read_raw(&m->S, name, w->q, 0); st_read_f32(&m->S, sn, w->scale, 0); return;
    }
    if (t->nbytes == int4_bytes && s->dtype == 3 &&
        s->numel == (int64_t)O * ((I + 31) / 32)) {
        w->fmt = 7; w->q = v4_bytes(int4_bytes); w->mscale = (uint8_t *)v4_bytes(s->nbytes);
        int drop = g_v4_drop_expert && strstr(name, ".experts.") != NULL;
        st_read_raw(&m->S, name, w->q, drop); st_read_raw(&m->S, sn, w->mscale, drop); return;
    }
    int gs = v4_scale_groups(s, O, I);
    if (t->nbytes == int4_bytes && gs > 0 && s->dtype != 3) {
        w->fmt = 4; w->gs = gs; w->q = v4_bytes(int4_bytes);
        w->scale = v4_alloc(s->numel);
        st_read_raw(&m->S, name, w->q, 0); st_read_f32(&m->S, sn, w->scale, 0); return;
    }
    fprintf(stderr, "DeepSeek-V4: %s has unsupported byte/scale layout (%lld/%lld)\n", name,
            (long long)t->nbytes, (long long)s->numel); exit(1);
}

static float v4_wdot(const V4W *w, int row, const float *x) {
    float a = 0.f;
    if (w->fmt == 0) {
        const float *p = w->f + (int64_t)row * w->I;
        for (int i = 0; i < w->I; i++) a += p[i] * x[i];
        return a;
    }
    if (w->fmt == 1) {
        const int8_t *p = (const int8_t *)w->q + (int64_t)row * w->I;
        for (int i = 0; i < w->I; i++) a += (float)p[i] * x[i];
        return a * w->scale[row];
    }
    if (w->fmt == 4) {
        int rb = (w->I + 1) / 2, ng = (w->I + w->gs - 1) / w->gs;
        const uint8_t *p = (const uint8_t *)w->q + (int64_t)row * rb;
        const float *sc = w->scale + (int64_t)row * ng;
        for (int g = 0; g < ng; g++) {
            int beg = g * w->gs, end = beg + w->gs; if (end > w->I) end = w->I;
            float z = 0.f;
            for (int i = beg; i < end; i += 2) {
                uint8_t b = p[i >> 1]; z += x[i] * (float)((int)(b & 15) - 8);
                if (i + 1 < end) z += x[i + 1] * (float)((int)(b >> 4) - 8);
            }
            a += z * sc[g];
        }
        return a;
    }
    if (w->fmt == 7) {
        int rb = (w->I + 1) / 2, ng = (w->I + 31) / 32;
        const uint8_t *p = (const uint8_t *)w->q + (int64_t)row * rb;
        const uint8_t *sc = w->mscale + (int64_t)row * ng;
        for (int g = 0; g < ng; g++) {
            int beg = g * 32, end = beg + 32; if (end > w->I) end = w->I;
            float z = 0.f, scale = mx4_scale(sc[g]);
            for (int i = beg; i < end; i += 2) {
                uint8_t b = p[i >> 1]; z += x[i] * mx4_lut[b & 15];
                if (i + 1 < end) z += x[i + 1] * mx4_lut[b >> 4];
            }
            a += z * scale;
        }
    }
    return a;
}

static void v4_wmatvec(float *y, const float *x, const V4W *w) {
    if (w->fmt == 0) matmul(y, x, w->f, 1, w->I, w->O);
    else if (w->fmt == 1) matmul_q(y, x, (const int8_t *)w->q, w->scale, 1, w->I, w->O);
    else if (w->fmt == 4) matmul_i4_grouped(y, x, (const uint8_t *)w->q, w->scale, 1, w->I, w->O, w->gs);
    else if (w->fmt == 7) matmul_mxfp4(y, x, (const uint8_t *)w->q, w->mscale, 1, w->I, w->O);
    else { fprintf(stderr, "DeepSeek-V4: invalid matrix format\n"); exit(1); }
}

static void v4_wmatvec_rows(float *y, const float *x, const V4W *w, int first, int rows) {
    if (first == 0 && rows == w->O) { v4_wmatvec(y, x, w); return; }
    if (w->fmt == 0) { matmul(y, x, w->f + (int64_t)first * w->I, 1, w->I, rows); return; }
    if (w->fmt == 1) { matmul_q(y, x, (const int8_t *)w->q + (int64_t)first * w->I,
                                  w->scale + first, 1, w->I, rows); return; }
    if (w->fmt == 4) {
        int rb = (w->I + 1) / 2, ng = (w->I + w->gs - 1) / w->gs;
        matmul_i4_grouped(y, x, (const uint8_t *)w->q + (int64_t)first * rb,
                          w->scale + (int64_t)first * ng, 1, w->I, rows, w->gs); return;
    }
    if (w->fmt == 7) {
        int rb = (w->I + 1) / 2, ng = (w->I + 31) / 32;
        matmul_mxfp4(y, x, (const uint8_t *)w->q + (int64_t)first * rb,
                     w->mscale + (int64_t)first * ng, 1, w->I, rows); return;
    }
}

static void v4_rms(float *out, const float *x, const float *w, int n, float eps) {
    double s = 0.0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(s / n) + eps);
    if (w) for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
    else for (int i = 0; i < n; i++) out[i] = x[i] * r;
}

static float v4_sigmoid(float x) {
    if (x >= 0.f) { float e = expf(-x); return 1.f / (1.f + e); }
    float e = expf(x); return e / (1.f + e);
}

static void v4_rope(float *v, int n, int pos, float theta, int compressed, const V4Cfg *c, int inverse) {
    int half = n / 2;
    for (int j = 0; j < half; j++) {
        float freq = powf(theta, -2.f * (float)j / (float)n);
        if (compressed && c->original_seq_len > 0) {
            float low = floorf((float)half * logf((float)c->original_seq_len /
                        (2.f * (float)c->beta_fast * (float)M_PI)) / (2.f * logf(theta)));
            float high = ceilf((float)half * logf((float)c->original_seq_len /
                         (2.f * (float)c->beta_slow * (float)M_PI)) / (2.f * logf(theta)));
            if (low < 0.f) low = 0.f; if (high > (float)(half - 1)) high = (float)(half - 1);
            if (high < low) { float t = low; low = high; high = t; }
            float ramp = high == low ? (j >= high ? 1.f : 0.f) : ((float)j - low) / (high - low);
            if (ramp < 0.f) ramp = 0.f; if (ramp > 1.f) ramp = 1.f;
            freq = freq * (1.f - ramp) + freq / c->rope_factor * ramp;
        }
        float a = (float)pos * freq; if (inverse) a = -a;
        float cs = cosf(a), sn = sinf(a);
        float x = v[2 * j], y = v[2 * j + 1];
        v[2 * j] = x * cs - y * sn; v[2 * j + 1] = x * sn + y * cs;
    }
}

static int v4_mix_hc(const V4Cfg *c, const float *h, const float *fn, const float *base,
                     const float *scale, float *x, float *post, float *comb) {
    int H = c->hc_mult, D = c->dim, M = (H + 2) * H;
    if (!fn || !base || !scale) {
        for (int d = 0; d < D; d++) { x[d] = 0.f; for (int j = 0; j < H; j++) x[d] += h[(int64_t)j * D + d] / H; }
        for (int j = 0; j < H; j++) post[j] = 1.f, comb[j * H + j] = 0.f;
        return 0;
    }
    double ss = 0.0; for (int j = 0; j < H * D; j++) ss += (double)h[j] * h[j];
    float inv = 1.f / sqrtf((float)(ss / (H * D)) + c->norm_eps);
    float *mix = (float *)alloca((size_t)M * sizeof(float));
    memset(x, 0, (size_t)D * sizeof(float));
    for (int j = 0; j < M; j++) {
        double z = 0.0; const float *row = fn + (int64_t)j * H * D;
        for (int k = 0; k < H * D; k++) z += (double)row[k] * h[k];
        mix[j] = (float)z * inv;
    }
    for (int j = 0; j < H; j++) {
        float q = v4_sigmoid(mix[j] * scale[0] + base[j]) + c->hc_eps;
        for (int d = 0; d < D; d++) x[d] += q * h[(int64_t)j * D + d];
        post[j] = 2.f * v4_sigmoid(mix[H + j] * scale[1] + base[H + j]);
    }
    for (int j = 0; j < H; j++) {
        float mx = -INFINITY;
        for (int k = 0; k < H; k++) { float z = mix[2 * H + j * H + k] * scale[2] + base[2 * H + j * H + k]; if (z > mx) mx = z; }
        float sum = 0.f;
        for (int k = 0; k < H; k++) {
            float z = mix[2 * H + j * H + k] * scale[2] + base[2 * H + j * H + k];
            comb[j * H + k] = expf(z - mx) + c->hc_eps; sum += comb[j * H + k];
        }
        for (int k = 0; k < H; k++) comb[j * H + k] /= sum;
    }
    /* Sinkhorn: the reference first normalizes columns, then alternates rows
     * and columns.  H is four, so this small dense loop is cheap. */
    for (int k = 0; k < H; k++) { float s = 0.f; for (int j = 0; j < H; j++) s += comb[j * H + k]; if (s > 0) for (int j = 0; j < H; j++) comb[j * H + k] /= s; }
    for (int it = 1; it < c->hc_iters; it++) {
        for (int j = 0; j < H; j++) { float s = 0.f; for (int k = 0; k < H; k++) s += comb[j * H + k]; if (s > 0) for (int k = 0; k < H; k++) comb[j * H + k] /= s; }
        for (int k = 0; k < H; k++) { float s = 0.f; for (int j = 0; j < H; j++) s += comb[j * H + k]; if (s > 0) for (int j = 0; j < H; j++) comb[j * H + k] /= s; }
    }
    return 1;
}

static void v4_hc_post(const V4Cfg *c, float *h, const float *res, const float *x,
                       const float *post, const float *comb, int ready) {
    int H = c->hc_mult, D = c->dim;
    if (!ready) { memcpy(h, res, (size_t)H * D * sizeof(float)); for (int d = 0; d < D; d++) h[d] += x[d]; return; }
    for (int j = 0; j < H; j++) for (int d = 0; d < D; d++) {
        float z = post[j] * x[d];
        for (int k = 0; k < H; k++) z += comb[j * H + k] * res[(int64_t)k * D + d];
        h[(int64_t)j * D + d] = z;
    }
}

static int v4_cache_init(V4KV *k, int ratio, int window, int dim, int max_ctx) {
    memset(k, 0, sizeof(*k)); k->ratio = ratio;
    k->local = (float *)calloc((size_t)window * dim, sizeof(float));
    k->local_pos = (int *)malloc((size_t)window * sizeof(int));
    if (!k->local || !k->local_pos) { fprintf(stderr, "DeepSeek-V4: OOM KV cache\n"); exit(1); }
    for (int i = 0; i < window; i++) k->local_pos[i] = -1;
    if (ratio > 0) {
        k->compressed_cap = max_ctx / ratio + 4;
        k->compressed = (float *)calloc((size_t)k->compressed_cap * dim, sizeof(float));
        k->compressed_pos = (int *)malloc((size_t)k->compressed_cap * sizeof(int));
        k->pending = (float *)calloc((size_t)ratio * 2 * dim, sizeof(float));
        k->pending_score = (float *)calloc((size_t)ratio * 2 * dim, sizeof(float));
        if (ratio == 4) {
            k->overlap = (float *)calloc((size_t)ratio * dim, sizeof(float));
            k->overlap_score = (float *)malloc((size_t)ratio * dim * sizeof(float));
            k->next_overlap = (float *)calloc((size_t)ratio * dim, sizeof(float));
            k->next_overlap_score = (float *)malloc((size_t)ratio * dim * sizeof(float));
            if (k->overlap_score) for (int i = 0; i < ratio * dim; i++) k->overlap_score[i] = -INFINITY;
            if (k->next_overlap_score) for (int i = 0; i < ratio * dim; i++) k->next_overlap_score[i] = -INFINITY;
        }
        if (!k->compressed || !k->compressed_pos || !k->pending || !k->pending_score ||
            (ratio == 4 && (!k->overlap || !k->overlap_score || !k->next_overlap || !k->next_overlap_score))) {
            fprintf(stderr, "DeepSeek-V4: OOM compressed KV cache\n"); exit(1);
        }
        if (ratio == 4) {
            for (int i = 0; i < ratio * dim; i++) k->pending_score[i] = -INFINITY;
        }
    }
    return 1;
}

typedef struct { float score; int index; } V4Pair;
static int v4_pair_desc(const void *a, const void *b) {
    float x = ((const V4Pair *)a)->score, y = ((const V4Pair *)b)->score;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void v4_add_compressed(V4Model *m, V4Layer *l, int pos, const float *x, const float *fallback) {
    V4KV *k = &l->kv; int D = m->c.head_dim, r = k->ratio;
    int has_compressor = l->compressor_wkv.f || l->compressor_wkv.q;
    int coff = has_compressor && l->compressor_coff == 2 ? 2 : 1;
    int slots = coff == 2 ? r * 2 : r;
    (void)fallback;
    if (k->pending_n < slots) return;
    if (k->compressed_n < k->compressed_cap) {
        float *dst = k->compressed + (int64_t)k->compressed_n * D;
        for (int d = 0; d < D; d++) {
            float mx = -INFINITY, sum = 0.f;
            for (int j = 0; j < slots; j++) {
                float z = k->pending_score[(int64_t)j * D + d]; if (z > mx) mx = z;
            }
            for (int j = 0; j < slots; j++)
                sum += expf(k->pending_score[(int64_t)j * D + d] - mx);
            float z = 0.f;
            for (int j = 0; j < slots; j++)
                z += k->pending[(int64_t)j * D + d] *
                    expf(k->pending_score[(int64_t)j * D + d] - mx);
            dst[d] = sum > 0.f ? z / sum : 0.f;
        }
        if (l->compressor_norm) v4_rms(dst, dst, l->compressor_norm, D, m->c.norm_eps);
        v4_rope(dst + D - m->c.rope_dim, m->c.rope_dim, pos - r + 1,
                m->c.compress_rope_theta, 1, &m->c, 0);
        k->compressed_pos[k->compressed_n++] = pos - r + 1;
    }
    if (coff == 2) {
        memcpy(k->pending, k->next_overlap, (size_t)r * D * sizeof(float));
        memcpy(k->pending_score, k->next_overlap_score, (size_t)r * D * sizeof(float));
    }
    k->pending_n = 0;
    (void)x;
}

static void v4_attention(V4Model *m, V4Layer *l, int li, const float *x, int pos, float *out) {
    const V4Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, rd = c->rope_dim;
    float *qr = v4_alloc(c->q_lora), *q = v4_alloc((int64_t)H * hd);
    float *kv0 = v4_alloc(hd), *kv = v4_alloc(hd), *tmp = v4_alloc((int64_t)H * hd);
    v4_wmatvec(qr, x, &l->wq_a); v4_rms(qr, qr, l->q_norm, c->q_lora, c->norm_eps);
    v4_wmatvec(q, qr, &l->wq_b);
    v4_wmatvec(kv0, x, &l->wkv); v4_rms(kv0, kv0, l->kv_norm, hd, c->norm_eps);
    memcpy(kv, kv0, (size_t)hd * sizeof(float));
    for (int h = 0; h < H; h++) {
        float *qh = q + (int64_t)h * hd;
        double ms = 0.0; for (int d = 0; d < hd; d++) ms += (double)qh[d] * qh[d];
        float r = 1.f / sqrtf((float)(ms / hd) + c->norm_eps);
        for (int d = 0; d < hd; d++) qh[d] *= r;
        v4_rope(qh + hd - rd, rd, pos, c->compress_ratios[li] ? c->compress_rope_theta : c->rope_theta,
                c->compress_ratios[li] != 0, c, 0);
    }
    v4_rope(kv + hd - rd, rd, pos, c->compress_ratios[li] ? c->compress_rope_theta : c->rope_theta,
            c->compress_ratios[li] != 0, c, 0);
    V4KV *cache = &l->kv; int slot = pos % c->window;
    memcpy(cache->local + (int64_t)slot * hd, kv, (size_t)hd * sizeof(float)); cache->local_pos[slot] = pos;
    if (cache->ratio > 0) {
        int r = cache->ratio;
        if (l->compressor_wkv.f || l->compressor_wkv.q) {
            int coff = l->compressor_coff, n = coff * hd;
            float *cv = v4_alloc(n), *cs = v4_alloc(n);
            v4_wmatvec(cv, x, &l->compressor_wkv); v4_wmatvec(cs, x, &l->compressor_wgate);
            int j = cache->pending_n;
            if (j < r) {
                if (coff == 2) {
                    memcpy(cache->next_overlap + (int64_t)j * hd, cv, (size_t)hd * sizeof(float));
                    for (int d = 0; d < hd; d++) cache->next_overlap_score[(int64_t)j * hd + d] =
                        cs[d] + (l->compressor_ape ? l->compressor_ape[(int64_t)j * n + d] : 0.f);
                    memcpy(cache->pending + (int64_t)(r + j) * hd, cv + hd, (size_t)hd * sizeof(float));
                    for (int d = 0; d < hd; d++) cache->pending_score[(int64_t)(r + j) * hd + d] =
                        cs[hd + d] + (l->compressor_ape ? l->compressor_ape[(int64_t)j * n + hd + d] : 0.f);
                } else {
                    memcpy(cache->pending + (int64_t)j * hd, cv, (size_t)hd * sizeof(float));
                    for (int d = 0; d < hd; d++) cache->pending_score[(int64_t)j * hd + d] = cs[d] +
                        (l->compressor_ape ? l->compressor_ape[(int64_t)j * n + d] : 0.f);
                }
                cache->pending_n++;
            }
            if (cache->pending_n == r) v4_add_compressed(m, l, pos, x, cache->pending);
            free(cv); free(cs);
        } else {
            int j = cache->pending_n;
            if (j < r) {
                memcpy(cache->pending + (int64_t)j * hd, kv0, (size_t)hd * sizeof(float));
                for (int d = 0; d < hd; d++) cache->pending_score[(int64_t)j * hd + d] = 0.f;
                cache->pending_n++;
            }
            if (cache->pending_n == r) v4_add_compressed(m, l, pos, x, cache->pending);
        }
    }
    memset(tmp, 0, (size_t)H * hd * sizeof(float));
    V4Pair *pairs = cache->compressed_n ? (V4Pair *)v4_bytes((int64_t)cache->compressed_n * sizeof(V4Pair)) : NULL;
    for (int h = 0; h < H; h++) {
        const float *qh = q + (int64_t)h * hd;
        int np = 0;
        for (int j = 0; j < cache->compressed_n; j++) {
            float s = 0.f; const float *v = cache->compressed + (int64_t)j * hd;
            for (int d = 0; d < hd; d++) s += qh[d] * v[d];
            pairs[np++] = (V4Pair){s / sqrtf((float)hd), j};
        }
        if (cache->ratio == 4 && np > c->index_topk) { qsort(pairs, (size_t)np, sizeof(V4Pair), v4_pair_desc); np = c->index_topk; }
        float mx = l->sink ? l->sink[h] : 0.f;
        for (int j = 0; j < c->window; j++) if (cache->local_pos[j] >= 0 && cache->local_pos[j] <= pos) {
            float s = 0.f; const float *v = cache->local + (int64_t)j * hd;
            for (int d = 0; d < hd; d++) s += qh[d] * v[d]; s /= sqrtf((float)hd); if (s > mx) mx = s;
        }
        for (int j = 0; j < np; j++) if (pairs[j].score > mx) mx = pairs[j].score;
        float den = expf((l->sink ? l->sink[h] : 0.f) - mx);
        for (int j = 0; j < c->window; j++) if (cache->local_pos[j] >= 0 && cache->local_pos[j] <= pos) {
            float s = 0.f; const float *v = cache->local + (int64_t)j * hd;
            for (int d = 0; d < hd; d++) s += qh[d] * v[d]; den += expf(s / sqrtf((float)hd) - mx);
        }
        for (int j = 0; j < np; j++) den += expf(pairs[j].score - mx);
        float *oh = tmp + (int64_t)h * hd;
        for (int j = 0; j < c->window; j++) if (cache->local_pos[j] >= 0 && cache->local_pos[j] <= pos) {
            float s = 0.f; const float *v = cache->local + (int64_t)j * hd;
            for (int d = 0; d < hd; d++) s += qh[d] * v[d]; float a = expf(s / sqrtf((float)hd) - mx) / den;
            for (int d = 0; d < hd; d++) oh[d] += a * v[d];
        }
        for (int j = 0; j < np; j++) {
            const float *v = cache->compressed + (int64_t)pairs[j].index * hd;
            float a = expf(pairs[j].score - mx) / den;
            for (int d = 0; d < hd; d++) oh[d] += a * v[d];
        }
        v4_rope(oh + hd - rd, rd, pos, c->compress_ratios[li] ? c->compress_rope_theta : c->rope_theta,
                c->compress_ratios[li] != 0, c, 1);
    }
    float *groups = v4_alloc((int64_t)c->o_groups * c->o_lora), *gout = v4_alloc(c->o_lora);
    int heads_per_group = H / c->o_groups;
    for (int g = 0; g < c->o_groups; g++) {
        v4_wmatvec_rows(gout, tmp + (int64_t)g * heads_per_group * hd, &l->wo_a,
                        g * c->o_lora, c->o_lora);
        memcpy(groups + (int64_t)g * c->o_lora, gout, (size_t)c->o_lora * sizeof(float));
    }
    v4_wmatvec(out, groups, &l->wo_b);
    free(qr); free(q); free(kv0); free(kv); free(tmp); free(groups); free(gout); free(pairs);
}

static float v4_softplus_sqrt(float x) {
    float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(sp > 0.f ? sp : 0.f);
}

static void v4_expert(V4Model *m, int li, int eid, const float *x, float route, float *out) {
    char n1[256], n2[256], n3[256];
    snprintf(n1, sizeof(n1), "layers.%d.ffn.experts.%d.w1.weight", li, eid);
    snprintf(n2, sizeof(n2), "layers.%d.ffn.experts.%d.w2.weight", li, eid);
    snprintf(n3, sizeof(n3), "layers.%d.ffn.experts.%d.w3.weight", li, eid);
    V4W w1, w2, w3;
    v4_wload(m, &w1, n1, m->c.moe_inter, m->c.dim, 0);
    v4_wload(m, &w2, n2, m->c.dim, m->c.moe_inter, 0);
    v4_wload(m, &w3, n3, m->c.moe_inter, m->c.dim, 0);
    float *gate = v4_alloc(m->c.moe_inter), *up = v4_alloc(m->c.moe_inter);
    v4_wmatvec(gate, x, &w1); v4_wmatvec(up, x, &w3);
    for (int i = 0; i < m->c.moe_inter; i++) {
        if (gate[i] > m->c.swiglu_limit) gate[i] = m->c.swiglu_limit;
        if (up[i] > m->c.swiglu_limit) up[i] = m->c.swiglu_limit;
        if (up[i] < -m->c.swiglu_limit) up[i] = -m->c.swiglu_limit;
        gate[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i] * route;
    }
    v4_wmatvec(out, gate, &w2);
    v4_wfree(&w1); v4_wfree(&w2); v4_wfree(&w3); free(gate); free(up);
}

static void v4_shared(V4Model *m, V4Layer *l, const float *x, float *out) {
    int I = m->c.moe_inter; float *g = v4_alloc(I), *u = v4_alloc(I);
    v4_wmatvec(g, x, &l->shared_w1); v4_wmatvec(u, x, &l->shared_w3);
    for (int i = 0; i < I; i++) {
        if (g[i] > m->c.swiglu_limit) g[i] = m->c.swiglu_limit;
        if (u[i] > m->c.swiglu_limit) u[i] = m->c.swiglu_limit;
        if (u[i] < -m->c.swiglu_limit) u[i] = -m->c.swiglu_limit;
        g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
    }
    v4_wmatvec(out, g, &l->shared_w2); free(g); free(u);
}

static void v4_moe(V4Model *m, V4Layer *l, int li, const float *x, int token, float *out) {
    int E = m->c.n_routed, K = m->c.n_active, D = m->c.dim;
    float *original = v4_alloc(E), *select = v4_alloc(E), *sel_score = v4_alloc(K);
    int *sel = (int *)v4_bytes((int64_t)K * sizeof(int));
    v4_wmatvec(original, x, &l->gate);
    for (int e = 0; e < E; e++) original[e] = v4_softplus_sqrt(original[e]);
    memcpy(select, original, (size_t)E * sizeof(float));
    if (l->gate_bias) for (int e = 0; e < E; e++) select[e] += l->gate_bias[e];
    if (li < m->c.n_hash && l->tid2eid) {
        int base = token * K;
        for (int k = 0; k < K; k++) sel[k] = (int)lroundf(l->tid2eid[base + k]);
    } else {
        for (int k = 0; k < K; k++) {
            int best = 0; float bv = -INFINITY;
            for (int e = 0; e < E; e++) {
                int used = 0; for (int j = 0; j < k; j++) if (sel[j] == e) used = 1;
                if (!used && select[e] > bv) bv = select[e], best = e;
            }
            sel[k] = best;
        }
    }
    float sum = 0.f;
    for (int k = 0; k < K; k++) { sel_score[k] = original[sel[k]]; sum += sel_score[k]; }
    memset(out, 0, (size_t)D * sizeof(float));
    float *tmp = v4_alloc(D);
    for (int k = 0; k < K; k++) { float route = sum > 0.f ? m->c.route_scale * sel_score[k] / sum : 0.f; v4_expert(m, li, sel[k], x, route, tmp); for (int d = 0; d < D; d++) out[d] += tmp[d]; }
    v4_shared(m, l, x, tmp); for (int d = 0; d < D; d++) out[d] += tmp[d];
    free(tmp); free(original); free(select); free(sel_score); free(sel);
}

static void v4_layer(V4Model *m, V4Layer *l, int li, int token, int pos, float *h) {
    int H = m->c.hc_mult, D = m->c.dim;
    float *res = v4_alloc((int64_t)H * D), *x = v4_alloc(D), *norm = v4_alloc(D), *y = v4_alloc(D);
    float *post = (float *)alloca((size_t)H * sizeof(float));
    float *comb = (float *)alloca((size_t)H * H * sizeof(float));
    memcpy(res, h, (size_t)H * D * sizeof(float));
    int ready = v4_mix_hc(&m->c, h, l->hc_attn_fn, l->hc_attn_base, l->hc_attn_scale, x, post, comb);
    v4_rms(norm, x, l->attn_norm, D, m->c.norm_eps); v4_attention(m, l, li, norm, pos, y);
    v4_hc_post(&m->c, h, res, y, post, comb, ready);
    memcpy(res, h, (size_t)H * D * sizeof(float));
    ready = v4_mix_hc(&m->c, h, l->hc_ffn_fn, l->hc_ffn_base, l->hc_ffn_scale, x, post, comb);
    v4_rms(norm, x, l->ffn_norm, D, m->c.norm_eps); v4_moe(m, l, li, norm, token, y);
    v4_hc_post(&m->c, h, res, y, post, comb, ready);
    free(res); free(x); free(norm); free(y);
}

static void v4_head(V4Model *m, const float *h, float *logits) {
    int H = m->c.hc_mult, D = m->c.dim;
    float *x = v4_alloc(D); double ss = 0.0;
    if (m->hc_head_ready) {
        float inv;
        for (int j = 0; j < H * D; j++) ss += (double)h[j] * h[j];
        inv = 1.f / sqrtf((float)(ss / (H * D)) + m->c.norm_eps);
        for (int d = 0; d < D; d++) x[d] = 0.f;
        for (int j = 0; j < H; j++) {
            double z = 0.0; for (int k = 0; k < H * D; k++) z += (double)m->hc_head_fn[(int64_t)j * H * D + k] * h[k];
            float p = v4_sigmoid((float)z * m->hc_head_scale[0] * inv + m->hc_head_base[j]) + m->c.hc_eps;
            for (int d = 0; d < D; d++) x[d] += p * h[(int64_t)j * D + d];
        }
    } else { for (int d = 0; d < D; d++) x[d] = h[d]; }
    v4_rms(x, x, m->final_norm, D, m->c.norm_eps);
    v4_wmatvec(logits, x, &m->head); free(x);
}

static int v4_forward(V4Model *m, int token, int pos, float *logits) {
    int H = m->c.hc_mult, D = m->c.dim;
    float *h = v4_alloc((int64_t)H * D);
    float *emb = v4_alloc(D);
    int id = token; if (id < 0 || id >= m->c.vocab) id = 0;
    /* Embedding is stored row-wise as int8 in the converted layout. */
    if (m->embed.fmt == 0) memcpy(emb, m->embed.f + (int64_t)id * D, (size_t)D * sizeof(float));
    else {
        int8_t *row = (int8_t *)m->embed.q + (int64_t)id * D;
        for (int d = 0; d < D; d++) emb[d] = (float)row[d] * m->embed.scale[id];
    }
    for (int j = 0; j < H; j++) memcpy(h + (int64_t)j * D, emb, (size_t)D * sizeof(float));
    for (int li = 0; li < m->c.n_layers; li++) v4_layer(m, &m->layers[li], li, token, pos, h);
    v4_head(m, h, logits); free(emb); free(h); return 1;
}

static void v4_layer_load(V4Model *m, V4Layer *l, int li) {
    const V4Cfg *c = &m->c; char n[256]; int H = c->hc_mult, D = c->dim, M = (H + 2) * H;
#define N(fmt, ...) (snprintf(n, sizeof(n), fmt, __VA_ARGS__), n)
    l->attn_norm = v4_f32_load(m, N("layers.%d.attn_norm.weight", li), D, 0);
    l->ffn_norm = v4_f32_load(m, N("layers.%d.ffn_norm.weight", li), D, 0);
    v4_wload(m, &l->wq_a, N("layers.%d.attn.wq_a.weight", li), c->q_lora, D, 0);
    l->q_norm = v4_f32_load(m, N("layers.%d.attn.q_norm.weight", li), c->q_lora, 0);
    v4_wload(m, &l->wq_b, N("layers.%d.attn.wq_b.weight", li), c->n_heads * c->head_dim, c->q_lora, 0);
    v4_wload(m, &l->wkv, N("layers.%d.attn.wkv.weight", li), c->head_dim, D, 0);
    l->kv_norm = v4_f32_load(m, N("layers.%d.attn.kv_norm.weight", li), c->head_dim, 0);
    v4_wload(m, &l->wo_a, N("layers.%d.attn.wo_a.weight", li), c->o_groups * c->o_lora,
             (c->n_heads / c->o_groups) * c->head_dim, 0);
    v4_wload(m, &l->wo_b, N("layers.%d.attn.wo_b.weight", li), D, c->o_groups * c->o_lora, 0);
    l->sink = v4_f32_load(m, N("layers.%d.attn.attn_sink", li), c->n_heads, 0);
    int r = c->compress_ratios[li]; v4_cache_init(&l->kv, r, c->window, c->head_dim, c->max_ctx);
    if (r > 0) {
        int coff = r == 4 ? 2 : 1; l->compressor_coff = coff;
        v4_wload(m, &l->compressor_wkv, N("layers.%d.attn.compressor.wkv.weight", li), coff * c->head_dim, D, 1);
        v4_wload(m, &l->compressor_wgate, N("layers.%d.attn.compressor.wgate.weight", li), coff * c->head_dim, D, 1);
        l->compressor_ape = v4_f32_load(m, N("layers.%d.attn.compressor.ape", li), (int64_t)r * coff * c->head_dim, 1);
        l->compressor_norm = v4_f32_load(m, N("layers.%d.attn.compressor.norm.weight", li), c->head_dim, 1);
    }
    v4_wload(m, &l->gate, N("layers.%d.ffn.gate.weight", li), c->n_routed, D, 0);
    l->gate_bias = v4_f32_load(m, N("layers.%d.ffn.gate.bias", li), c->n_routed, 1);
    if (li < c->n_hash) l->tid2eid = v4_f32_load(m, N("layers.%d.ffn.gate.tid2eid", li), (int64_t)c->vocab * c->n_active, 1);
    v4_wload(m, &l->shared_w1, N("layers.%d.ffn.shared_experts.w1.weight", li), c->moe_inter, D, 0);
    v4_wload(m, &l->shared_w2, N("layers.%d.ffn.shared_experts.w2.weight", li), D, c->moe_inter, 0);
    v4_wload(m, &l->shared_w3, N("layers.%d.ffn.shared_experts.w3.weight", li), c->moe_inter, D, 0);
    l->hc_attn_fn = v4_f32_load(m, N("layers.%d.hc_attn_fn", li), (int64_t)M * H * D, 1);
    l->hc_attn_base = v4_f32_load(m, N("layers.%d.hc_attn_base", li), M, 1);
    l->hc_attn_scale = v4_f32_load(m, N("layers.%d.hc_attn_scale", li), 3, 1);
    l->hc_ffn_fn = v4_f32_load(m, N("layers.%d.hc_ffn_fn", li), (int64_t)M * H * D, 1);
    l->hc_ffn_base = v4_f32_load(m, N("layers.%d.hc_ffn_base", li), M, 1);
    l->hc_ffn_scale = v4_f32_load(m, N("layers.%d.hc_ffn_scale", li), 3, 1);
    l->hc_ready = l->hc_attn_fn && l->hc_attn_base && l->hc_attn_scale && l->hc_ffn_fn && l->hc_ffn_base && l->hc_ffn_scale;
#undef N
}

static void v4_model_init(V4Model *m, const char *snap) {
    memset(m, 0, sizeof(*m)); v4_load_config(&m->c, snap); st_init(&m->S, snap);
    const V4Cfg *c = &m->c; double start = v4_now();
    v4_wload(m, &m->embed, "embed.weight", c->vocab, c->dim, 0);
    m->final_norm = v4_f32_load(m, "norm.weight", c->dim, 0);
    v4_wload(m, &m->head, "head.weight", c->vocab, c->dim, 0);
    int M = c->hc_mult;
    m->hc_head_fn = v4_f32_load(m, "hc_head_fn", (int64_t)M * M * c->dim, 1);
    m->hc_head_base = v4_f32_load(m, "hc_head_base", M, 1);
    m->hc_head_scale = v4_f32_load(m, "hc_head_scale", 1, 1);
    m->hc_head_ready = m->hc_head_fn && m->hc_head_base && m->hc_head_scale;
    m->layers = (V4Layer *)calloc((size_t)c->n_layers, sizeof(V4Layer));
    if (!m->layers) { fprintf(stderr, "DeepSeek-V4: OOM layer table\n"); exit(1); }
    for (int i = 0; i < c->n_layers; i++) {
        v4_layer_load(m, &m->layers[i], i);
        if ((i + 1) % 4 == 0 || i + 1 == c->n_layers)
            fprintf(stderr, "[DSV4] loaded dense layer %d/%d (%.1fs)\n", i + 1, c->n_layers, v4_now() - start);
    }
    fprintf(stderr, "[DSV4] CPU streaming ready: %d layers, %d experts, ctx=%d, resident dense+KV; routed experts stream from SSD\n",
            c->n_layers, c->n_routed, c->max_ctx);
}

static int v4_is_eos(const V4Cfg *c, int token) {
    for (int i = 0; i < c->n_eos; i++) if (c->eos[i] == token) return 1;
    return token == 1;
}

static int v4_argmax(const float *x, int n) {
    int best = 0; float value = -INFINITY;
    for (int i = 0; i < n; i++) if (isfinite(x[i]) && x[i] > value) value = x[i], best = i;
    return best;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const char *snap = getenv("SNAP");
    const char *prompt = getenv("PROMPT");
    if (!snap || !*snap) { fprintf(stderr, "DeepSeek-V4: SNAP=<converted model directory> is required\n"); return 2; }
    if (!prompt) prompt = "";
    g_v4_drop_expert = getenv("DSV4_DROP_EXPERT_CACHE") ? atoi(getenv("DSV4_DROP_EXPERT_CACHE")) != 0 : 0;
    V4Model m; v4_model_init(&m, snap);
    Tok tok; char tokenizer[2048]; snprintf(tokenizer, sizeof(tokenizer), "%s/tokenizer.json", snap); tok_load(&tok, tokenizer);
    int max_ctx = m.c.max_ctx;
    const char *ctx_env = getenv("CTX"); if (ctx_env && atoi(ctx_env) > 0 && atoi(ctx_env) < max_ctx) max_ctx = atoi(ctx_env);
    int ngen = getenv("NGEN") ? atoi(getenv("NGEN")) : 128; if (ngen < 0) ngen = 0;
    int *ids = (int *)v4_bytes((int64_t)max_ctx * sizeof(int));
    int n = tok_encode(&tok, prompt, (int)strlen(prompt), ids, max_ctx);
    if (n == 0) ids[n++] = 0;
    if (n > max_ctx) n = max_ctx;
    float *logits = v4_alloc(m.c.vocab);
    double t0 = v4_now();
    for (int i = 0; i < n; i++) v4_forward(&m, ids[i], i, logits);
    int produced = 0;
    int next = v4_argmax(logits, m.c.vocab);
    while (produced < ngen && n + produced < max_ctx) {
        if (v4_is_eos(&m.c, next)) break;
        char text[8192]; int one = next; int bytes = tok_decode(&tok, &one, 1, text, (int)sizeof(text) - 1);
        if (bytes > 0) { fwrite(text, 1, (size_t)bytes, stdout); fflush(stdout); }
        ids[n + produced] = next;
        produced++;
        v4_forward(&m, next, n + produced - 1, logits);
        next = v4_argmax(logits, m.c.vocab);
    }
    double elapsed = v4_now() - t0;
    fprintf(stderr, "\n[DSV4] generated %d token(s) in %.2fs (%.3f tok/s) | target mode: CPU, SSD expert streaming\n",
            produced, elapsed, produced > 0 ? produced / elapsed : 0.0);
    free(logits); free(ids); return 0;
}
