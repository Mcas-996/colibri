/* Small no-model regression checks for the V4 packing/rope helpers. */
#define main deepseek_v4_engine_main
#include "../deepseek_v4.c"
#undef main

int main(void) {
    V4W w = {0};
    uint8_t packed[16] = {0};
    uint8_t scales[1] = {127};
    for (int i = 0; i < 16; i++) packed[i] = 0x11; /* two x 0.5 */
    w.fmt = 7; w.O = 1; w.I = 32; w.q = packed; w.mscale = scales;
    float x[32]; for (int i = 0; i < 32; i++) x[i] = 1.f;
    float got = v4_wdot(&w, 0, x);
    if (fabsf(got - 16.f) > 1e-5f) return 1;

    V4Cfg c = {0}; c.norm_eps = 1e-6f; c.original_seq_len = 0; c.rope_factor = 1.f;
    float pair[2] = {1.f, 0.f};
    v4_rope(pair, 2, 1, 10000.f, 0, &c, 0);
    if (fabsf(pair[0] - cosf(1.f)) > 1e-5f || fabsf(pair[1] - sinf(1.f)) > 1e-5f) return 2;
    return 0;
}
