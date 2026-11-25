#pragma once

#if defined(__APPLE__)
inline const char* metal_source() {
    return R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        acc += lhs[row * k + inner] * rhs[inner * n + col];
    }
    out[row * n + col] = acc;
}

kernel void linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        acc += input[row * k + inner] * weight[inner * n + col];
    }
    acc += bias[col];
    out[row * n + col] = acc;
}

kernel void linear_no_bias(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        acc += input[row * k + inner] * weight[inner * n + col];
    }
    out[row * n + col] = acc;
}

kernel void embedding(
    device const float* indices [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& index_count [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= embedding_dim || gid.y >= index_count) {
        return;
    }
    uint index = uint(max(indices[gid.y], 0.0f));
    out[gid.y * embedding_dim + gid.x] = weight[index * embedding_dim + gid.x];
}

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = max(input[gid], 0.0f);
}

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float x = input[gid];
    out[gid] = x / (1.0f + exp(-x));
}

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float x = input[gid];
    out[gid] = 0.5f * x * (1.0f + tanh(0.7978846f * (x + 0.044715f * x * x * x)));
}

kernel void tanh_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = tanh(input[gid]);
}

kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    float x = input[gid];
    out[gid] = 1.0f / (1.0f + exp(-x));
}

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float max_value = input[row * width];
    for (uint col = 1; col < width; ++col) {
        max_value = max(max_value, input[row * width + col]);
    }
    float sum = 0.0f;
    for (uint col = 0; col < width; ++col) {
        float e = exp(input[row * width + col] - max_value);
        out[row * width + col] = e;
        sum += e;
    }
    for (uint col = 0; col < width; ++col) {
        out[row * width + col] = out[row * width + col] / sum;
    }
}

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float mean_sq = 0.0f;
    for (uint col = 0; col < width; ++col) {
        float x = input[row * width + col];
        mean_sq += x * x;
    }
    mean_sq /= float(width);
    float inv_rms = rsqrt(mean_sq + 1.0e-5f);
    for (uint col = 0; col < width; ++col) {
        out[row * width + col] = input[row * width + col] * inv_rms;
    }
}

kernel void cross_entropy(
    device const float* logits [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    float max_value = logits[row * width];
    for (uint col = 1; col < width; ++col) {
        max_value = max(max_value, logits[row * width + col]);
    }
    float sum = 0.0f;
    for (uint col = 0; col < width; ++col) {
        sum += exp(logits[row * width + col] - max_value);
    }
    float loss = 0.0f;
    for (uint col = 0; col < width; ++col) {
        float probability = exp(logits[row * width + col] - max_value) / sum;
        loss -= target[row * width + col] * log(max(probability, 1.0e-6f));
    }
    out[row] = loss;
}

kernel void sqrt_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = sqrt(input[gid]);
}

kernel void rsqrt_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = rsqrt(input[gid]);
}

kernel void copy_op(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    out[gid] = input[gid];
}

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }
    out[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
}

kernel void repeat_kv(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& out_heads [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& repeats [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inner || gid.y >= outer * out_heads) {
        return;
    }
    uint outer_idx = gid.y / out_heads;
    uint dst_head = gid.y % out_heads;
    uint src_head = dst_head / repeats;
    uint in_heads = out_heads / repeats;
    uint src = (outer_idx * in_heads + src_head) * inner + gid.x;
    uint dst = gid.y * inner + gid.x;
    out[dst] = input[src];
}

kernel void causal_mask(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& q [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= k || gid.y >= outer * q) {
        return;
    }
    uint outer_idx = gid.y / q;
    uint row = gid.y % q;
    uint index = outer_idx * q * k + row * k + gid.x;
    float value = input[index];
    if (gid.x > row) {
        value = -10000.0f;
    }
    out[index] = value;
}

kernel void rope(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& half_dim [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= half_dim || gid.y >= outer * seq_len) {
        return;
    }
    uint outer_idx = gid.y / seq_len;
    uint pos = gid.y % seq_len;
    uint head_dim = half_dim * 2;
    uint base = outer_idx * seq_len * head_dim + pos * head_dim;
    float inv_freq = pow(theta, -(float(gid.x) / float(half_dim)));
    float angle = float(pos) * inv_freq;
    float c = cos(angle);
    float s = sin(angle);
    float x1 = input[base + gid.x];
    float x2 = input[base + half_dim + gid.x];
    out[base + gid.x] = x1 * c - x2 * s;
    out[base + half_dim + gid.x] = x1 * s + x2 * c;
}

kernel void sum_reduce(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    if (axis == 0) {
        if (gid >= cols) return;
        for (uint row = 0; row < rows; ++row) {
            acc += input[row * cols + gid];
        }
        out[gid] = acc;
        return;
    }
    if (axis == 1) {
        if (gid >= rows) return;
        uint base = gid * cols;
        for (uint col = 0; col < cols; ++col) {
            acc += input[base + col];
        }
        out[gid] = acc;
        return;
    }
    if (gid > 0) return;
    uint count = rows * cols;
    for (uint index = 0; index < count; ++index) {
        acc += input[index];
    }
    out[0] = acc;
}

kernel void mean_reduce(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    if (axis == 0) {
        if (gid >= cols) return;
        for (uint row = 0; row < rows; ++row) {
            acc += input[row * cols + gid];
        }
        out[gid] = acc / float(rows);
        return;
    }
    if (axis == 1) {
        if (gid >= rows) return;
        uint base = gid * cols;
        for (uint col = 0; col < cols; ++col) {
            acc += input[base + col];
        }
        out[gid] = acc / float(cols);
        return;
    }
    if (gid > 0) return;
    uint count = rows * cols;
    for (uint index = 0; index < count; ++index) {
        acc += input[index];
    }
    out[0] = acc / float(count);
}

#define TYSOR_BINARY_TT(name, expr) \
kernel void name(device const float* lhs [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) { \
    if (gid >= count) return; \
    float l = lhs[gid]; \
    float r = rhs[gid]; \
    out[gid] = (expr); \
}

#define TYSOR_BINARY_TS(name, expr) \
kernel void name(device const float* input [[buffer(0)]], device float* out [[buffer(1)]], constant uint& count [[buffer(2)]], constant float& scalar [[buffer(3)]], uint gid [[thread_position_in_grid]]) { \
    if (gid >= count) return; \
    float t = input[gid]; \
    float s = scalar; \
    out[gid] = (expr); \
}

TYSOR_BINARY_TT(add_tt, l + r)
TYSOR_BINARY_TT(sub_tt, l - r)
TYSOR_BINARY_TT(mul_tt, l * r)
TYSOR_BINARY_TT(div_tt, l / r)
TYSOR_BINARY_TT(floordiv_tt, floor(l / r))
TYSOR_BINARY_TS(add_ts, t + s)
TYSOR_BINARY_TS(sub_ts, t - s)
TYSOR_BINARY_TS(mul_ts, t * s)
TYSOR_BINARY_TS(div_ts, t / s)
TYSOR_BINARY_TS(floordiv_ts, floor(t / s))
TYSOR_BINARY_TS(add_st, s + t)
TYSOR_BINARY_TS(sub_st, s - t)
TYSOR_BINARY_TS(mul_st, s * t)
TYSOR_BINARY_TS(div_st, s / t)
TYSOR_BINARY_TS(floordiv_st, floor(s / t))

#undef TYSOR_BINARY_TT
#undef TYSOR_BINARY_TS
)";
}
#endif
