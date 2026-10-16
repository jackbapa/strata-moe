// Standalone A3B single-step GDN parity check. Compile this with
// src/kernels/cuda/fused_gdn.cu; it does not alter the inference graph.
#include "strata/kernels/fused_gdn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int S = 128;
constexpr int HK = 16;
constexpr int HV = 32;
constexpr float EPS = 1.0e-6f;

void check(cudaError_t error, const char * at) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", at, cudaGetErrorString(error));
        std::exit(1);
    }
}

// Independent CUDA transcription of GGML's gated_delta_net_cuda<128, false>.
// It intentionally leaves normalization and output gating for a second launch.
__global__ void reference_step(const float * state_in, float * state_out,
                               const float * q, const float * k, const float * v,
                               const float * gate, const float * beta, float * score) {
    const int head = blockIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int qh = head % HK;
    const float * old = state_in + ((size_t) head * S + col) * S + lane;
    float * updated = state_out + ((size_t) head * S + col) * S + lane;
    float s[4], kr[4], qr[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        s[r] = old[r * 32];
        kr[r] = k[qh * S + r * 32 + lane];
        qr[r] = q[qh * S + r * 32 + lane];
    }
    const float decay = expf(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) kv += s[r] * kr[r];
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) kv += __shfl_xor_sync(0xffffffffu, kv, shift);
    const float delta = (v[head * S + col] - decay * kv) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        s[r] = decay * s[r] + kr[r] * delta;
        o += s[r] * qr[r];
        updated[r * 32] = s[r];
    }
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) o += __shfl_xor_sync(0xffffffffu, o, shift);
    if (lane == 0) score[head * S + col] = o * (1.0f / sqrtf((float) S));
}

__global__ void reference_norm(const float * score, const float * z,
                               const float * gamma, float * y) {
    __shared__ float partial[4];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const float a = score[head * S + col];
    float ss = a * a;
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, shift);
    if ((col & 31) == 0) partial[col >> 5] = ss;
    __syncthreads();
    const float scale = rsqrtf((partial[0] + partial[1] + partial[2] + partial[3]) / (float) S + EPS);
    const float zz = z[head * S + col];
    y[head * S + col] = a * scale * gamma[col] * (zz / (1.0f + expf(-zz)));
}

template <typename T> T * device(const std::vector<T> & host) {
    T * ptr = nullptr;
    check(cudaMalloc(&ptr, host.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return ptr;
}

std::vector<float> download(const float * ptr, size_t n) {
    std::vector<float> host(n);
    check(cudaMemcpy(host.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return host;
}

bool compare(const char * label, const std::vector<float> & actual,
             const std::vector<float> & expected, float tol) {
    float max_abs = 0.0f;
    size_t bad = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        max_abs = std::max(max_abs, diff);
        if (!std::isfinite(actual[i]) || diff > tol) ++bad;
    }
    std::printf("%s: max_abs=%.9g bad=%zu/%zu (tol=%.1g)\n",
                label, max_abs, bad, actual.size(), tol);
    return bad == 0;
}

template <typename Launch> float bench_ms(Launch launch) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    for (int i = 0; i < 10; ++i) launch();
    check(cudaDeviceSynchronize(), "GDN bench warmup");
    check(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < 200; ++i) launch();
    check(cudaEventRecord(stop), "cudaEventRecord stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return elapsed / 200.0f;
}

bool test_ab_f32() {
    constexpr int N = 2048;
    std::vector<float> x(N), wa((size_t) HV * N), wb((size_t) HV * N);
    std::vector<float> dt(HV), ssm_a(HV), expected_gate(HV), expected_beta(HV);
    for (int i = 0; i < N; ++i) x[i] = 0.1f * std::sin((float) i * 0.061f);
    for (size_t i = 0; i < wa.size(); ++i) {
        wa[i] = 0.01f * std::cos((float) i * 0.011f);
        wb[i] = 0.01f * std::sin((float) i * 0.013f);
    }
    for (int h = 0; h < HV; ++h) {
        dt[h] = 0.02f * h;
        ssm_a[h] = -0.5f - 0.01f * h;
        double alpha = 0.0, beta = 0.0;
        for (int i = 0; i < N; ++i) {
            alpha += (double) wa[(size_t) h * N + i] * x[i];
            beta += (double) wb[(size_t) h * N + i] * x[i];
        }
        const double av = alpha + dt[h];
        expected_gate[h] = (float) (std::log1p(std::exp(av)) * ssm_a[h]);
        expected_beta[h] = (float) (1.0 / (1.0 + std::exp(-beta)));
    }
    float *d_x = device(x), *d_wa = device(wa), *d_wb = device(wb);
    float *d_dt = device(dt), *d_a = device(ssm_a);
    float *d_g = nullptr, *d_b = nullptr;
    check(cudaMalloc(&d_g, HV * sizeof(float)), "cudaMalloc AB gate");
    check(cudaMalloc(&d_b, HV * sizeof(float)), "cudaMalloc AB beta");
    strata::kernels::fused_gdn_a3b_ab_f32(d_x, d_wa, d_wb, d_dt, d_a, d_g, d_b, N, HV, nullptr);
    check(cudaDeviceSynchronize(), "AB parity synchronize");
    const auto gate = download(d_g, HV), beta = download(d_b, HV);
    const bool good = compare("AB gate F32 vs double", gate, expected_gate, 2.0e-6f) &&
                      compare("AB beta F32 vs double", beta, expected_beta, 2.0e-6f);
    for (float *ptr : {d_x, d_wa, d_wb, d_dt, d_a, d_g, d_b}) check(cudaFree(ptr), "cudaFree AB");
    return good;
}
} // namespace

int main() {
    const bool ab_good = test_ab_f32();
    const size_t state_size = (size_t) HV * S * S;
    const size_t qk_size = (size_t) HK * S;
    const size_t out_size = (size_t) HV * S;
    std::vector<float> state(state_size), q(qk_size), k(qk_size), v(out_size),
                       gate(HV), beta(HV), z(out_size), gamma(S);
    for (size_t i = 0; i < state.size(); ++i) state[i] = 0.01f * std::sin((float) i * 0.031f);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.07f * std::cos((float) i * 0.037f);
        k[i] = 0.07f * std::sin((float) i * 0.041f);
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.08f * std::cos((float) i * 0.027f);
        z[i] = 0.7f * std::sin((float) i * 0.017f);
    }
    for (int h = 0; h < HV; ++h) {
        gate[h] = -0.12f - 0.005f * h;
        beta[h] = 0.35f + 0.01f * h;
    }
    for (int c = 0; c < S; ++c) gamma[c] = 0.9f + 0.002f * c;

    float *d_sf = device(state), *d_sr = device(state), *d_q = device(q), *d_k = device(k);
    float *d_v = device(v), *d_gate = device(gate), *d_beta = device(beta);
    float *d_z = device(z), *d_gamma = device(gamma);
    float *d_yf = nullptr, *d_yr = nullptr, *d_score = nullptr;
    check(cudaMalloc(&d_yf, out_size * sizeof(float)), "cudaMalloc y fused");
    check(cudaMalloc(&d_yr, out_size * sizeof(float)), "cudaMalloc y reference");
    check(cudaMalloc(&d_score, out_size * sizeof(float)), "cudaMalloc score");

    bool good = ab_good;
    // The service graph reads the prior state and writes a distinct cache
    // snapshot. Exercise that alias pattern before the in-place recurrence.
    float *d_snapshot_f = nullptr, *d_snapshot_r = nullptr;
    check(cudaMalloc(&d_snapshot_f, state_size * sizeof(float)), "cudaMalloc fused snapshot");
    check(cudaMalloc(&d_snapshot_r, state_size * sizeof(float)), "cudaMalloc reference snapshot");
    strata::kernels::fused_gdn_a3b_step_norm(d_sf, d_snapshot_f, d_q, d_k, d_v,
        d_gate, d_beta, d_z, d_gamma, EPS, d_yf, HK, HV, nullptr);
    reference_step<<<dim3(HV, S / 4), dim3(32, 4)>>>(d_sr, d_snapshot_r, d_q, d_k,
        d_v, d_gate, d_beta, d_score);
    reference_norm<<<HV, S>>>(d_score, d_z, d_gamma, d_yr);
    check(cudaDeviceSynchronize(), "separate-state GDN parity synchronize");
    good &= compare("separate-state snapshot", download(d_snapshot_f, state_size),
                    download(d_snapshot_r, state_size), 1.0e-6f);
    good &= compare("separate-state output", download(d_yf, out_size),
                    download(d_yr, out_size), 1.0e-6f);
    check(cudaFree(d_snapshot_f), "cudaFree fused snapshot");
    check(cudaFree(d_snapshot_r), "cudaFree reference snapshot");

    for (int step = 0; step < 4; ++step) {
        strata::kernels::fused_gdn_a3b_step_norm(d_sf, d_sf, d_q, d_k, d_v,
            d_gate, d_beta, d_z, d_gamma, EPS, d_yf, HK, HV, nullptr);
        reference_step<<<dim3(HV, S / 4), dim3(32, 4)>>>(d_sr, d_sr, d_q, d_k,
            d_v, d_gate, d_beta, d_score);
        reference_norm<<<HV, S>>>(d_score, d_z, d_gamma, d_yr);
        check(cudaDeviceSynchronize(), "GDN parity synchronize");
        const auto sf = download(d_sf, state_size), sr = download(d_sr, state_size);
        const auto yf = download(d_yf, out_size), yr = download(d_yr, out_size);
        std::printf("step %d\n", step + 1);
        good &= compare("state", sf, sr, 1.0e-6f);
        good &= compare("output", yf, yr, 1.0e-6f);
    }

    if (good) {
        const auto fused = [&] {
            strata::kernels::fused_gdn_a3b_step_norm(d_sf, d_sf, d_q, d_k, d_v,
                d_gate, d_beta, d_z, d_gamma, EPS, d_yf, HK, HV, nullptr);
        };
        const auto reference = [&] {
            reference_step<<<dim3(HV, S / 4), dim3(32, 4)>>>(d_sr, d_sr,
                d_q, d_k, d_v, d_gate, d_beta, d_score);
            reference_norm<<<HV, S>>>(d_score, d_z, d_gamma, d_yr);
        };
        const float fused_us = bench_ms(fused) * 1000.0f;
        const float reference_us = bench_ms(reference) * 1000.0f;
        std::printf("GDN-only GPU time: candidate %.2f us; standalone GGML transcription %.2f us\n",
                    fused_us, reference_us);
    }

    for (float *ptr : {d_sf, d_sr, d_q, d_k, d_v, d_gate, d_beta, d_z,
                       d_gamma, d_yf, d_yr, d_score}) check(cudaFree(ptr), "cudaFree");
    return good ? 0 : 1;
}
