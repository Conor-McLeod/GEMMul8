// Minimal C ABI wrapper around gemmul8::gemm so Python (ctypes) can call it.
// Inputs/outputs are host arrays in column-major order.
#include "gemmul8.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" int gemmul8_dgemm_int8(int m, int n, int k,
                                  const double *hA, const double *hB, double *hC,
                                  int num_moduli, int fastmode) {
    static cublasHandle_t handle = nullptr; // reused across warm invocations
    if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return -1;

    const double alpha = 1.0, beta = 0.0;
    double *A, *B, *C;
    void *work = nullptr;
    if (cudaMalloc(&A, sizeof(double) * m * k) != cudaSuccess) return -2;
    if (cudaMalloc(&B, sizeof(double) * k * n) != cudaSuccess) return -2;
    if (cudaMalloc(&C, sizeof(double) * m * n) != cudaSuccess) return -2;
    cudaMemcpy(A, hA, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    // First call with work == nullptr returns the workspace size in res[0].
    auto res = gemmul8::gemm<gemmul8::Backend::INT8>(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m,
        num_moduli, fastmode != 0, work);
    if (cudaMalloc(&work, size_t(res[0])) != cudaSuccess) return -2;
    gemmul8::gemm<gemmul8::Backend::INT8>(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m,
        num_moduli, fastmode != 0, work);

    cudaMemcpy(hC, C, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
    const int err = cudaGetLastError() == cudaSuccess ? 0 : -3;
    cudaFree(work); cudaFree(C); cudaFree(B); cudaFree(A);
    return err;
}
