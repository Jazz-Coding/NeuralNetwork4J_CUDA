extern "C"
__global__ void hadamard_product(const float* a, const float* b, float* c, float scalar, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = scalar * (a[idx] * b[idx]);
    }
}