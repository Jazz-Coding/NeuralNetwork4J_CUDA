__device__ __forceinline__ float sigmoid (float a)
{
    return 1.0 / (1.0 + exp (-a));
}

extern "C"
__global__ void sigmoid_kernel (const float * __restrict__ src, 
                                float * __restrict__ dst, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n)
    {
        dst[i] = sigmoid(src[i]);
    }
}  