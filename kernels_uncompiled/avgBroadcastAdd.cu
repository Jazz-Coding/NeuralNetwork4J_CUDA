extern "C"
__global__ void avg_broadcast_add(float* input_matrix, float* output_matrix, float scalar, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        // Compute row average
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += input_matrix[idx * cols + j];
        }
        float avg = sum / cols;

        // Scale average and add to output matrix
        for (int j = 0; j < cols; j++) {
            output_matrix[idx * cols + j] += scalar * avg;
        }
    }
}