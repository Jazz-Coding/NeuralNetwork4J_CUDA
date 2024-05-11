package com.jazz.nn.gpu.utils;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.jcublas.cublasComputeType.CUBLAS_COMPUTE_32F_FAST_TF32;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

public class MatrixMath {
    /**
     * Custom CUDA kernels.
     */
    private static CUfunction sigKernel;
    private static CUfunction sigPrimeKernel;
    private static CUfunction hadKernel;
    private static CUfunction avgBroadcastAdd;

    private static cublasHandle handle;

    /**
     * Initialize CUBLAS backend.
     */
    public static void initialize(){
        JCublas2.initialize();
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        loadKernels();
    }

    /**
     * Load pre-compiled CUDA kernels.
     */
    private static final String KERNEL_DIRECTORY = "kernels_compiled"+"/";
    private static void loadKernels(){
        CUmodule module4 = new CUmodule();
        cuModuleLoad(module4, KERNEL_DIRECTORY+"avgBroadcastAdd.ptx");
        avgBroadcastAdd = new CUfunction();
        cuModuleGetFunction(avgBroadcastAdd, module4, "avg_broadcast_add");

        CUmodule module3 = new CUmodule();
        cuModuleLoad(module3, KERNEL_DIRECTORY+"SigKernel.ptx");
        sigKernel = new CUfunction();
        cuModuleGetFunction(sigKernel, module3, "sigmoid_kernel");

        CUmodule module = new CUmodule();
        cuModuleLoad(module, KERNEL_DIRECTORY+"SigPrimeKernel.ptx");
        sigPrimeKernel = new CUfunction();
        cuModuleGetFunction(sigPrimeKernel, module, "sigmoid_kernel");

        CUmodule module2 = new CUmodule();
        cuModuleLoad(module2, KERNEL_DIRECTORY+"HadKernel.ptx");
        hadKernel = new CUfunction();
        cuModuleGetFunction(hadKernel, module2, "hadamard_product");
    }

    public static void shutdown(){
        JCublas2.cublasDestroy(handle);
    }

    /**
     * Sigmoid (AKA logistic) function:
     * f(x) = 1 / (1+e^-x)
     * f'(x) = f(x) * (1-f(x))
     *
     * The sigmoid function is a common neuron activation function.
     * The derivative is used to compute the cost function gradient for parameter updates during backpropagation.
     */
    public static void executeSigmoid(MatrixPointer _input,
                                       MatrixPointer _output){

        int length1d = _input.getLength1d();
        Pointer kernelParameters = Pointer.to(
                Pointer.to(_input.getDevicePointer()),
                Pointer.to(_output.getDevicePointer()),
                Pointer.to(new int[]{length1d})
        );

        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)length1d / blockSizeX);
        cuLaunchKernel(sigKernel,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }
    public static void executeSigmoidPrime(MatrixPointer _input,
                                      MatrixPointer _output){

        int length1d = _input.getLength1d();
        Pointer kernelParameters = Pointer.to(
                Pointer.to(_input.getDevicePointer()),
                Pointer.to(_output.getDevicePointer()),
                Pointer.to(new int[]{length1d})
        );

        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)length1d / blockSizeX);
        cuLaunchKernel(sigPrimeKernel,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }

    /***
     * Matrix multiplication, expects column major (CM) matrices.
     * AT = transpose matrix A beforehand. BT = transpose matrix B beforehand.
     */
    private static final int MAT_DATA_TYPE = CUDA_R_32F; // 32-bit float/real number (single precision)
    private static final int MAT_COMPUTE_TYPE = CUBLAS_COMPUTE_32F_FAST_TF32; // Use TF32 tensor cores (faster), expects 32F input/output matrices.
    public static void executeMultiplicationCM( MatrixPointer _matA,
                                                float scaleA,
                                                MatrixPointer _matB,
                                                float scaleResult,
                                                MatrixPointer _acc){
        int m = _matA.getRows();
        int n = _matB.getCols();
        int k = _matA.getCols();

        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        JCublas2.cublasGemmEx_new(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, Pointer.to(new float[]{scaleA}),
                _matA.getDevicePointer(), MAT_DATA_TYPE, lda,
                _matB.getDevicePointer(), MAT_DATA_TYPE, ldb,
                Pointer.to(new float[]{scaleResult}), _acc.getDevicePointer(), MAT_DATA_TYPE, ldc, MAT_COMPUTE_TYPE, CUBLAS_GEMM_DEFAULT);
    }
    public static void executeMultiplicationCMBT( MatrixPointer _matA,
                                                float scaleA,
                                                MatrixPointer _matB,
                                                float scaleResult,
                                                MatrixPointer _acc){
        int m = _matA.getRows();
        int n = _matB.getRows();
        int k = _matA.getCols();

        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        JCublas2.cublasGemmEx_new(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k, Pointer.to(new float[]{scaleA}),
                _matA.getDevicePointer(), MAT_DATA_TYPE, lda,
                _matB.getDevicePointer(), MAT_DATA_TYPE, ldb,
                Pointer.to(new float[]{scaleResult}), _acc.getDevicePointer(), MAT_DATA_TYPE, ldc, MAT_COMPUTE_TYPE, CUBLAS_GEMM_DEFAULT);
    }

    public static void executeMultiplicationCMAT( MatrixPointer _matA,
                                                float scaleA,
                                                MatrixPointer _matB,
                                                float scaleResult,
                                                MatrixPointer _acc){
        int m = _matA.getRows();
        int n = _matB.getCols();
        int k = _matA.getCols();

        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        JCublas2.cublasGemmEx_new(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                k, n, m, Pointer.to(new float[]{scaleA}),
                _matA.getDevicePointer(), MAT_DATA_TYPE, lda,
                _matB.getDevicePointer(), MAT_DATA_TYPE, ldb,
                Pointer.to(new float[]{scaleResult}), _acc.getDevicePointer(), MAT_DATA_TYPE, ldc, MAT_COMPUTE_TYPE, CUBLAS_GEMM_DEFAULT);
    }

    public static void executeAdditionWithTranspose(MatrixPointer _matA,
                                                    int transposeA,
                                                    float scalarA,
                                                    MatrixPointer _matB,
                                                    int transposeB,
                                                    float scalarB,
                                                    MatrixPointer _acc){
        int lda = _matA.getLD();
        int ldb = _matB.getLD();
        int ldc = _acc.getLD();

        JCublas2.cublasSgeam(handle,
                transposeA, transposeB,
                _matA.getCols(), _matB.getRows(),
                Pointer.to(new float[]{scalarA}),
                _matA.getDevicePointer(), lda,
                Pointer.to(new float[]{scalarB}),
                _matB.getDevicePointer(), ldb,
                _acc.getDevicePointer(),  ldc);
    }

    /**
     * Matrix ACC = Matrix A + Matrix B
     */
    public static void executeAddition(MatrixPointer _matA,
                                       float scalar,
                                       MatrixPointer _matB,
                                       MatrixPointer _acc){
        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        int m = _matA.getRows();
        int n = _matB.getCols();

        JCublas2.cublasSgeam(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n,
                Pointer.to(new float[]{1F}),
                _matA.getDevicePointer(), lda, Pointer.to(new float[]{scalar}),
                _matB.getDevicePointer(), ldb,
                _acc.getDevicePointer(),  ldc);
    }

    /**
     * Matrix ACC = Matrix A - Matrix B
     */
    public static void executeDiff(MatrixPointer _matA,
                                    MatrixPointer _matB,
                                    MatrixPointer _acc){
        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        int m = _matA.getRows();
        int n = _matB.getCols();
        JCublas2.cublasSgeam(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n,
                Pointer.to(new float[]{1F}),
                _matA.getDevicePointer(), lda, Pointer.to(new float[]{-1F}),
                _matB.getDevicePointer(), ldb,
                _acc.getDevicePointer(),  ldc);
    }

    /**
     * Matrix ACC[i] = Matrix A[i] * Matrix B[i] (element wise multiplication)
     */
    public static void executeHadamard(MatrixPointer _matA,
                                float scaleA,
                                MatrixPointer _matB,
                                MatrixPointer _acc){
        int length1d = _matA.getLength1d();
        Pointer kernelParameters = Pointer.to(
                Pointer.to(_matA.getDevicePointer()),
                Pointer.to(_matB.getDevicePointer()),
                Pointer.to(_acc.getDevicePointer()),
                Pointer.to(new float[]{scaleA}),
                Pointer.to(new int[]{length1d})
                );
        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)length1d / blockSizeX);
        cuLaunchKernel(hadKernel,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }

    /**
     * Averages each row of the matrix, scales by a factor, and then adds the
     * resulting column vector to each column in the output.
     */
    public static void executeAvgBroadcastAdd(MatrixPointer _mat,
                                              MatrixPointer _acc,
                                              float scalar){
        int length1d = _mat.getLength1d();
        Pointer kernelParameters = Pointer.to(
                Pointer.to(_mat.getDevicePointer()),
                Pointer.to(_acc.getDevicePointer()),
                Pointer.to(new float[]{scalar}),
                Pointer.to(new int[]{_mat.getRows()}),
                Pointer.to(new int[]{_mat.getCols()}       )
        );
        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)length1d / blockSizeX);
        cuLaunchKernel(avgBroadcastAdd,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }

    /**
     * Computes the derivative of the mean squared error (MSE) cost function, for use in calculating the final layer error during backpropagation.
     */
    public static void executeMSEDerivative(MatrixPointer _networkOutputs,
                                            MatrixPointer _expectedOutputs,
                                            MatrixPointer _outputDifferenceHolder, // We could
                                            // create these on the fly and clear them after,
                                            // but, since they will always be created to the same
                                            // dimensions, perhaps it would be faster to give them
                                            // a permanent place in VRAM and just reference them
                                            // afterwards where needed. Hence their inclusion here.
                                            MatrixPointer _outputPreActivations,
                                            MatrixPointer _sigmoidDerivativeHolder,
                                            MatrixPointer _acc){
        // First, find the difference.
        executeDiff(_networkOutputs,_expectedOutputs,_outputDifferenceHolder);

        // Then, find the derivative of the activation function wrt. pre-activations.
        executeSigmoidPrime(_outputPreActivations,_sigmoidDerivativeHolder);

        // Finally, hadamard multiply the two (scaling the former by 2 to account for
        // the precise form of the mean squared error we used.
        executeHadamard(_outputDifferenceHolder,2F,_sigmoidDerivativeHolder,_acc);
    }
}
