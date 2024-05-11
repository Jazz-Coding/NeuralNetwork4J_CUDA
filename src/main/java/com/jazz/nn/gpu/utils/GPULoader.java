package com.jazz.nn.gpu.utils;

import jcuda.Pointer;
import jcuda.Sizeof;

import static com.jazz.nn.cpu.utils.VectorMath.*;
import static jcuda.jcublas.JCublas.*;

public class GPULoader {
    /**
     * Loads vector into VRAM.
     */
    public static Pointer loadVector(float[] vector){
        Pointer devicePointer = new Pointer();
        cublasAlloc(vector.length, Sizeof.FLOAT, devicePointer);
        cublasSetVector(vector.length, Sizeof.FLOAT, Pointer.to(vector), 1, devicePointer, 1);
        return devicePointer;
    }

    /**
     * Loads matrix into VRAM.
     * Wraps in a "flatten" call and uses the vector loading.
     * We could use cublasSetMatrix but this simplifies things to just one type of call.
     */
    public static Pointer loadMatrix(float[][] matrix){
        return loadVector(flatten(matrix));
    }

    /**
     * Flattens into column-major format.
     */
    public static Pointer loadMatrixColumnMajor(float[][] matrix){
        return loadVector(flattenColumnMajor(matrix));
    }

    /**
     * Loads vector out of VRAM.
     */
    public static void unloadVector(Pointer devicePointer, Pointer extractionPoint, int length){
        cublasGetVector(length, Sizeof.FLOAT, devicePointer, 1, extractionPoint, 1);
        cublasFree(devicePointer);
    }

    /**
     * Copies vector out of VRAM.
     */
    public static void peekVector(Pointer devicePointer, Pointer extractionPoint, int length){
        cublasGetVector(length, Sizeof.FLOAT, devicePointer, 1, extractionPoint, 1);
    }

    /**
     * Copies whole matrix out of VRAM.
     */
    public static float[][] peekMatrix(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        peekVector(source,Pointer.to(vector),length1d);
        return inflate(vector,rows,cols);
    }

    /**
     * Loads matrix out of VRAM.
     */
    public static float[][] unloadMatrix(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        unloadVector(source,Pointer.to(vector),length1d);
        return inflate(vector,rows,cols);
    }

    /**
     * Copies whole matrix out of VRAM using a matrix pointer.
     */
    public static float[][] peekMatrix(MatrixPointer matrixPointer){
        return peekMatrix(matrixPointer.getDevicePointer(), matrixPointer.getRows(), matrixPointer.getCols());
    }

    /**
     * Same for full unloading.
     */
    public static float[][] unloadMatrix(MatrixPointer matrixPointer){
        return unloadMatrix(matrixPointer.getDevicePointer(),matrixPointer.getRows(),matrixPointer.getCols());
    }

    /**
     * Copies whole matrix out of VRAM.
     */
    public static float[][] peekMatrixColumnMajor(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        peekVector(source,Pointer.to(vector),length1d);
        return inflateColumnMajor(vector,rows,cols);
    }

    /**
     * Loads matrix out of VRAM.
     */
    public static float[][] unloadMatrixColumnMajor(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        unloadVector(source,Pointer.to(vector),length1d);
        return inflateColumnMajor(vector,rows,cols);
    }

    /**
     * Copies whole matrix out of VRAM using a matrix pointer.
     */
    public static float[][] peekMatrixColumnMajor(MatrixPointer matrixPointer){
        return peekMatrixColumnMajor(matrixPointer.getDevicePointer(), matrixPointer.getRows(), matrixPointer.getCols());
    }

    /**
     * Same for full unloading.
     */
    public static float[][] unloadMatrixColumnMajor(MatrixPointer matrixPointer){
        return unloadMatrixColumnMajor(matrixPointer.getDevicePointer(),matrixPointer.getRows(),matrixPointer.getCols());
    }

}
