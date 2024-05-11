package com.jazz.nn.gpu.utils;

import jcuda.Pointer;

import static com.jazz.nn.cpu.utils.PrettyPrint.matrixPrint;
import static com.jazz.nn.cpu.utils.RandomArrays.fillRandomly;

/**
 * Helper class that remembers how big the matrices pointed to actually are.
 * Also allows for convenient loading/unloading from VRAM and ensures
 * memory is freed.
 */
public class MatrixPointer {
    private Pointer devicePointer;
    private int rows;
    private int cols;
    private int length1d; // Frequently used, saves recalculating each time.

    /**
     * Facade pattern - matrix allocation on the GPU is handled automatically behind the scenes so that
     * we can just say "here's my matrix, figure out the details and give me the pointer".
     *
     * Automatic memory freeing is NOT performed though, so must be handled explicitly.
     */
    public MatrixPointer(float[][] matrix) {
        this.rows = matrix.length;
        this.cols = matrix[0].length;
        this.length1d = this.rows*this.cols;
        this.devicePointer = GPULoader.loadMatrixColumnMajor(matrix);
    }

    /**
     * Some commonly used default matrices.
     */
    public static MatrixPointer randomMatrix(int rows, int cols){
        float[][] matrix = new float[rows][cols];
        fillRandomly(matrix);

        return new MatrixPointer(matrix);
    }
    public static MatrixPointer zeroMatrix(int rows, int cols){
        float[][] matrix = new float[rows][cols];
        return new MatrixPointer(matrix);
    }
    public static MatrixPointer oneMatrix(int rows, int cols){
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = 1F;
            }
        }
        return new MatrixPointer(matrix);
    }

    public Pointer getDevicePointer() {
        return devicePointer;
    }
    public int getRows() {
        return rows;
    }

    /**
     * Aliases.
     */
    public int getCols() {
        return cols;
    }
    public int getLD(){
        return rows;
    }
    public int getLength1d() {
        return length1d;
    }

    /**
     * Copies contents off GPU.
     */
    public float[][] peek(){
        return GPULoader.peekMatrixColumnMajor(this);
    }

    /**
     * Loads off GPU (frees memory afterwards).
     */
    public float[][] unload(){
        return GPULoader.unloadMatrixColumnMajor(this);
    }

    @Override
    public String toString() {
        return matrixPrint(peek());
    }
}
