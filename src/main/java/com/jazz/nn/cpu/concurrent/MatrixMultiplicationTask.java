package com.jazz.nn.cpu.concurrent;

import java.util.concurrent.RecursiveTask;

public class MatrixMultiplicationTask extends RecursiveTask<float[][]> {
    private static final int THRESHOLD = 32;

    private final float[][] matA;
    private final float[][] matB;

    private final int start;
    private final int end;

    public MatrixMultiplicationTask(float[][] matA, float[][] matB, int start, int end) {
        this.matA = matA;
        this.matB = matB;
        this.start = start;
        this.end = end;
    }

    @Override
    protected float[][] compute() {
        int len = end - start;

        int colsB = matB[0].length;
        int colsA = matA[0].length;

        float[][] result = new float[len][colsB];

        if (len <= THRESHOLD) {
            for (int i = start; i < end; i++) {
                for (int j = 0; j < colsB; j++) {
                    for (int k = 0; k < colsA; k++) {
                        result[i-start][j] += matA[i][k] * matB[k][j];
                    }
                }
            }
        } else {
            // Recursive case: divide the problem into smaller ones.
            int mid = (start + end) / 2;
            MatrixMultiplicationTask task1 = new MatrixMultiplicationTask(matA, matB, start, mid);
            MatrixMultiplicationTask task2 = new MatrixMultiplicationTask(matA, matB, mid, end);
            invokeAll(task1, task2);

            float[][] result1 = task1.join();
            float[][] result2 = task2.join();

            // Combine the results.
            System.arraycopy(result1, 0, result, 0, result1.length);
            System.arraycopy(result2, 0, result, result1.length, result2.length);
        }

        return result;
    }
}
