package com.jazz.nn.cpu.utils;

import java.util.Arrays;

public class PrettyPrint {
    public static String matrixPrint(float[][] mat){
        StringBuilder sb = new StringBuilder();
        for (float[] row : mat) {
            sb.append(Arrays.toString(row)).append("\n");
        }
        return sb.toString();
        /*StringBuilder sb = new StringBuilder();

        int rows = mat.length;
        int cols = mat[0].length;

        for (int i = 0; i < rows; i++) {
            float[] row = mat[i];

            for (int j = 0; j < cols-1; j++) {
                sb.append(row[0]).append(", ");
            }
            sb.append(row[cols-1]);

            if(i != rows-1) {
                sb.append("\n");
            }
        }

        return sb.toString();*/
    }

    public static String matrixPrint(float[][][] mat){
        StringBuilder sb = new StringBuilder();

        int aisles = mat.length;
        for (int i = 0; i < aisles; i++) {
            sb.append(matrixPrint(mat[i]));

            if(i != aisles-1) {
                sb.append("\n");
            }
        }
        return sb.toString();
    }
}
