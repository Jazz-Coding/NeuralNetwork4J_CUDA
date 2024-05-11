package com.jazz.nn.cpu.utils;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class RandomArrays {
    public static void fillRandomly(float[] array){
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) ThreadLocalRandom.current().nextGaussian();
        }
    }
    public static void fillRandomly(float[][] array2D){
        for (float[] row : array2D) {
            fillRandomly(row);
        }
    }

    public static void fillRandomly(float[] array, int seed){
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) new Random(seed+(299*i)).nextGaussian();
        }
    }

    public static void fillRandomly(float[][] array2D, int seed){
        for (float[] row : array2D) {
            fillRandomly(row, seed);
        }
    }
    public static void fillRandomlyScaled(float[] array, int inboundConnections){
        Random RNG = ThreadLocalRandom.current();
        double adjustment = 1/Math.sqrt(inboundConnections); // scale by the number of inbound connections

        for (int i = 0; i < array.length; i++) {
            array[i] = (float) RNG.nextGaussian(0,adjustment);
        }
    }

    public static void fillRandomlyScaled(float[][] array2D, int inboundConnections){
        for (float[] row : array2D) {
            fillRandomlyScaled(row, inboundConnections);
        }
    }

    public static void fillOnes(float[][] array2D){
        for (int i = 0; i < array2D.length; i++) {
            for (int j = 0; j < array2D[0].length; j++) {
                array2D[i][j] = 1F;
            }
        }
    }
}
