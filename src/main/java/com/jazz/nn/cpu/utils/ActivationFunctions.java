package com.jazz.nn.cpu.utils;

public class ActivationFunctions {
    public static float sigmoid(float preActivation){
        // 1/(1+e^-z)
        return (float) (1F / (1F+Math.exp(-preActivation)));
    }

    public static float[][] sigmoid(float[][] preActivation){
        int rows = preActivation.length;
        int cols = preActivation[0].length;

        float[][] activations = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                activations[i][j] = sigmoid(preActivation[i][j]);
            }
        }
        return activations;
    }

    public static void sigmoidIP(float[][] preActivation, float[][] acc){
        int rows = preActivation.length;
        int cols = preActivation[0].length;

        //float[][] activations = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                acc[i][j] = sigmoid(preActivation[i][j]);
            }
        }
    }

    public static float sigmoidDerivative(float preActivation){
        float normalSigmoid = sigmoid(preActivation);
        return normalSigmoid * (1-normalSigmoid);
    }

    public static float[][] sigmoidDerivative(float[][] preActivation){
        int rows = preActivation.length;
        int cols = preActivation[0].length;

        float[][] activations = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                activations[i][j] = sigmoidDerivative(preActivation[i][j]);
            }
        }
        return activations;
    }

    public static void main(String[] args) {
        System.out.println("Modestly large:");
        System.out.println(sigmoid(1337));
        System.out.println(sigmoid(-1337));

        System.out.println("Extremely large:");
        System.out.println(sigmoid(1337e12F));
        System.out.println(sigmoid(-1337312F));

        System.out.println("Close to zero:");
        System.out.println(sigmoid(1F/1337));

        System.out.println("Extremely close to zero:");
        System.out.println(sigmoid(1F/1337e9F));

        System.out.println("Equal to zero:");
        System.out.println(sigmoid(0));

        System.out.println("Infinity: ");
        System.out.println(sigmoid(Float.POSITIVE_INFINITY));
        System.out.println("Negative infinity: ");
        System.out.println(sigmoid(Float.NEGATIVE_INFINITY));
        System.out.println("NAN:");
        System.out.println(sigmoid(Float.NaN));
    }
}
