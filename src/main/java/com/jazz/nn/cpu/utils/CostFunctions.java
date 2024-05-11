package com.jazz.nn.cpu.utils;

import static com.jazz.nn.cpu.utils.ActivationFunctions.sigmoidDerivative;
import static com.jazz.nn.cpu.utils.VectorMath.*;

public class CostFunctions {
    // Mean of the error squared. Error = difference between actual and expected outputs.
    public static float MSE(float[][] networkOutputs, float[][] expectedOutputs){
        float sum = 0f;

        int samples = expectedOutputs.length;
        for (int i = 0; i < samples; i++) {
            float[] error = vectorSubtract(expectedOutputs[i], networkOutputs[i]);
            float norm = norm2(error);
            sum += norm*norm;
        }

        return sum / samples;
    }

    public static float MAE(float[][] networkOutputs, float[][] expectedOutputs){
        float sum = 0f;

        int samples = networkOutputs.length;
        for (int i = 0; i < samples; i++) {
            float[] error = vectorSubtract(expectedOutputs[i], networkOutputs[i]);
            float norm = norm2(error);
            sum += norm;
        }

        return sum / samples;
    }

    public static float[][] MAE_derivative(float[][] networkOutputs,
                                       float[][] expectedOutputs,
                                       float[][] outputPreActivations){
        float[][] difference = vectorSubtract(networkOutputs, expectedOutputs);
        float[][] scaled = scale(difference, 2);

        float[][] activationDerivative = sigmoidDerivative(outputPreActivations);

        return hadamardMultiply(scaled,activationDerivative);
    }

    /**
     * Calculates the derivative of the mean squared error (MSE) cost function
     * with respect to the output activations.
     *
     * Function:
     * 2*(activations - expectedOutputs) *
     * activationDerivative(preActivations)
     */
    public static float[][] MSE_derivative(float[][] networkOutputs,
                                           float[][] expectedOutputs,
                                           float[][] outputPreActivations){
        float[][] difference = vectorSubtract(networkOutputs, expectedOutputs);
        float[][] scaled = scale(difference, 2);

        float[][] activationDerivative = sigmoidDerivative(outputPreActivations);

        return hadamardMultiply(scaled,activationDerivative);
    }

    public static float gradMSE(       float neuronActivation,
                                       float expectedOutput){
        return (-2*(expectedOutput-neuronActivation));
    }

    public static float errorOutput(float neuronActivation,
                                    float neuronPreActivation,
                                    float expectedOutput){
        return gradMSE(neuronActivation,expectedOutput) * sigmoidDerivative(neuronPreActivation);
    }

    /**
     * Computes the error in the output layer.
     */
    public static void errorOutput(     float[][] neuronActivations,
                                        float[][] neuronPreActivations,
                                        float[][] expectedOutputs,
                                        float[][] acc){
        for (int i = 0; i < neuronActivations.length; i++) {
            for (int j = 0; j < neuronActivations[0].length; j++) {
                acc[i][j] = errorOutput(neuronActivations[i][j],
                                        neuronPreActivations[i][j],
                                        expectedOutputs[i][j]);
            }
        }
    }

    /**
     * Computes the error in a given hidden layer (which is a function of the error in the layer succeeding it),
     * inevitably depending on the output error.
     *
     * This layer's output gets weighed by the weights of the next layer, hence the next layer's weights
     * serve as a "significance" of this neuron's output. Low weights in the next layer means a neuron doesn't contribute
     * very much to the ultimate error of the network, and conversely a high one means it does. If the network wasn't
     * that far off though this neuron has a low error anyway.
     */
    public static void errorHidden(float[][] neuronPreActivations,
                                   float[][] nextLayerWeights,
                                   float[][] nextLayerError,
                                   float[][] acc){
        mAT(nextLayerWeights,1F,nextLayerError,1F,acc);
        hadamardMultiplyIP(acc,sigmoidDerivative(neuronPreActivations),acc);
    }

    public static void main(String[] args) {
        float[][] ones = new float[1][16];
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 16; j++) {
                ones[i][j] = 0.125F;
            }
        }

        float[][] zeroes = new float[1][16];
        float mse = MSE(ones, zeroes);
        System.out.println(mse);
    }
}
