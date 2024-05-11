package com.jazz.nn.cpu;

import com.jazz.nn.cpu.utils.Data;

import static com.jazz.nn.cpu.utils.ActivationFunctions.sigmoid;
import static com.jazz.nn.cpu.utils.CostFunctions.*;
import static com.jazz.nn.cpu.utils.PrettyPrint.matrixPrint;
import static com.jazz.nn.cpu.utils.RandomArrays.fillRandomly;
import static com.jazz.nn.cpu.utils.VectorMath.*;

public class NN_CPU {
    private static float[][][] weights;
    private static float[][] biases;

    // Cache of information produced on the forward pass for use in training.
    private static float[][][] activationCache;
    private static float[][][] preActivationCache;
    public static float[][][] layerErrorCache;

    // Nielsen network specification, e.g.
    // {784,30,10} = 784 input neurons, 30 hidden neurons, 10 output neurons
    private static int[] networkDimensions = {5,128,16,2};
    private static final int maxEpochs = 30;
    private static int batchSize = 512;
    private static final float learningRate = 0.001F;

    private static void initializeParameters(){
        int layerCount = networkDimensions.length;

        biases = new float[layerCount-1][];
        weights = new float[layerCount-1][][];

        activationCache    = new float[layerCount][][];
        preActivationCache = new float[layerCount-1][][];
        layerErrorCache    = new float[layerCount-1][][];
        trainingScalars = new float[networkDimensions.length];

        for (int i = 1; i < layerCount; i++) {
            int neurons = networkDimensions[i];
            int inboundConnections = networkDimensions[i-1];

            activationCache[i-1] = new float[neurons][batchSize];
            preActivationCache[i-1] = new float[neurons][batchSize];
            layerErrorCache[i-1] = new float[neurons][batchSize];

            biases[i-1] = new float[neurons];
            fillRandomly(biases[i-1]);

            weights[i-1] = new float[neurons][inboundConnections];
            fillRandomly(weights[i-1]);

            trainingScalars[i-1] = -learningRate / (activationCache[i-1].length);
        }
    }

    public static void loadExistingNetwork(int[] HnetworkDimensions, int bs, float[][][] Hweights, float[][] Hbiases){
        networkDimensions = HnetworkDimensions;
        batchSize = bs;
        initializeParameters();
        weights = Hweights;
        biases = Hbiases;
    }
    // Input is a matrix due to the possibility of supplying a whole batch
    // at once instead of only a single example.
    public static float[][] feedforward(float[][] inputs){
        //int batchSize = inputs[0].length; // Batch size = columns in the input
        activationCache[0] = inputs;

        for (int l = 0; l < weights.length; l++) {
            multiplyIP(weights[l], activationCache[l], preActivationCache[l]);

            addIPWithBroadcast(preActivationCache[l], biases[l]);
            //preActivationCache[l] = preActivations;

            float[][] activations = sigmoid(preActivationCache[l]);
            activationCache[l+1] = activations;
            inputs = activations; // Form the inputs to the next layer.
        }

        return inputs;
    }

    private static float[][] feedforwardEmpty(float[][] inputs){
        int batchSize = inputs[0].length; // Batch size = columns in the input

        for (int l = 0; l < weights.length; l++) {
            float[][] batchBiases    = vectorBroadcast(biases[l], batchSize);
            float[][] preActivations = add(multiply(weights[l], inputs), batchBiases);

            inputs = sigmoid(preActivations); // Form the inputs to the next layer.
        }

        return inputs;
    }

    private static float evaluateLoss(float[][] inputs,
                                      float[][] expectedOutputs){
        float[][] output = feedforwardEmpty(inputs);
        return MSE(transpose(output),transpose(expectedOutputs));
    }

    /**
     * The error in the j'th neuron in the output layer.
     */
    /*private static float outputLayerError(int j){
        // gradient of cost wrt. activation of this neuron
        // multiplied by
        // gradient of sigmoid wrt. the pre-activation of this neuron

        // the cost function is 1/n sum_j(expectedOutput_j - activation_j)^2
        // This sum vanishes when considering only a single neuron, as does the
        // averaging over n, since n=1.

        // Hence. the gradient wrt. this neuron's activation is:
        // -2*(expectedOutput_j - activation_j) = 2*(activation_j - expectedOutput_j)


    }*/

    /**
     * Evaluates the error in the neurons for each layer, given the expected
     * outputs.
     */
    public static void calculateLayerErrors(float[][] expectedOutputs){
        int layerCount = networkDimensions.length;
        int finalLayer = layerCount-1;

        //int rows = preActivationCache[preActivationCache.length-1].length;
        //int cols = preActivationCache[preActivationCache.length-1][0].length;

        //float[][] outputError = new float[rows][cols];
                /*MSE_derivative(activationCache[activationCache.length-1],
                        expectedOutputs,
                        preActivationCache[preActivationCache.length-1]);*/
        errorOutput(
                activationCache[activationCache.length-1],
                preActivationCache[preActivationCache.length-1],
                expectedOutputs,
                layerErrorCache[layerErrorCache.length-1]);

        //layerErrorCache[layerErrorCache.length-1] = outputError;

        int finalWeightsIndex = weights.length-1;
        // The "error" in the output neurons across the whole batch.
        //layerErrorCache[parameterIndex] = outputError;

        // Backpropagate the error through the previous layers.
        // The loop goes from the penultimate layer (hidden) through the remainder
        // of the hidden layers, without touching the input layer (index 0), since
        // we have no influence over the network input.
        int backwardIterations = 0;
        for (int currentLayer = finalLayer-1; currentLayer > 0; currentLayer--) { // finalLayer = 4-1=3; currentLayer = finalLayer-1 = 2;
            backwardIterations++;
            //int nextLayer = currentLayer+1; // nextLayer = currentLayer+1 = 2+1=3; -> implies access to 4 objects
            // What influence did these neurons have on the next layer's error?

           /* float[][] activationGradient = sigmoidDerivative(preActivationCache[preActivationCache.length-1-backwardIterations]);
            *//*float[][] weightInfluence =    multiply(transpose(weights[finalWeightsIndex-backwardIterations+1]),
                    layerErrorCache[layerErrorCache.length-1-backwardIterations+1]);*//*

            float[][] weightInfluence = new float[activationGradient.length][activationGradient[0].length];
            multiplySmart(weights[finalWeightsIndex-backwardIterations+1],1F,true,
                          layerErrorCache[layerErrorCache.length-1-backwardIterations+1],1F,false,
                          weightInfluence);*/
            //outputError = hadamardMultiply(weightInfluence,activationGradient);
             errorHidden(
                     preActivationCache[preActivationCache.length-1-backwardIterations],
                     weights[finalWeightsIndex-backwardIterations+1],
                     layerErrorCache[layerErrorCache.length-1-backwardIterations+1],
                     layerErrorCache[layerErrorCache.length-1-backwardIterations]);//hadamardMultiply(weightInfluence,activationGradient);
        }
    }

    /**
     * After layer errors have been calculated, use them to update the
     * weights and biases.
     */
    private static float[] trainingScalars;
    private static void updateParameters(float learningRate){
        for (int i = 0; i < layerErrorCache.length; i++) {
            //float[][] layerError = layerErrorCache[i];
            //float[][] biasUpdates = layerError;

            //float[][] inputsHere = activationCache[i];

            /*System.out.println(matrixPrint(inputsHere));
            System.out.println("*");
            System.out.println(matrixPrint(layerError));*/
            //int cols = inputsHere.length;
            //float[][] weightUpdates = multiply(layerError,transpose(inputsHere));
            // The multiplication in the weight updates implicitly summed, we need
            // only scale properly.
            //int cols = weightUpdates[0].length;
            //float[][] weightUpdatesAverage = scale(weightUpdates,1F/cols);
            //System.out.println(matrixPrint(biasUpdates));
            //float[] biasUpdateAveraged     = averageColumns(biasUpdates);

            // Perform the actual updates, take a step in accordance with the learning rate.
            //float[][] weightChanges = scale(weightUpdatesAverage, -learningRate);
            //float[] biasChanges     = scale(biasUpdateAveraged, -learningRate);

            /*System.out.println(matrixPrint(weights[i]));
            System.out.println(matrixPrint(weightChanges));*/

            //float scalar = (-learningRate/cols);
            multiplySmart(  layerErrorCache[i],1F,false,
                            activationCache[i],trainingScalars[i],true,
                            weights[i]);
            averageColumnsAndScaleIP(layerErrorCache[i],-learningRate,biases[i]);
            //addIP(weights[i],weightChanges);
            //addIP(biases[i],biasChanges);
        }
    }

    /**
     * Evaluate the gradient across the batch and average it. Then use this
     * information to perform an update to the network's weights and biases via
     * gradient descent.
     */
    private static void trainOnMinibatch(float[][] inputs,
                                         float[][] expectedOutputs) {
        feedforward(inputs);
        calculateLayerErrors(expectedOutputs);
        updateParameters(learningRate);
    }

    private static boolean sumEven(float[] vector){
        // Sum rounds to odd or even?
        float sum = 0f;
        for (float v : vector) {
            sum += v;
        }
        return Math.round(sum) % 2 == 0;
    }

    /**
     * Generates some training data.
     * @param inputNeurons test1
     * @param outputNeurons test2
     * @param quantity test3
     * @return Formatted training data ready for training.
     */
    private static Data generateData(int inputNeurons, int outputNeurons, int quantity){
        // Classification. Does the sum of the inputs round to an odd number or an even number?
        // [1,0] = the sum is odd
        // [0,1] = the sum is even

        float[][] inputMatrix = new float[quantity][inputNeurons];
        float[][] outputMatrix = new float[quantity][outputNeurons];
        for (int i = 0; i < quantity; i++) {
            float[] input = new float[inputNeurons];
            float[] output = new float[outputNeurons];

            // Generate columns.
            fillRandomly(input);

            // Sum rounds to odd or even?
            if(sumEven(input)){
                output[0] = 0;
                output[1] = 1;
            } else {
                output[0] = 1;
                output[1] = 0;
            }

            inputMatrix[i] = input;
            outputMatrix[i] = output;
        }

        return new Data(transpose(inputMatrix),transpose(outputMatrix));
    }

    private static float accuracy(Data testData){
        float[][] inputs = testData.getInputs();
        float[][] trueOutputs = feedforward(inputs);
        float[][] expectedOutputs = testData.getExpectedOutputs();

        float[][] transposedInputs = transpose(inputs);
        float[][] transposedTrueOutputs = transpose(trueOutputs);
        float[][] transposedExpectedOutputs = transpose(expectedOutputs);

        int correct = 0;
        for (int i = 0; i < transposedInputs.length; i++) {
            float[] instance = transposedTrueOutputs[i];
            float[] correctInstance = transposedExpectedOutputs[i];
            if(argMax(instance) == argMax(correctInstance)){
                correct++;
            }
        }

        return ((float)correct / transposedInputs.length);
    }

    public static void main(String[] args) {
        initializeParameters();

       /* System.out.println("Network parameters: ");
        System.out.println("Weights: ");
        System.out.println(matrixPrint(weights));
        //System.out.println(Arrays.deepToString(weights));

        System.out.println("Biases: ");
        System.out.println(matrixPrint(biases));*/
        //System.out.println(Arrays.deepToString(biases));

        /*float[][] inputs = new float[5][batchSize];
        fillRandomly(inputs);*/

        System.out.println("Some manual tests:");
        float[][] input = new float[5][1];
        fillRandomly(input);
        System.out.println(matrixPrint(input));
        System.out.println("Sum even?");
        System.out.println(sumEven(input[0]));
        System.out.println("Network output?");
        System.out.println(matrixPrint(feedforwardEmpty(input)));

        // We will attempt to learn the identity function, inputs = outputs.
        for (int i = 0; i < Integer.MAX_VALUE; i++) {
            //System.out.println("Iteration: " + i);
            Data trainingData = generateData(5, 2, batchSize);
            long start = System.nanoTime();
            trainOnMinibatch(trainingData.getInputs(), trainingData.getExpectedOutputs());
            long end = System.nanoTime();
            long duration = end-start;

            float itsPerSecond = 1e9F/duration;

            if(i%1000 == 0) {
                System.out.println("it= " + i + " MSE: " + evaluateLoss(trainingData.getInputs(), trainingData.getExpectedOutputs()) + "\nits/sec=" + itsPerSecond);
            }
        }

        System.out.println("Same manual tests:");
        System.out.println(matrixPrint(input));
        System.out.println("Sum even?");
        System.out.println(sumEven(input[0]));
        System.out.println("Network output?");
        System.out.println(matrixPrint(feedforwardEmpty(input)));

        System.out.println("Classification accuracy: ");
        Data testData = generateData(5,2,2048*16);
        float accuracy = accuracy(testData);
        System.out.println(accuracy*100F + "%");
    }
}
