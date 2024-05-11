package com.jazz.nn.gpu;

import com.jazz.nn.cpu.utils.Data;
import com.jazz.nn.gpu.utils.*;
import com.jazz.nn.logging.Logger;
import com.jazz.nn.logging.Timer;

import java.util.List;

import static com.jazz.nn.gpu.utils.MatrixMath.*;
import static com.jazz.nn.cpu.utils.RandomArrays.fillRandomly;
import static com.jazz.nn.cpu.utils.RandomArrays.fillRandomlyScaled;
import static com.jazz.nn.cpu.utils.VectorMath.*;

/**
 * MNIST handwritten digit classifier using the GPU for accelerated training and inference.
 * JCublas and JCuda libraries serve as interface with native Cublas and Cuda libraries (must be installed beforehand).
 */
public class NN_GPU {
    // Values needed to infer matrix dimensions.
    private static int[] networkDimensions;
    private static int batchSize;
    private static float learningRate;

    // Actual matrices. The "_" prefix denotes a GPU (device) pointer to distinguish it from a CPU (host) value.
    private static MatrixPointer[] _weights;
    private static MatrixPointer[] _biases;
    private static MatrixPointer _minibatchInputs;
    private static MatrixPointer _minibatchExpectedOutputs;

    // Caches of information produced on the forward pass for use in training.
    private static MatrixPointer[] _activationCache;
    private static MatrixPointer[] _preActivationCache;
    private static MatrixPointer[] _layerErrorCache;

    private static MatrixPointer[][] trainingBatches;

    // Intermediate matrices used to store the result of sub-computations.
    private static MatrixPointer _outputDiffPlaceHolder; // same as activs
    private static MatrixPointer[] _sigDerivPlaceHolders; // same as pre-activs
    private static MatrixPointer[] _weightInfluencePlaceHolders; // same as weights

    /**
     * Launches GPU backend and plans memory layout for the intermediate values used during training.
     */
    public static void init(int[] HnetworkDimensions, float HlearningRate, int HbatchSize){
        MatrixMath.initialize();
        networkDimensions = HnetworkDimensions;
        learningRate = HlearningRate;
        batchSize = HbatchSize;

        int layerCount = networkDimensions.length;
        int trainableLayerCount = layerCount - 1;

        // Initialize caches.
        _activationCache = new MatrixPointer[layerCount];
        _preActivationCache = new MatrixPointer[trainableLayerCount];
        _layerErrorCache = new MatrixPointer[trainableLayerCount];

        int outputNeurons = networkDimensions[networkDimensions.length-1];
        _outputDiffPlaceHolder = MatrixPointer.zeroMatrix(outputNeurons, batchSize);

        _sigDerivPlaceHolders = new MatrixPointer[trainableLayerCount];
        _weightInfluencePlaceHolders = new MatrixPointer[trainableLayerCount];
        for (int i = 1; i < layerCount; i++) {
            int neurons = networkDimensions[i];

            _activationCache[i] = MatrixPointer.zeroMatrix(neurons, batchSize);
            _preActivationCache[i-1] = MatrixPointer.zeroMatrix(neurons, batchSize);

            _sigDerivPlaceHolders[i-1] = MatrixPointer.zeroMatrix(neurons,batchSize);
            _weightInfluencePlaceHolders[i-1] = MatrixPointer.zeroMatrix(neurons,batchSize);
            _layerErrorCache[i-1] = MatrixPointer.zeroMatrix(neurons, batchSize);
        }
    }

    /**
     * Utils for loading MNIST data.
     */
    private static DatasetLoader dl = new DatasetLoader();
    public static void loadDataset(){
        dl.loadMnist();
    }

    private static void unloadSafe(MatrixPointer matrixPointer){
        if (matrixPointer != null) matrixPointer.unload();
    }

    /**
     * Subdivides the training data into mini-batches.
     * These batches are random samples from the larger dataset, and the resulting learning process
     * is called stochastic gradient descent (SGD), from this random sampling process.
     */
    public static void batchDataset(int batchSize){
        List<List<Example>> batches = dl.batch(batchSize);
        trainingBatches = new MatrixPointer[batches.size()][2];
        // One column for the input data. One column for the expected output (the label).
        // We intend to learn a relationship between the two to predict the label from the input.

        for (int i = 0; i < batches.size(); i++) {
            List<Example> batch = batches.get(i);

            float[][][] result = dl.batchFloats(batch);

            unloadSafe(trainingBatches[i][0]);
            trainingBatches[i][0] = new MatrixPointer(result[0]);

            unloadSafe(trainingBatches[i][1]);
            trainingBatches[i][1] = new MatrixPointer(result[1]);
        }
    }

    /**
     * Given a CPU (host) set of matrices, allocate and load the network parameters onto the GPU (device).
     * In future, the GPU can use its VRAM for training and inference, increasing GPU utilization and
     * performance considerably due to less time spent on I/O between host and device.
     */
    public static void loadNetworkOntoGPU(float[][][] weights, float[][] biases){
        _weights = new MatrixPointer[weights.length];
        _biases = new MatrixPointer[biases.length];

        for (int i = 0; i < weights.length; i++) {;
            _weights[i] = new MatrixPointer(weights[i]);
            _biases[i] = new MatrixPointer(vectorBroadcast(biases[i],batchSize));
        }
    }

    public static void loadBatchInputsOntoGPU(float[][] batchInputs, float[][] expectedOutputs){
        unloadSafe(_minibatchInputs);
        _minibatchInputs = new MatrixPointer(batchInputs);

        unloadSafe(_minibatchExpectedOutputs);
        _minibatchExpectedOutputs = new MatrixPointer(expectedOutputs);

        // Set the first pointer in the activation cache to the batch inputs.
        unloadSafe(_activationCache[0]);
        _activationCache[0] = new MatrixPointer(batchInputs);

    }

    /**
     * Forward pass (inference).
     * Each layer's output is calculated as:
     * f(x) = a(W[][]*x[] + b[])
     * Where:
     * - a() is the activation function (sigmoid in this case),
     * - W[][] is the layer's weights
     * - b[] is the layer's biases.
     */
    private static void feedforward(){
        int trainableLayerCount = _weights.length;
        for (int l = 0; l < trainableLayerCount; l++) {
            // Weights * inputs:
            MatrixMath.executeMultiplicationCM(
                    _weights[l],
                    1F,
                    _activationCache[l], // activationCache is reused for initial inputs too
                    0F, // 0 = replace whatever is in there already (1 would be FMA with existing data)
                    _preActivationCache[l]); // weights * inputs

            // + biases:
            MatrixMath.executeAddition(
                    _preActivationCache[l],
                    1F,
                    _biases[l],
                    _preActivationCache[l]); // Assumes biases are pre-broadcasted.

            // Activation function:
            executeSigmoid(_preActivationCache[l], _activationCache[l+1]);
        }
    }

    /**
     * To train the network with backpropagation, we calculate the relative error of each layer in the network.
     * Layers with a greater error receive greater updates to their weights and biases.
     *
     * Put simply, the error is calculated with the calculus chain rule.
     * Linear algebra is used to apply the chain rule to the matrices involved.
     */
    private static void calculateLayerErrors(){
        int layerCount = networkDimensions.length;
        int finalLayer = layerCount-1;

        executeMSEDerivative(
                _activationCache[_activationCache.length-1],
                _minibatchExpectedOutputs,
                _outputDiffPlaceHolder,
                _preActivationCache[_preActivationCache.length-1],
                _sigDerivPlaceHolders[_sigDerivPlaceHolders.length-1],
                _layerErrorCache[_layerErrorCache.length-1]);

        int finalWeightsIndex = _weights.length-1;
        // The "error" in the output neurons across the whole batch.
        //layerErrorCache[parameterIndex] = outputError;

        // Backpropagate the error through the previous layers.
        // The loop goes from the penultimate layer (hidden) through the remainder
        // of the hidden layers, without touching the input layer (index 0), since
        // we have no influence over the network input.

        int backwardIterations = 0;
        for (int currentLayer = finalLayer-1; currentLayer > 0; currentLayer--) { // finalLayer = 4-1=3; currentLayer = finalLayer-1 = 2;
            backwardIterations++;
            // What influence did these neurons have on the next layer's error?
            executeSigmoidPrime(
                    _preActivationCache[_preActivationCache.length-1-backwardIterations],
                    _sigDerivPlaceHolders[_sigDerivPlaceHolders.length-1-backwardIterations]);

            executeMultiplicationCMAT(_weights[finalWeightsIndex-backwardIterations+1],   1F,
                    _layerErrorCache[_layerErrorCache.length-1-backwardIterations+1], 0F,
                    _weightInfluencePlaceHolders[_weightInfluencePlaceHolders.length-1-backwardIterations]);

            executeHadamard(_weightInfluencePlaceHolders[_weightInfluencePlaceHolders.length-1-backwardIterations],
                    1F,_sigDerivPlaceHolders[_sigDerivPlaceHolders.length-1-backwardIterations],
                    _layerErrorCache[_layerErrorCache.length-1-backwardIterations]);
        }
    }

    /**
     * Use calculated layer errors to update the weights and biases of the network to (hopefully)
     * improve its accuracy.
     * The "learningRate" parameter controls the speed at which this happens.
     * A value too high can lead to unstable (poor) learning.
     * A value too low may take too long to achieve good accuracy.
     * In practice we set this heuristically or via a brute-force grid search of possible parameters.
     */
    private static void updateParameters(){
        for (int i = 0; i < _layerErrorCache.length; i++) {
            float scalar = -learningRate/batchSize;

            executeMultiplicationCMBT( // weights += learningRate * (error * inputs at the time)
                    _layerErrorCache[i],scalar,
                    _activationCache[i],1F,
                    _weights[i]);

            executeAvgBroadcastAdd(_layerErrorCache[i],_biases[i],scalar); // biases += learningRate * (error at the time)
        }
    }

    private static boolean sumEven(float[] vector){
        // Sum rounds to odd or even?
        float sum = 0f;
        for (float v : vector) {
            sum += v;
        }
        return Math.round(sum) % 2 == 0;
    }


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

    /**
     * Free all memory on the GPU. Avoids memory leaks.
     */
    private static void cleanUp(){
        _minibatchInputs.unload();
        for (int i = 0; i < _weights.length; i++) {
            _weights[i].unload();
            _biases[i].unload();
            _preActivationCache[i].unload();
            _layerErrorCache[i].unload();
        }

        for (MatrixPointer matrixPointer : _activationCache) {
            matrixPointer.unload();
        }

        _minibatchExpectedOutputs.unload();
        _minibatchInputs.unload();
        for (MatrixPointer sigDerivPlaceHolder : _sigDerivPlaceHolders) {
            sigDerivPlaceHolder.unload();
        }

        _outputDiffPlaceHolder.unload();
        for (MatrixPointer weightInfluencePlaceHolder : _weightInfluencePlaceHolders) {
            weightInfluencePlaceHolder.unload();
        }
    }

    /**
     * Evaluates the network's performance on test data (data it has never seen before).
     * Randomly guessing each time, the network would be correct about 10% of the time (as there are 10 classes).
     * Anything above this (and in practice we get FAR above this), indicates successful training.
     *
     * Currently this executes on the CPU, and significantly slows down training, but is useful for seeing if we are on the right track with parameters.
     * Once we are, we can evaluate less often and do more training.
     */
    private static float evaluateOnTestData(int batchSize){
        List<Example> testingExamples = dl.getAllTestingExamples();

        float[][][] giantBatch = dl.batchFloats(testingExamples);
        int n_testing_examples = giantBatch[0][0].length;

        int correct = 0;
        for (int i = 0; i < n_testing_examples; i+=batchSize) {
            List<Example> section = testingExamples.subList(i, Math.min(i + batchSize, n_testing_examples));
            float[][][] section_tests = dl.batchFloats(section);

            float[][] testInputs = section_tests[0];
            float[][] testExpectedOutputs = section_tests[1];

            loadBatchInputsOntoGPU(testInputs, testExpectedOutputs);
            feedforward();

            // Read the result off the GPU
            float[][] networkOutput = _activationCache[_activationCache.length-1].unload();

            // Transposed forms.
            float[][] networkOutput_T = transpose(networkOutput);
            float[][] expectedOutputs_T = transpose(testExpectedOutputs);
            int testBatchSize = expectedOutputs_T.length;

            for (int j = 0; j < Math.min(batchSize,testBatchSize); j++) {
                int networkGuess = argMax(networkOutput_T[j]);
                int answer = argMax(expectedOutputs_T[j]);

                if(answer == networkGuess){
                    correct++;
                }
            }
        }
        float score = correct / (float) n_testing_examples;
        float pct = score*100f;

        return pct;
    }


    /**
     * Do some training on an existing network.
     * Trains for a certain number of "epochs".
     * One epoch involves going over all available training data in mini-batches, performing updates after each one.
     * @param network The packed neural network parameters. May be loaded from file or randomly generated.
     * @param epochs The number of epochs to train for. One epoch spans the whole dataset batched by batchSize.
     * @param batchSize The size of the mini-batches the dataset will be divided into. Typically around n=32.
     * @param learningRate Learning speed. Typically around 0.1.
     * @param testFrequency Test on test data every testFrequency epochs. Set to -1 to disable testing until the very end.
     * @param log Logger instance.
     */
    private static void train(PackedNetwork network, int epochs, int batchSize, float learningRate, int testFrequency, Logger log){
        float[][][] weights = network.getWeights();
        float[][] biases= network.getBiases();
        int[] networkDimensions = network.getNetworkDimensions();

        log.info("Initializing GPU backend and allocating space for network outputs...");
        init(networkDimensions,learningRate,batchSize);

        log.info("Loading parameters onto the GPU...");
        loadNetworkOntoGPU(weights,biases);

        log.info("Loading dataset...");
        loadDataset();

        log.info("Beginning training...");
        float accuracy = evaluateOnTestData(batchSize);
        log.debug("Accuracy (test):", accuracy + "%");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            batchDataset(batchSize); // Split into mini-batches and load them onto the GPU.
            log.info("Epoch: " + epoch);

            // Perform learning cycles on each mini-batch.
            for (MatrixPointer[] trainingBatch : trainingBatches) {
                MatrixPointer inputs = trainingBatch[0];
                MatrixPointer expectedOutputs = trainingBatch[1];

                // Memory manage and switch existing pointers.
                unloadSafe(_minibatchInputs);
                _minibatchInputs = inputs;

                unloadSafe(_minibatchExpectedOutputs);
                _minibatchExpectedOutputs = expectedOutputs;

                unloadSafe(_activationCache[0]);
                _activationCache[0] = _minibatchInputs;

                // Train.
                feedforward();
                calculateLayerErrors();
                updateParameters();
            }

            // Evaluate on test set.
            if(testFrequency > 0 && (epoch%testFrequency==0)) {
                accuracy = evaluateOnTestData(batchSize);
                log.debug("Accuracy (test):", accuracy + "%");
            }
        }

        if(testFrequency <= 0){
            accuracy = evaluateOnTestData(batchSize);
            log.debug("Accuracy (test):", accuracy + "%");
        }
    }

    private static PackedNetwork newRandomNetwork(int[] networkDimensions, Logger log){
        log.info("Initializing new network parameters randomly...");
        // Randomly initialize the network parameters (weights and biases).
        float[][][] w = new float[networkDimensions.length-1][][];
        float[][] b = new float[networkDimensions.length-1][];
        for (int i = 1; i < networkDimensions.length; i++) {
            int neurons = networkDimensions[i];
            int inboundConnections = networkDimensions[i-1];
            b[i-1] = new float[neurons];
            fillRandomly(b[i-1]);

            w[i-1] = new float[neurons][inboundConnections];
            fillRandomlyScaled(w[i-1],inboundConnections);
        }

        return new PackedNetwork(networkDimensions,w,b);
    }
    private static void saveNetworkToFile(NetworkSerializer networkSerializer, String name){
        float[][][] newWeights = new float[_weights.length][][];
        float[][] newBiases = new float[_biases.length][];
        for (int i = 0; i < _weights.length; i++) {
            newWeights[i] = _weights[i].peek();
        }
        for (int i = 0; i < _biases.length; i++) {
            float[][] biases = transpose(_biases[i].peek());
            newBiases[i] = biases[0]; // The rest of the columns are merely copies for batching purposes.
        }
        networkSerializer.save(networkDimensions,newWeights,newBiases,name);
    }


    /**
     * A basic performance test.
     */
    private static void testPerformance(PackedNetwork network, int batchSize, float learningRate, Logger log){
        log.debug("--------------PERFORMANCE TEST--------------");
        float[][][] weights = network.getWeights();
        float[][] biases= network.getBiases();
        int[] networkDimensions = network.getNetworkDimensions();

        init(networkDimensions,learningRate,batchSize);

        Timer t1 = new Timer("loadNetworkOntoGPU");
        t1.start();
        loadNetworkOntoGPU(weights,biases);
        t1.stop();
        t1.report(log);

        Timer t2 = new Timer("loadDataset");
        t2.start();
        loadDataset();
        t2.stop();
        t2.report(log);

        Timer t5 = new Timer("batchDataset");
        t5.start();
        batchDataset(batchSize); // Split into mini-batches and load them onto the GPU.
        t5.stop();
        t5.report(log);

        Timer t3 = new Timer("train(epochs=1)");

        t3.start();
        // Perform learning cycles on each mini-batch.
        for (MatrixPointer[] trainingBatch : trainingBatches) {
            MatrixPointer inputs = trainingBatch[0];
            MatrixPointer expectedOutputs = trainingBatch[1];

            // Memory manage and switch existing pointers.
            unloadSafe(_minibatchInputs);
            _minibatchInputs = inputs;

            unloadSafe(_minibatchExpectedOutputs);
            _minibatchExpectedOutputs = expectedOutputs;

            unloadSafe(_activationCache[0]);
            _activationCache[0] = _minibatchInputs;

            // Train.
            feedforward();
            calculateLayerErrors();
            updateParameters();
        }

        t3.stop();
        t3.report(log);

        Timer t4 = new Timer("evaluateOnTestData");
        t4.start();
        float accuracy = evaluateOnTestData(batchSize);
        t4.stop();
        t4.report(log);
    }

    /**
     * The piloting code.
     * Trains for a certain number of "epochs".
     * One epoch involves going over all available training data in mini-batches, performing updates after each one.
     */
    public static void main(String[] args) {
        Logger log = new Logger();
        NetworkSerializer networkSerializer = new NetworkSerializer("saved_networks");

        networkDimensions = new int[]{784, 32, 10}; // {input neurons, hidden1, hidden2..., output neurons}
        PackedNetwork neuralNetwork = newRandomNetwork(networkDimensions, log);
        //PackedNetwork neuralNetwork = networkSerializer.load("tiny.txt");

        train(neuralNetwork,100,32,0.1F,1,log);

        saveNetworkToFile(networkSerializer,"new_network.txt");
        cleanUp();
    }
}
