package com.jazz.nn.gpu.utils;

import com.jazz.nn.cpu.utils.Data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static com.jazz.nn.cpu.utils.VectorMath.inflate;
import static com.jazz.nn.cpu.utils.VectorMath.transpose;

public class Dataset {
    private Data trainingData;
    private Data testData;

    private float[][] trainingInputs;
    private float[][] trainingOutputs;

    private float[][] testInputs;
    private float[][] testOutputs;

    private List<Data> trainingMinibatches;
    private List<Data> testingMinibatches;

    private static float[] labelToArgMaxVector(int label, int outputs){
        float[] vector = new float[outputs];
        vector[label] = 1F;
        return vector;
    }

    private static float[] pixelValuesToFlattenedVector(String[] pixelValues, int inputs, int indexOffset){
        float[] vector = new float[inputs];
        for (int i = indexOffset; i < pixelValues.length; i++) {
            vector[i] = Float.parseFloat(pixelValues[i]) / 255F;
        }
        return vector;
    }

    private static float[][] flatPixelsToMatrix(float[] flat, int rows, int cols){
        return inflate(flat,rows,cols);
    }

    public List<Data> batchTrainingData(int batchSize){
        for (int i = 0; i < trainingInputs.length; i+=batchSize) {
            int endpoint = Math.min(i+batchSize,trainingInputs.length);
            int thisBatchSize = endpoint-i;

            float[][] minibatchInputs = new float[trainingInputs[0].length][thisBatchSize];
            float[][] minibatchOutputs = new float[trainingOutputs[0].length][thisBatchSize];

            int index = 0;
            for (int j = i; j < endpoint; j++) {
                minibatchInputs[index] = trainingInputs[j];
                minibatchOutputs[index] = trainingOutputs[j];
                index++;
            }

            trainingMinibatches.add(new Data(transpose(minibatchInputs),transpose(minibatchOutputs)));
        }

        return trainingMinibatches;
    }

    public Dataset(String trainingDataCSVPath, String testDataCSVPath) {
        int inputSize = 28*28;
        int outputSize = 10*1;
        try {
            List<String> trainingSamplesLines = Files.readAllLines(Paths.get(trainingDataCSVPath));
            trainingInputs = new float[trainingSamplesLines.size()][inputSize];
            trainingOutputs = new float[trainingSamplesLines.size()][outputSize];

            for (int i = 0; i < trainingSamplesLines.size(); i++) {
                String[] components = trainingSamplesLines.get(i).split(",");
                int label = Integer.parseInt(components[0]);
                float[] trainingOutput = labelToArgMaxVector(label, outputSize);
                float[] trainingInput = pixelValuesToFlattenedVector(components, inputSize, 1);

                trainingInputs[i] = trainingInput;
                trainingOutputs[i] = trainingOutput;
            }

            // Convert into column vectors.
            /*trainingInputs = transpose(trainingInputs);
            trainingOutputs = transpose(trainingOutputs);*/

            List<String> testingSamplesLines = Files.readAllLines(Paths.get(testDataCSVPath));
            testInputs = new float[testingSamplesLines.size()][inputSize];
            testOutputs = new float[testingSamplesLines.size()][outputSize];

            for (int i = 0; i < testingSamplesLines.size(); i++) {
                String[] components = testingSamplesLines.get(i).split(",");
                int label = Integer.parseInt(components[0]);
                float[] trainingOutput = labelToArgMaxVector(label, outputSize);
                float[] trainingInput = pixelValuesToFlattenedVector(components, inputSize, 1);

                testInputs[i] = trainingInput;
                testOutputs[i] = trainingOutput;
            }

            // Convert into column vectors.
            /*testInputs = transpose(testInputs);
            testOutputs = transpose(testOutputs);*/
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
