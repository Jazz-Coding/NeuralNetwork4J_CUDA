package com.jazz.nn.gpu.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.jazz.nn.cpu.utils.PrettyPrint.matrixPrint;
import static com.jazz.nn.cpu.utils.VectorMath.transpose;

public class DatasetLoader {
    private List<Example> trainingExamples;
    private List<Example> testingExamples;

    public List<Example> loadFromCSV(String csvFilePath, boolean skipFirstLine) throws IOException {
        List<Example> examples = new ArrayList<>();

        List<String> lines = Files.readAllLines(Paths.get(csvFilePath));

        int i = skipFirstLine ? 1 : 0;
        for (; i < lines.size(); i++) {
            String line = lines.get(i);

            String[] csvComponents = line.split(",");

            String label = csvComponents[0];

            // Convert to an arg-max form.
            float[] vector = new float[10];
            int index = Integer.parseInt(label);
            vector[index] = 1F;

            float[] pixels = new float[csvComponents.length-1];
            for (int j = 1; j < csvComponents.length; j++) {
                String pixel = csvComponents[j];

                // 0 = black, 1 = white
                float normalized = Float.parseFloat(pixel)/255F;
                pixels[j-1]=normalized;
            }

            // And these define an example.
            Example example = new Example(pixels, vector);
            examples.add(example);
        }
        return examples;
    }

    public List<List<Example>> batch(int batchSize){
        List<List<Example>> batches = new ArrayList<>();

        Collections.shuffle(trainingExamples);
        for (int i = 0; i < trainingExamples.size(); i+=batchSize) {
            int endpoint = Math.min(trainingExamples.size(),i+batchSize);
            batches.add(trainingExamples.subList(i,endpoint));
        }
        return batches;
    }

    public float[][][] getSomeTestData(int n){
        Collections.shuffle(testingExamples);
        return batchFloats(testingExamples.subList(0,n));
    }

    public List<Example> getAllTestingExamples() {
        return testingExamples;
    }

    // result[0] = inputs, result[1] = expectedOutputs
    public float[][][] batchFloats(List<Example> batch){
        int batchSize = batch.size();

        int inputSize = batch.get(0).inputSize();
        int outputSize = batch.get(0).expectedOutputSize();

        float[][] inputs = new float[batchSize][inputSize];
        float[][] expectedOutputs = new float[batchSize][outputSize];

        for (int i = 0; i < batch.size(); i++) {
            inputs[i] = batch.get(i).getInput();
            expectedOutputs[i] = batch.get(i).getOutput();
        }

        // Transpose both so each column is a new input.
        float[][][] result = new float[2][][];

        result[0] = transpose(inputs);
        result[1] = transpose(expectedOutputs);

        return result;
    }

    public void loadMnist(){
        try {
            trainingExamples = loadFromCSV("datasets/mnist/mnist_train.csv",false);
            testingExamples = loadFromCSV("datasets/mnist/mnist_test.csv",false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        DatasetLoader datasetLoader = new DatasetLoader();
        datasetLoader.loadMnist();
        List<List<Example>> batches = datasetLoader.batch(32);

        for (List<Example> batch : batches) {
            float[][][] result = datasetLoader.batchFloats(batch);
            System.out.println(matrixPrint(result[0]));
            System.out.println(matrixPrint(result[1]));
        }
    }
}
