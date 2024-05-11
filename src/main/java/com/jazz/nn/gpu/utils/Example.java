package com.jazz.nn.gpu.utils;

import com.jazz.nn.cpu.utils.PrettyPrint;
import com.jazz.nn.cpu.utils.VectorMath;

public class Example {
    private float[] input;
    private float[] output;

    public Example(float[] input, float[] output) {
        this.input = input;
        this.output = output;
    }

    public float[] getInput() {
        return input;
    }
    public float[] getOutput() {
        return output;
    }

    public int inputSize(){
        return input.length;
    }
    public int expectedOutputSize(){
        return output.length;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        float[][] squared = VectorMath.inflate(input,28,28);
        int label = VectorMath.argMax(output);

        sb.append(PrettyPrint.matrixPrint(squared));
        sb.append("=").append(label);

        return sb.toString();
    }
}
