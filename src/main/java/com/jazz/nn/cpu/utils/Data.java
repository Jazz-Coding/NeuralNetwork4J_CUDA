package com.jazz.nn.cpu.utils;

public class Data {
    private float[][] inputs;
    private float[][] expectedOutputs;

    public Data(float[][] inputs, float[][] expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

    public float[][] getInputs() {
        return inputs;
    }

    public void setInputs(float[][] inputs) {
        this.inputs = inputs;
    }

    public float[][] getExpectedOutputs() {
        return expectedOutputs;
    }

    public void setExpectedOutputs(float[][] expectedOutputs) {
        this.expectedOutputs = expectedOutputs;
    }
}
