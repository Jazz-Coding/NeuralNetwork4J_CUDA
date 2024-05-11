package com.jazz.nn.gpu.utils;

public class PackedNetwork {
    int[] networkDimensions;
    private float[][][] weights;
    private float[][] biases;

    public PackedNetwork(int[] networkDimensions, float[][][] weights, float[][] biases) {
        this.networkDimensions = networkDimensions;
        this.weights = weights;
        this.biases = biases;
    }

    public int[] getNetworkDimensions() {
        return networkDimensions;
    }

    public float[][][] getWeights() {
        return weights;
    }

    public float[][] getBiases() {
        return biases;
    }

    public static String commaSeparatedIntArray(int[] arr){
        StringBuilder CSA = new StringBuilder();
        for (int o : arr) {
            CSA.append(o).append(",");
        }

        CSA.deleteCharAt(CSA.length()-1); // trim last comma
        return CSA.toString();
    }
    public static String commaSeparatedFloatArray(float[] arr){
        StringBuilder CSA = new StringBuilder();
        for (float o : arr) {
            CSA.append(o).append(",");
        }

        CSA.deleteCharAt(CSA.length()-1); // trim last comma
        return CSA.toString();
    }
    public static String slashSeparatedFloatMatrix(float[][] mat){
        StringBuilder sb = new StringBuilder();
        for (float[] row : mat) {
            String rowString = commaSeparatedFloatArray(row);
            sb.append(rowString).append("/");
        }
        sb.deleteCharAt(sb.length()-1); // trim last slash
        return sb.toString();
    }
    public static String barSeparatedFloatTensor(float[][][] tensor){
        StringBuilder sb = new StringBuilder();
        for (float[][] mat : tensor) {
            String matString = slashSeparatedFloatMatrix(mat);
            sb.append(matString).append("|");
        }
        sb.deleteCharAt(sb.length()-1); // trim last bar
        return sb.toString();
    }

    public static int[] unpackCommaSeparatedIntArray(String arrString){
        String[] components = arrString.split(",");
        int length = components.length;
        int[] arr = new int[length];
        for (int i = 0; i < length; i++) {
            arr[i] = Integer.parseInt(components[i]);
        }
        return arr;
    }

    public static float[] unpackCommaSeparatedFloatArray(String arrString){
        String[] components = arrString.split(",");
        int length = components.length;
        float[] arr = new float[length];
        for (int i = 0; i < length; i++) {
            arr[i] = Float.parseFloat(components[i]);
        }
        return arr;
    }

    public static float[][] unpackSlashSeparatedFloatMatrix(String matString){
        String[] components = matString.split("/");
        int length = components.length;
        float[][] mat = new float[length][];
        for (int i = 0; i < length; i++) {
            mat[i] = unpackCommaSeparatedFloatArray(components[i]);
        }
        return mat;
    }
    public static float[][][] unpackBarSeparatedFloatTensor(String tensorString){
        String[] components = tensorString.split("\\|");
        int length = components.length;
        float[][][] tensor = new float[length][][];
        for (int i = 0; i < length; i++) {
            tensor[i] = unpackSlashSeparatedFloatMatrix(components[i]);
        }
        return tensor;
    }

    @Override
    public String toString() {
        return commaSeparatedIntArray(networkDimensions) + "\n" +
                barSeparatedFloatTensor(weights) + "\n" +
                slashSeparatedFloatMatrix(biases);
    }
}
