package com.jazz.nn.gpu.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static com.jazz.nn.gpu.utils.PackedNetwork.*;

public class NetworkSerializer {
    private String saveDirectory = "";

    public NetworkSerializer(String saveDirectory) {
        this.saveDirectory = saveDirectory;
    }

    /**
     * Saves trained weights and biases to a file.
     */
    public void save(int[] networkDimensions, float[][][] networkWeights, float[][] networkBiases, String name){
        PackedNetwork packedNetwork = new PackedNetwork(networkDimensions, networkWeights, networkBiases);

        Path path = Paths.get(saveDirectory + "/" + (name.endsWith(".txt") ? name : (name + ".txt")));
        try {
            Files.writeString(path,packedNetwork.toString());
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    /**
     * Loads network parameters from a file.
     */
    public PackedNetwork load(String fileName){
        String netString;
        try {
            netString = Files.readString(Paths.get(saveDirectory + "/" + (fileName.endsWith(".txt") ? fileName : (fileName+".txt"))));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String[] upperComponents = netString.split("\n");

        String networkDimensionsString = upperComponents[0];
        String networkWeightsString = upperComponents[1];
        String networkBiasesString = upperComponents[2];

        int[] networkDimensions = unpackCommaSeparatedIntArray(networkDimensionsString);
        float[][][] networkWeights = unpackBarSeparatedFloatTensor(networkWeightsString);
        float[][] networkBiases = unpackSlashSeparatedFloatMatrix(networkBiasesString);

        return new PackedNetwork(networkDimensions,networkWeights,networkBiases);
    }
}
