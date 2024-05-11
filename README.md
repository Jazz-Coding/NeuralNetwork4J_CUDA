# NeuralNetwork4J_CUDA #
Handwritten digit classifier written in Java using the GPU for accelerated training and inference. Trained on the MNIST handwritten digit dataset (included in .csv format split into training and test data).
 - JCublas and JCuda libraries serve as interface with native Cublas and Cuda libraries (versions >=12.0 must be installed beforehand).

By using the GPU, significant speedups over CPU-based training are achieved.

The main code for training/inference can be found in [nn/gpu/NN_GPU.java](https://github.com/Jazz-Coding/NeuralNetwork4J_CUDA/blob/master/src/main/java/com/jazz/nn/gpu/NN_GPU.java)

## Currently available ##
**Layer Types**:

\- _Fully-connected_

  
**Training Algorithms**:

\- _Stochastic Gradient Descent (SGD)_

  
**Cost Functions**:

\- _Mean-squared error (MSE)_

  
**Activation Functions**:

\- _Sigmoid_
  


**Parameter saving**: Saving/loading from custom local file formats (examples are provided under "saved_networks").

## Example Performance ##
![Network Performance](https://github.com/Jazz-Coding/NeuralNetwork4J_CUDA/assets/52354702/851ae09e-ffb8-4008-ae32-bc93e341c462)


_Test Accuracy=**95.96%**_

_Network specifications: 784x32x10 (i.e. a single hidden layer with 32 neurons)_

_(AKA. saved_networks/tiny.txt)_

_Hyper-parameters: Batch size=32, Learning rate = 0.1_

Training took approximately 1 minute on my machine (with a RTX 4090 GPU). GPU utilization peaked at only 20% on such a small network, but larger networks such as 784x3000x10 (saved_networks/wide.txt) take the same amount of time to train and utilize nearly 100%.
