package temp.ea.ann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Helper class for 3D array (volume) operations is used for filter, image data and feature maps.
 * 
 * @author Gemini
 * @version 1.0
 *
 */
class Volume {
	
	
	/**
	 * Internal data as data[depth][height][width].
	 */
	public double[][][] data;
	
	
	/**
	 * Volume depth.
	 */
	public int depth;
    
	
	/**
	 * Volume height.
	 */
	public int height;
	
	
	/**
	 * Volume width.
	 */
	public int width;

	
	/**
	 * Constructor with depth, height, and width.
	 * @param depth depth.
	 * @param height height.
	 * @param width width.
	 */
	public Volume(int depth, int height, int width) {
		this.depth = depth;
		this.height = height;
		this.width = width;
		this.data = new double[depth][height][width];
	}

	
	/**
	 * Constructor as initializing with existing data (e.g., input image, filter).
	 * @param data volume data.
	 */
	public Volume(double[][][] data) {
		this.data = data;
		this.depth = data.length;
		this.height = data[0].length;
		this.width = data[0][0].length;
	}

	
	/**
	 * Get a specific value.
	 * @param d depth.
	 * @param h height.
	 * @param w width.
	 * @return value.
	 */
	public double get(int d, int h, int w) {
		return data[d][h][w];
	}

	
	/**
	 * Setting a specific value.
	 * @param d depth.
	 * @param h height.
	 * @param w width.
	 * @param val value.
	 */
	public void set(int d, int h, int w, double val) {
		data[d][h][w] = val;
	}


	/**
	 * Simple fill function for debugging/resetting.
	 * @param val value.
	 */
	public void fill(double val) {
		for (int d = 0; d < depth; d++) {
			for (int h = 0; h < height; h++) {
				Arrays.fill(data[d][h], val);
			}
		}
	}
    
    
}



/**
 * This class implements Convolutional Layer.
 * @author Gemini
 * @version 1.0
 *
 */
class ConvLayer {
    public int inputDepth, inputHeight, inputWidth;
    public int numFilters;
    public int filterSize;
    public int stride;
    public int outputDepth, outputHeight, outputWidth;

    public Volume[] filters; // Array of filter volumes [filter_size][filter_size][inputDepth]
    public double[] biases; // Bias for each filter

    private Random random;

    // Stored for backpropagation
    private Volume lastInput; // Store input for gradient calculation

    public ConvLayer(int inputDepth, int inputHeight, int inputWidth,
                     int numFilters, int filterSize, int stride) {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.random = new Random();

        // Calculate output dimensions
        this.outputHeight = (inputHeight - filterSize) / stride + 1;
        this.outputWidth = (inputWidth - filterSize) / stride + 1;
        this.outputDepth = numFilters;

        // Initialize filters and biases
        this.filters = new Volume[numFilters];
        this.biases = new double[numFilters];
        for (int i = 0; i < numFilters; i++) {
            // Each filter has dimensions [inputDepth][filterSize][filterSize]
            filters[i] = new Volume(inputDepth, filterSize, filterSize);
            // Initialize filter weights with small random values
            for (int d = 0; d < inputDepth; d++) {
                for (int r = 0; r < filterSize; r++) {
                    for (int c = 0; c < filterSize; c++) {
                        filters[i].set(d, r, c, random.nextDouble() * 0.01 - 0.005); // Small random for weights
                    }
                }
            }
            biases[i] = random.nextDouble() * 0.01; // Small random for biases
        }
    }

    /**
     * Performs the forward pass of the convolutional layer.
     * Applies each filter to the input volume to produce feature maps.
     *
     * @param inputVolume The input image or feature map.
     * @return The output feature map volume.
     */
    public Volume forward(Volume inputVolume) {
        this.lastInput = inputVolume; // Store input for backprop

        Volume outputVolume = new Volume(outputDepth, outputHeight, outputWidth);

        for (int f = 0; f < numFilters; f++) { // For each filter
            for (int h_out = 0; h_out < outputHeight; h_out++) {
                for (int w_out = 0; w_out < outputWidth; w_out++) {
                    double activation = biases[f]; // Start with bias

                    // Calculate receptive field start coordinates
                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    // Convolve filter with receptive field
                    for (int d_in = 0; d_in < inputDepth; d_in++) {
                        for (int kh = 0; kh < filterSize; kh++) {
                            for (int kw = 0; kw < filterSize; kw++) {
                                activation += inputVolume.get(d_in, h_start + kh, w_start + kw) *
                                              filters[f].get(d_in, kh, kw);
                            }
                        }
                    }
                    outputVolume.set(f, h_out, w_out, activation);
                }
            }
        }
        return outputVolume;
    }

    /**
     * Conceptual backward pass for the Convolutional Layer.
     * This is highly simplified for brevity. A full implementation involves:
     * 1. Computing gradients with respect to weights (`dW`).
     * 2. Computing gradients with respect to biases (`db`).
     * 3. Computing gradients with respect to the input (`dInput`) to pass back to the previous layer.
     *
     * In a real implementation:
     * - `dW` involves convolving `lastInput` with `dOut` (rotated).
     * - `db` is simply the sum of `dOut` for each filter.
     * - `dInput` involves convolving `dOut` with `filters` (rotated and padded).
     *
     * @param dOut Gradient from the next layer (Volume of output error).
     * @param learningRate Learning rate for parameter updates.
     * @return Gradient to pass to the previous layer (Volume of input error).
     */
    public Volume backward(Volume dOut, double learningRate) {
        Volume dInput = new Volume(inputDepth, inputHeight, inputWidth);
        Volume[] dFilters = new Volume[numFilters];
        double[] dBiases = new double[numFilters];

        for(int f = 0; f < numFilters; f++) {
            dFilters[f] = new Volume(inputDepth, filterSize, filterSize);
            dBiases[f] = 0.0;
        }

        // --- Simplified Backpropagation logic for illustration ---
        // In reality, this would involve complex convolutions and sums.
        // For dBiases: sum all elements in dOut for that filter.
        for (int f = 0; f < numFilters; f++) {
            for (int h = 0; h < outputHeight; h++) {
                for (int w = 0; w < outputWidth; w++) {
                    dBiases[f] += dOut.get(f, h, w);
                }
            }
        }

        // For dFilters and dInput: This is where complex cross-correlation/convolution operations happen.
        // This simplified loop demonstrates the concept of how gradients are 'spread'.
        // It does NOT represent the correct mathematical operation for convolution backprop.
        for (int f = 0; f < numFilters; f++) {
            for (int h_out = 0; h_out < outputHeight; h_out++) {
                for (int w_out = 0; w_out < outputWidth; w_out++) {
                    double dout_val = dOut.get(f, h_out, w_out);
                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int d_in = 0; d_in < inputDepth; d_in++) {
                        for (int kh = 0; kh < filterSize; kh++) {
                            for (int kw = 0; kw < filterSize; kw++) {
                                // Simplified dFilters update
                                dFilters[f].set(d_in, kh, kw, dFilters[f].get(d_in, kh, kw) + lastInput.get(d_in, h_start + kh, w_start + kw) * dout_val);

                                // Simplified dInput update
                                dInput.set(d_in, h_start + kh, w_start + kw, dInput.get(d_in, h_start + kh, w_start + kw) + filters[f].get(d_in, kh, kw) * dout_val);
                            }
                        }
                    }
                }
            }
        }

        // Update filters and biases
        for (int f = 0; f < numFilters; f++) {
            biases[f] -= learningRate * dBiases[f];
            for (int d = 0; d < inputDepth; d++) {
                for (int r = 0; r < filterSize; r++) {
                    for (int c = 0; c < filterSize; c++) {
                        filters[f].set(d, r, c, filters[f].get(d, r, c) - learningRate * dFilters[f].get(d, r, c));
                    }
                }
            }
        }

        return dInput; // Return gradient to the previous layer
    }
}


// --- Max Pooling Layer Implementation ---
class PoolingLayer {
    public int inputDepth, inputHeight, inputWidth;
    public int poolSize; // Size of the pooling window (e.g., 2x2)
    public int stride;
    public int outputDepth, outputHeight, outputWidth;

    // Store indices of max elements for backpropagation
    private int[][][][] maxIndices; // [depth][h_out][w_out][2] -> {h_in_idx, w_in_idx} of max

    public PoolingLayer(int inputHeight, int inputWidth, int inputDepth, int poolSize, int stride) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputDepth = inputDepth;
        this.poolSize = poolSize;
        this.stride = stride;

        // Calculate output dimensions
        this.outputHeight = (inputHeight - poolSize) / stride + 1;
        this.outputWidth = (inputWidth - poolSize) / stride + 1;
        this.outputDepth = inputDepth; // Depth remains the same in pooling

        this.maxIndices = new int[outputDepth][outputHeight][outputWidth][2];
    }

    /**
     * Performs the forward pass of the max pooling layer.
     * Selects the maximum value within each pooling window.
     *
     * @param inputVolume The input feature map volume.
     * @return The pooled output volume.
     */
    public Volume forward(Volume inputVolume) {
        Volume outputVolume = new Volume(outputDepth, outputHeight, outputWidth);

        for (int d = 0; d < inputDepth; d++) { // For each feature map
            for (int h_out = 0; h_out < outputHeight; h_out++) {
                for (int w_out = 0; w_out < outputWidth; w_out++) {
                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    double maxValue = Double.MIN_VALUE;
                    int max_h_idx = -1;
                    int max_w_idx = -1;

                    // Find max in pooling window
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            double val = inputVolume.get(d, h_start + ph, w_start + pw);
                            if (val > maxValue) {
                                maxValue = val;
                                max_h_idx = h_start + ph;
                                max_w_idx = w_start + pw;
                            }
                        }
                    }
                    outputVolume.set(d, h_out, w_out, maxValue);
                    maxIndices[d][h_out][w_out][0] = max_h_idx; // Store index for backprop
                    maxIndices[d][h_out][w_out][1] = max_w_idx; // Store index for backprop
                }
            }
        }
        return outputVolume;
    }

    /**
     * Conceptual backward pass for the Max Pooling Layer.
     * This is highly simplified for brevity. A full implementation involves:
     * Distributing the gradient (`dOut`) only to the position that was the maximum in the forward pass.
     * All other positions in the pooling window receive a gradient of 0.
     *
     * @param dOut Gradient from the next layer (Volume of output error).
     * @return Gradient to pass to the previous layer (Volume of input error).
     */
    public Volume backward(Volume dOut) {
        Volume dInput = new Volume(inputDepth, inputHeight, inputWidth);
        dInput.fill(0.0); // Initialize with zeros

        for (int d = 0; d < outputDepth; d++) {
            for (int h_out = 0; h_out < outputHeight; h_out++) {
                for (int w_out = 0; w_out < outputWidth; w_out++) {
                    double dout_val = dOut.get(d, h_out, w_out);
                    int max_h_idx = maxIndices[d][h_out][w_out][0];
                    int max_w_idx = maxIndices[d][h_out][w_out][1];
                    // Only the max element gets the gradient from the next layer
                    dInput.set(d, max_h_idx, max_w_idx, dInput.get(d, max_h_idx, max_w_idx) + dout_val);
                }
            }
        }
        return dInput;
    }
}


public class NeuralNetwork {

    // Network architecture parameters
    private int inputHeight, inputWidth, inputDepth; // Dimensions of input image
    private ConvLayer convLayer;
    private PoolingLayer poolingLayer;

    // Parameters for Dense (Fully Connected) layers
    private int flattenedSize; // Size after conv and pooling, before dense layer
    private int hiddenSize;
    private int outputSize;    // Number of classes

    // Weights and biases for Dense layers (matrices/vectors represented as 2D arrays)
    private double[][] weightsHiddenDense;
    private double[][] biasHiddenDense;
    private double[][] weightsOutputDense;
    private double[][] biasOutputDense;

    // --- Variables for Backpropagation (storing intermediate values for Dense layers) ---
    private double[][] flattenedActivations; // Output from flattening
    private double[][] hiddenDenseZs;
    private double[][] hiddenDenseActivations;
    private double[][] outputZs;
    private double[][] outputActivations; // Final prediction

    // --- Accumulated gradients for Mini-Batch Gradient Descent (for Dense layers) ---
    private double[][] gradWeightsHiddenDenseAcc;
    private double[][] gradBiasHiddenDenseAcc;
    private double[][] gradWeightsOutputDenseAcc;
    private double[][] gradBiasOutputDenseAcc;

    private Random random;

    /**
     * Constructor for the NeuralNetwork (CNN architecture).
     * Initializes convolutional, pooling, and dense layers.
     *
     * @param inputHeight  Height of the input image.
     * @param inputWidth   Width of the input image.
     * @param inputDepth   Depth (channels) of the input image (e.g., 1 for grayscale, 3 for RGB).
     * @param numFilters   Number of filters in the convolutional layer.
     * @param filterSize   Size of each convolutional filter (e.g., 3 for 3x3).
     * @param convStride   Stride for convolution.
     * @param poolSize     Size of the pooling window (e.g., 2 for 2x2).
     * @param poolStride   Stride for pooling.
     * @param hiddenSize   Number of neurons in the dense hidden layer.
     * @param outputSize   Number of neurons in the output layer (number of classes).
     */
    public NeuralNetwork(int inputHeight, int inputWidth, int inputDepth,
                         int numFilters, int filterSize, int convStride,
                         int poolSize, int poolStride,
                         int hiddenSize, int outputSize) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputDepth = inputDepth;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.random = new Random();

        // Initialize Convolutional and Pooling Layers
        this.convLayer = new ConvLayer(inputDepth, inputHeight, inputWidth, numFilters, filterSize, convStride);

        // Output dimensions from ConvLayer will be input to PoolingLayer
        int convOutputHeight = convLayer.outputHeight;
        int convOutputWidth = convLayer.outputWidth;
        int convOutputDepth = convLayer.outputDepth;

        this.poolingLayer = new PoolingLayer(convOutputHeight, convOutputWidth, convOutputDepth, poolSize, poolStride);

        // Output dimensions from PoolingLayer will be flattened for Dense layer
        int poolOutputHeight = poolingLayer.outputHeight;
        int poolOutputWidth = poolingLayer.outputWidth;
        int poolOutputDepth = poolingLayer.outputDepth;
        this.flattenedSize = poolOutputHeight * poolOutputWidth * poolOutputDepth;

        // Initialize Dense layers
        // FIX: Corrected dimensions for weightsHiddenDense and weightsOutputDense
        // Weights should be (output_size, input_size) for matrix multiplication (W * X)
        this.weightsHiddenDense = initializeDenseWeights(hiddenSize, flattenedSize); // Corrected
        this.biasHiddenDense = initializeDenseWeights(hiddenSize, 1);
        this.weightsOutputDense = initializeDenseWeights(outputSize, hiddenSize); // Corrected
        this.biasOutputDense = initializeDenseWeights(outputSize, 1);

        // Initialize accumulated gradients for mini-batching (for Dense layers)
        // FIX: Corrected dimensions for gradWeightsHiddenDenseAcc and gradWeightsOutputDenseAcc
        this.gradWeightsHiddenDenseAcc = new double[hiddenSize][flattenedSize]; // Corrected
        this.gradBiasHiddenDenseAcc = new double[hiddenSize][1];
        this.gradWeightsOutputDenseAcc = new double[outputSize][hiddenSize]; // Corrected
        this.gradBiasOutputDenseAcc = new double[outputSize][1];
    }

    /**
     * Initializes a dense layer weight matrix with small random values.
     * Using a slightly different initialization for dense layers.
     */
    private double[][] initializeDenseWeights(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextDouble() * 0.1 - 0.05; // Smaller range for dense weights
            }
        }
        return matrix;
    }

    /**
     * Sigmoid activation function. Used for hidden dense layer.
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Derivative of the sigmoid activation function.
     */
    private double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    /**
     * Softmax activation function. Used for the output layer.
     */
    private double[][] softmax(double[][] z_values) {
        double[][] probabilities = new double[z_values.length][1];
        double sumExp = 0.0;
        for (int i = 0; i < z_values.length; i++) {
            sumExp += Math.exp(z_values[i][0]);
        }
        for (int i = 0; i < z_values.length; i++) {
            probabilities[i][0] = Math.exp(z_values[i][0]) / sumExp;
        }
        return probabilities;
    }

    /**
     * Performs the forward pass through the CNN.
     *
     * @param input An array representing the flattened image features (normalized).
     * @return The network's prediction (probability distribution over classes).
     */
    public double[] forwardPass(double[] input) {
        // 1. Reshape input to Volume for ConvLayer
        Volume inputVolume = new Volume(inputDepth, inputHeight, inputWidth);
        int k = 0;
        for (int d = 0; d < inputDepth; d++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    inputVolume.set(d, h, w, input[k++]);
                }
            }
        }

        // 2. Convolutional Layer Forward Pass
        Volume convOutput = convLayer.forward(inputVolume);
        // Apply ReLU-like activation after convolution for non-linearity (optional, can be done in ConvLayer)
        // For simplicity here, we'll assume linear activation within ConvLayer for Z values,
        // then apply activation to pooled output or within dense layer.
        // A common practice is ReLU after conv, then pooling.
        // Let's apply a non-linear activation (sigmoid) to the conv output before pooling.
        Volume convActivated = applyFunctionToVolume(convOutput, this::sigmoid);


        // 3. Pooling Layer Forward Pass
        Volume poolOutput = poolingLayer.forward(convActivated);

        // 4. Flattening
        flattenedActivations = new double[flattenedSize][1];
        k = 0;
        for (int d = 0; d < poolOutput.depth; d++) {
            for (int h = 0; h < poolOutput.height; h++) {
                for (int w = 0; w < poolOutput.width; w++) {
                    flattenedActivations[k++][0] = poolOutput.get(d, h, w);
                }
            }
        }

        // 5. Dense Hidden Layer Calculation
        hiddenDenseZs = matrixAdd(matrixMultiply(weightsHiddenDense, flattenedActivations), biasHiddenDense);
        hiddenDenseActivations = applyFunction(hiddenDenseZs, this::sigmoid);

        // 6. Dense Output Layer Calculation
        outputZs = matrixAdd(matrixMultiply(weightsOutputDense, hiddenDenseActivations), biasOutputDense);
        outputActivations = softmax(outputZs);

        // Convert final output matrix back to a 1D array
        double[] prediction = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            prediction[i] = outputActivations[i][0];
        }
        return prediction;
    }

    /**
     * Calculates the Categorical Cross-Entropy (CCE) loss.
     */
    public double calculateLoss(double[] predicted, double[] target) {
        double loss = 0;
        double epsilon = 1e-10;
        for (int i = 0; i < outputSize; i++) {
            double clampedPredicted = Math.max(epsilon, Math.min(1 - epsilon, predicted[i]));
            loss += -target[i] * Math.log(clampedPredicted);
        }
        return loss;
    }

    /**
     * Performs the backpropagation algorithm.
     * This orchestrates the backward passes through dense, pooling, and convolutional layers.
     *
     * @param target The actual target (one-hot encoded) for the current input.
     */
    public void backpropagate(double[] target) {
        double[][] targetMatrix = new double[outputSize][1];
        for (int i = 0; i < outputSize; i++) {
            targetMatrix[i][0] = target[i];
        }

        // --- 1. Output Dense Layer Error (Delta) Calculation ---
        // For Softmax with Categorical Cross-Entropy, outputDelta = (A_output - Target)
        double[][] outputDelta = matrixSubtract(outputActivations, targetMatrix);

        // Gradients for output dense layer weights and biases
        // Corrected order for matrix multiplication: delta * input_transposed
        double[][] gradWeightsOutputDense = matrixMultiply(outputDelta, matrixTranspose(hiddenDenseActivations));
        double[][] gradBiasOutputDense = outputDelta;

        addToAccumulatedGradientsDense(gradWeightsOutputDenseAcc, gradWeightsOutputDense);
        addToAccumulatedGradientsDense(gradBiasOutputDenseAcc, gradBiasOutputDense);

        // --- 2. Hidden Dense Layer Error (Delta) Calculation ---
        // Error for hidden dense layer: (W_output_dense_transposed * Delta_output)
        double[][] hiddenDenseError = matrixMultiply(matrixTranspose(weightsOutputDense), outputDelta);
        // Delta for hidden dense layer: Error_hidden_dense * SigmoidDerivative(A_hidden_dense)
        double[][] hiddenDenseDelta = elementWiseMultiply(hiddenDenseError, applyFunction(hiddenDenseActivations, this::sigmoidDerivative));

        // Gradients for hidden dense layer weights and biases
        // Corrected order for matrix multiplication: delta * input_transposed
        double[][] gradWeightsHiddenDense = matrixMultiply(hiddenDenseDelta, matrixTranspose(flattenedActivations));
        double[][] gradBiasHiddenDense = hiddenDenseDelta;

        addToAccumulatedGradientsDense(gradWeightsHiddenDenseAcc, gradWeightsHiddenDense);
        addToAccumulatedGradientsDense(gradBiasHiddenDenseAcc, gradBiasHiddenDense);

        // --- 3. Unflattening the gradient from dense layer to pooling layer ---
        // The gradient that needs to be passed back to the pooling layer (dPoolOutput)
        // is the 'hiddenDenseDelta' reshaped back into a volume.
        Volume dPoolOutput = new Volume(poolingLayer.outputDepth, poolingLayer.outputHeight, poolingLayer.outputWidth);
        int k = 0;
        for (int d = 0; d < dPoolOutput.depth; d++) {
            for (int h = 0; h < dPoolOutput.height; h++) {
                for (int w = 0; w < dPoolOutput.width; w++) {
                    dPoolOutput.set(d, h, w, hiddenDenseDelta[k++][0]);
                }
            }
        }

        // --- 4. Pooling Layer Backward Pass ---
        // This method will distribute the gradient to the original max locations.
        Volume dConvActivated = poolingLayer.backward(dPoolOutput);

        // --- 5. Convolutional Layer Backward Pass ---
        // First, apply derivative of activation function applied after conv layer to the gradient
        Volume dConvOutput = applyFunctionToVolumeElementWise(dConvActivated, this::sigmoidDerivative);

        // This method will calculate gradients for filters and biases of conv layer,
        // and also return the gradient to the input (image) layer.
        // The convLayer.backward method is simplified for illustrative purposes.
        convLayer.backward(dConvOutput, 0.0); // Learning rate is passed to convLayer.backward for its internal updates
                                               // We're setting it to 0.0 here because we're handling the main
                                               // update in updateParameters, but a full implementation would
                                               // update conv layer params directly within its backward call.
    }

    /**
     * Updates the network's dense layer weights and biases using accumulated gradients.
     * Note: Convolutional and Pooling layer parameters would ideally be updated within their
     * respective backward calls or via a separate optimizer for those layers.
     * For this simplified structure, only dense layer updates are explicitly handled here.
     *
     * @param learningRate The learning rate for gradient descent.
     * @param batchSize The number of samples in the mini-batch.
     */
    public void updateParameters(double learningRate, int batchSize) {
        // Average the accumulated gradients over the batch size for Dense layers
        double[][] avgGradWeightsHiddenDense = scalarMultiply(gradWeightsHiddenDenseAcc, 1.0 / batchSize);
        double[][] avgGradBiasHiddenDense = scalarMultiply(gradBiasHiddenDenseAcc, 1.0 / batchSize);
        double[][] avgGradWeightsOutputDense = scalarMultiply(gradWeightsOutputDenseAcc, 1.0 / batchSize);
        double[][] avgGradBiasOutputDense = scalarMultiply(gradBiasOutputDenseAcc, 1.0 / batchSize);

        // Update Dense weights and biases
        weightsHiddenDense = matrixSubtract(weightsHiddenDense, scalarMultiply(avgGradWeightsHiddenDense, learningRate));
        biasHiddenDense = matrixSubtract(biasHiddenDense, scalarMultiply(avgGradBiasHiddenDense, learningRate));
        weightsOutputDense = matrixSubtract(weightsOutputDense, scalarMultiply(avgGradWeightsOutputDense, learningRate));
        biasOutputDense = matrixSubtract(biasOutputDense, scalarMultiply(avgGradBiasOutputDense, learningRate));

        // Reset accumulated gradients for the next mini-batch
        resetAccumulatedGradients();
    }

    /**
     * Resets all accumulated gradient matrices to zero (for Dense layers).
     */
    private void resetAccumulatedGradients() {
        // FIX: Corrected iteration bounds for resetting accumulated gradients
        for (int i = 0; i < hiddenSize; i++) Arrays.fill(gradWeightsHiddenDenseAcc[i], 0.0); // Corrected
        for (int i = 0; i < hiddenSize; i++) Arrays.fill(gradBiasHiddenDenseAcc[i], 0.0);
        for (int i = 0; i < outputSize; i++) Arrays.fill(gradWeightsOutputDenseAcc[i], 0.0); // Corrected
        for (int i = 0; i < outputSize; i++) Arrays.fill(gradBiasOutputDenseAcc[i], 0.0);
    }

    /**
     * Adds the gradients from a single sample to the accumulated gradient matrix (for Dense layers).
     */
    private void addToAccumulatedGradientsDense(double[][] accumulatedGrad, double[][] newGrad) {
        // FIX: Ensure newGrad dimensions match accumulatedGrad for addition
        if (accumulatedGrad.length != newGrad.length || accumulatedGrad[0].length != newGrad[0].length) {
            throw new IllegalArgumentException("Accumulated gradient and new gradient matrices must have the same dimensions for addition.");
        }
        for (int i = 0; i < accumulatedGrad.length; i++) {
            for (int j = 0; j < accumulatedGrad[0].length; j++) {
                accumulatedGrad[i][j] += newGrad[i][j];
            }
        }
    }


    // --- Matrix Utility Methods (from previous example, adapted) ---
    private double[][] matrixMultiply(double[][] A, double[][] B) {
        int aRows = A.length;
        int aCols = A[0].length;
        int bRows = B.length;
        int bCols = B[0].length;
        if (aCols != bRows) throw new IllegalArgumentException("Matrices cannot be multiplied! A columns (" + aCols + ") must equal B rows (" + bRows + ")");
        double[][] C = new double[aRows][bCols];
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                for (int k = 0; k < aCols; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    private double[][] matrixAdd(double[][] A, double[][] B) {
        if (A.length != B.length || A[0].length != B[0].length) throw new IllegalArgumentException("Matrices must have same dimensions for addition.");
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) C[i][j] = A[i][j] + B[i][j];
        }
        return C;
    }

    private double[][] matrixSubtract(double[][] A, double[][] B) {
        if (A.length != B.length || A[0].length != B[0].length) throw new IllegalArgumentException("Matrices must have same dimensions for subtraction.");
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) C[i][j] = A[i][j] - B[i][j];
        }
        return C;
    }

    private double[][] scalarMultiply(double[][] matrix, double scalar) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) result[i][j] = matrix[i][j] * scalar;
        }
        return result;
    }

    private double[][] elementWiseMultiply(double[][] A, double[][] B) {
        if (A.length != B.length || A[0].length != B[0].length) throw new IllegalArgumentException("Matrices must have same dimensions for element-wise multiplication.");
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) C[i][j] = A[i][j] * B[i][j];
        }
        return C;
    }

    private double[][] matrixTranspose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) transposed[j][i] = matrix[i][j];
        }
        return transposed;
    }

    private double[][] applyFunction(double[][] matrix, java.util.function.DoubleUnaryOperator function) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) result[i][j] = function.applyAsDouble(matrix[i][j]);
        }
        return result;
    }

    // Helper to apply function to a Volume (3D array)
    private Volume applyFunctionToVolume(Volume volume, java.util.function.DoubleUnaryOperator function) {
        Volume result = new Volume(volume.depth, volume.height, volume.width);
        for (int d = 0; d < volume.depth; d++) {
            for (int h = 0; h < volume.height; h++) {
                for (int w = 0; w < volume.width; w++) {
                    result.set(d, h, w, function.applyAsDouble(volume.get(d, h, w)));
                }
            }
        }
        return result;
    }

    // Helper to apply element-wise function to a Volume (3D array)
    private Volume applyFunctionToVolumeElementWise(Volume volume, java.util.function.DoubleUnaryOperator function) {
        Volume result = new Volume(volume.depth, volume.height, volume.width);
        for (int d = 0; d < volume.depth; d++) {
            for (int h = 0; h < volume.height; h++) {
                for (int w = 0; w < volume.width; w++) {
                    result.set(d, h, w, function.applyAsDouble(volume.get(d, h, w)));
                }
            }
        }
        return result;
    }


    /**
     * Helper to find the index of the maximum value in an array.
     * This is used to determine the predicted class from the softmax output.
     */
    private static int indexOfMax(double[] array) {
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // --- Main training and testing method ---
    public static void main(String[] args) {
        // --- Simulated Image Data ---
        // Representing 5x5 grayscale images (1 channel).
        // Pixels are normalized (0.0 to 1.0).
        // Each array is a flattened image.
        int imgHeight = 5;
        int imgWidth = 5;
        int imgDepth = 1; // Grayscale
        int numClasses = 3; // Classes: 0, 1, 2

        // Simulated Image Inputs (Flattened and Normalized)
        // Image 0 (Class 0: e.g., a "circle" pattern)
        double[] img0 = {
            0.0, 0.2, 0.8, 0.2, 0.0,
            0.2, 0.8, 1.0, 0.8, 0.2,
            0.8, 1.0, 1.0, 1.0, 0.8,
            0.2, 0.8, 1.0, 0.8, 0.2,
            0.0, 0.2, 0.8, 0.2, 0.0
        };
        // Image 1 (Class 1: e.g., a "square" pattern)
        double[] img1 = {
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0
        };
        // Image 2 (Class 2: e.g., a "triangle" pattern)
        double[] img2 = {
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        };
        // Image 3 (Another Class 0 example, slightly varied)
        double[] img3 = {
            0.1, 0.3, 0.7, 0.3, 0.1,
            0.3, 0.7, 0.9, 0.7, 0.3,
            0.7, 0.9, 0.9, 0.9, 0.7,
            0.3, 0.7, 0.9, 0.7, 0.3,
            0.1, 0.3, 0.7, 0.3, 0.1
        };
        // Image 4 (Another Class 1 example, slightly varied)
        double[] img4 = {
            0.9, 0.9, 0.9, 0.9, 0.9,
            0.9, 0.1, 0.1, 0.1, 0.9,
            0.9, 0.1, 0.1, 0.1, 0.9,
            0.9, 0.1, 0.1, 0.1, 0.9,
            0.9, 0.9, 0.9, 0.9, 0.9
        };
        // Image 5 (Another Class 2 example, slightly varied)
        double[] img5 = {
            0.1, 0.1, 0.9, 0.1, 0.1,
            0.1, 0.9, 0.9, 0.9, 0.1,
            0.9, 0.9, 0.9, 0.9, 0.9,
            0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1
        };

        double[][] inputs = {
            img0, img1, img2, img3, img4, img5
        };

        // Corresponding target outputs (One-Hot Encoded)
        double[][] targets = {
            {1, 0, 0}, // Class 0
            {0, 1, 0}, // Class 1
            {0, 0, 1}, // Class 2
            {1, 0, 0}, // Class 0
            {0, 1, 0}, // Class 1
            {0, 0, 1}  // Class 2
        };

        // --- CNN Architecture Parameters ---
        int numFilters = 8;     // Number of filters in Conv Layer
        int filterSize = 3;     // 3x3 filter
        int convStride = 1;     // Stride for convolution
        int poolSize = 2;       // 2x2 pooling window
        int poolStride = 2;     // Stride for pooling

        int hiddenSize = 20; // Size of the dense hidden layer
        int outputSize = numClasses;

        // Create the Neural Network (CNN)
        NeuralNetwork nn = new NeuralNetwork(imgHeight, imgWidth, imgDepth,
                                             numFilters, filterSize, convStride,
                                             poolSize, poolStride,
                                             hiddenSize, outputSize);

        // Training Hyperparameters
        double learningRate = 0.05; // Adjusted learning rate, often smaller for CNNs
        int epochs = 20000;         // More epochs usually needed for CNNs with complex data
        int batchSize = 2;          // Mini-batch size

        System.out.println("--- Starting CNN Training (Image Classification) ---");
        System.out.println("Image Dimensions: " + imgHeight + "x" + imgWidth + "x" + imgDepth);
        System.out.println("Number of Classes: " + numClasses);
        System.out.println("Conv Layer: Filters=" + numFilters + ", Size=" + filterSize + "x" + filterSize + ", Stride=" + convStride);
        System.out.println("Pool Layer: Size=" + poolSize + "x" + poolSize + ", Stride=" + poolStride);
        System.out.printf("Flattened Size: %d -> Dense Hidden: %d -> Output: %d%n",
                          nn.flattenedSize, hiddenSize, outputSize);
        System.out.println("Learning Rate: " + learningRate + ", Epochs: " + epochs + ", Batch Size: " + batchSize);
        System.out.println("Loss Function: Categorical Cross-Entropy");
        System.out.println("Output Activation: Softmax");

        // List to hold shuffled indices for mini-batching
        List<Integer> dataIndices = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            dataIndices.add(i);
        }

        // Training Loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(dataIndices); // Shuffle data indices for SGD behavior

            double totalLoss = 0;
            int correctPredictions = 0;

            // Iterate through mini-batches
            for (int i = 0; i < inputs.length; i += batchSize) {
                int currentBatchSize = Math.min(batchSize, inputs.length - i);

                for (int j = 0; j < currentBatchSize; j++) {
                    int dataIndex = dataIndices.get(i + j);

                    double[] currentInput = inputs[dataIndex];
                    double[] currentTarget = targets[dataIndex];

                    double[] prediction = nn.forwardPass(currentInput);
                    nn.backpropagate(currentTarget); // Accumulates gradients for dense layers

                    totalLoss += nn.calculateLoss(prediction, currentTarget);

                    int predictedClass = indexOfMax(prediction);
                    int actualClass = indexOfMax(currentTarget);
                    if (predictedClass == actualClass) {
                        correctPredictions++;
                    }
                }
                // Update dense layer parameters after processing the mini-batch
                nn.updateParameters(learningRate, currentBatchSize);
                // Note: In a full CNN, convLayer and poolingLayer parameters would also be updated here
                // or within their backward methods, based on their accumulated gradients.
            }

            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                double accuracy = (double) correctPredictions / inputs.length * 100.0;
                System.out.printf("Epoch %d, Average Loss: %.4f, Training Accuracy: %.2f%%%n",
                                  epoch, totalLoss / inputs.length, accuracy);
            }
        }

        System.out.println("\n--- Training Complete ---");
        System.out.println("--- Testing Trained Network ---");

        int testCorrect = 0;
        System.out.println("Testing on training data (simulated test set)");
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] target = targets[i];
            double[] prediction = nn.forwardPass(input);

            int predictedClass = indexOfMax(prediction);
            int actualClass = indexOfMax(target);

            System.out.printf("Input (first 5 pixels): %s..., Expected Class: %d, Predicted Probabilities: %s, Classified: %d %s%n",
                    Arrays.toString(Arrays.copyOf(input, Math.min(input.length, 5))),
                    actualClass, Arrays.toString(prediction), predictedClass,
                    (predictedClass == actualClass ? "(Correct)" : "(Incorrect)"));

            if (predictedClass == actualClass) {
                testCorrect++;
            }
        }
        System.out.printf("Final Test Accuracy: %.2f%%%n", (double) testCorrect / inputs.length * 100.0);
    }
}
