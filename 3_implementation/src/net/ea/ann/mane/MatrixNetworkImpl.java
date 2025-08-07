/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.filter.DeconvConvFilter;
import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;

/**
 * This class implements matrix neural network in default.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixNetworkImpl extends MatrixNetworkAbstract implements MatrixLayer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default filter stride.
	 */
	final static int BASE_DEFAULT = 2;
	
	
	/**
	 * Default depth.
	 */
	final static int DEPTH_DEFAULT = 6;

	
	/**
	 * Previous layer.
	 */
	protected MatrixLayer prevLayer = null;
	
	
	/**
	 * Next layer.
	 */
	protected MatrixLayer nextLayer = null;
	
	
	/**
	 * List of trainers.
	 */
	protected List<TaskTrainer> trainers = Util.newList(0);
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}
	

	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixNetworkImpl(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	@Override
	public Id getIdRef() {
		return idRef;
	}


	@Override
	public int id() {
		return idRef.get();
	}


	@Override
	protected MatrixLayerAbstract newLayer() {
		MatrixLayerImpl layer = new MatrixLayerImpl(neuronChannel, activateRef, convActivateRef, idRef);
		layer.setNetwork(this);
		return layer;
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1.
	 * @param filter1 filter 1.
	 * @param depth1 the number 1 of hidden layers plus output layer.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1.
	 * @param depth2 the number 2 of hidden layers plus output layer.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension outputSize2, int depth2) {
		if (inputSize1 == null || inputSize1.height <= 0 || inputSize1.width <= 0) return false;
		if ((filter1 != null) && (filter1 instanceof DeconvConvFilter)) filter1 = null;
		if ((filter1 != null) && (filter1.getStrideWidth() < 2 || filter1.getStrideHeight() < 2)) filter1 = null;
		depth1 = depth1 < 0 ? 0 : depth1;
		depth2 = depth2 < 0 ? 0 : depth2;
		dual1 = filter1 != null ? dual1 : false;
		this.layers = null;
		
		//Calculating hidden layer number 1.
		int hBase1 = filter1 != null ? filter1.getStrideHeight() : BASE_DEFAULT;
		int wBase1 = filter1 != null ? filter1.getStrideWidth() : BASE_DEFAULT;
		int[][] numbers = MatrixNetworkInitializer.constructHiddenNeuronNumbers(inputSize1, outputSize1, hBase1, wBase1, depth1);
		if (numbers == null) return false;
		int[] heights = numbers[0];
		int[] widths = numbers[1];
		boolean[] filters = new boolean[heights.length];
		Arrays.fill(filters, filter1 != null);
		
		//Calculating hidden layer number 1.
		if (outputSize2 != null || depth2 > 0) {
			outputSize1 = new Dimension(widths[widths.length-1], heights[heights.length-1]);
			int[][] numbers2 = MatrixNetworkInitializer.constructHiddenNeuronNumbers(outputSize1, outputSize2, hBase1, wBase1, depth2);
			if (numbers2 != null) {
				int hLength = heights.length;
				heights = Arrays.copyOf(heights, hLength + numbers2[0].length);
				for (int i = 0; i < numbers2[0].length; i++) {
					heights[hLength + i] = numbers2[0][i];
				}
				
				int wLength = widths.length;
				widths = Arrays.copyOf(widths, widths.length + numbers2[1].length);
				for (int i = 0; i < numbers2[1].length; i++) {
					widths[wLength + i] = numbers2[1][i];
				}
				
				filters = Arrays.copyOf(filters, hLength + numbers2[0].length);
				Arrays.fill(filters, hLength, hLength + numbers2[0].length, false);
			}
		}
		
		//Constructing size array.
		Dimension[] sizes = new Dimension[1 + heights.length];
		sizes[0] = new Dimension(inputSize1.width, inputSize1.height);
		for (int i = 0; i < heights.length; i++) {
			sizes[i+1] = new Dimension(widths[i], heights[i]);
		}
		if (sizes.length < 2) return false;
		
		//Vectorizing size array. 
		Dimension[] newSizes = sizes;
		if (isVectorized()) {
			newSizes = new Dimension[sizes.length];
			for (int i = 0; i < sizes.length; i++) {
				newSizes[i] = new Dimension(1, sizes[i].height*sizes[i].width);
			}
		}
		
		//Initializing layer.
		List<MatrixLayerAbstract> layers = Util.newList(sizes.length);
		MatrixLayerImpl prevLayer = (MatrixLayerImpl)newLayer();
		if (isVectorized()) prevLayer.setVecRows(sizes[0].height);
		prevLayer.setLearnFilter(isLearnFilter());
		if (!new MatrixLayerInitializer(prevLayer).initialize(newSizes[0]))
			return false;
		layers.add(prevLayer);
		
		Dimension prevSize = prevLayer.getSize();
		if (prevSize.width != newSizes[0].width || prevSize.height != newSizes[0].height) return false;
		Dimension thisSize = prevSize;
		for (int i = 1; i < newSizes.length; i++) {
			int thisVecRows = sizes[i].height;
			MatrixLayerImpl layer = (MatrixLayerImpl)newLayer();
			if (isVectorized()) layer.setVecRows(thisVecRows);
			layer.setLearnFilter(isLearnFilter());
			
			thisSize = newSizes[i];
			prevSize = filters[i-1] ? thisSize : prevSize;
			if (!new MatrixLayerInitializer(layer).initialize(thisSize, prevSize, prevLayer, filters[i-1]?filter1:null))
				return false;
			Dimension currentSize = layer.getSize();
			if (currentSize.width != thisSize.width || currentSize.height != thisSize.height) return false;
			
			layers.add(layer);
			prevLayer = layer;
			prevSize = currentSize;
			if (filter1 == null || !dual1) continue;
			
			thisSize = prevSize;
			MatrixLayerImpl dualLayer = (MatrixLayerImpl)newLayer();
			if (isVectorized()) dualLayer.setVecRows(thisVecRows);
			dualLayer.setLearnFilter(isLearnFilter());
			if (!new MatrixLayerInitializer(dualLayer).initialize(thisSize, prevSize, prevLayer, null))
				return false;
			Dimension dualSize = dualLayer.getSize();
			if (dualSize.width != thisSize.width || dualSize.height != thisSize.height) return false;
			
			layers.add(dualLayer);
			prevLayer = dualLayer;
			prevSize = dualSize;
		}
		this.layers = layers.toArray(new MatrixLayerAbstract[] {});
		
		//Adjusting layers by removing redundant filters.
		for (int i = 1; i < newSizes.length; i++) {
//			MatrixLayerAbstract layer = this.layers[i];
//			if (layer.getFilter() == null) continue;
//			Filter2D filter = layer.getFilter();
//			ConvLayerSingle2D convLayer = layer.getPrevInputConvLayer();
//			int H = convLayer.getHeight(), h = filter.getStrideHeight();
//			int W = convLayer.getWidth(), w = filter.getStrideWidth();
//			if (H < h || W < w) {
//				layer.removeFilter();
//				continue;
//			}
			
//			MatrixLayerAbstract prevOutputLayer = layer.getPrevLayer();
//			if (prevOutputLayer == null) continue;
//			ConvLayerSingle2D prevConvOutputLayer = prevOutputLayer.getPrevInputConvLayer();
//			if (prevConvOutputLayer == null) continue;
//			if (prevConvOutputLayer.getHeight() < H+h || prevConvOutputLayer.getWidth() < W+w)
//				layer.removeFilter();
		}
		
		return true;
	}
	
	
	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1.
	 * @param filterStride1 filter stride 1.
	 * @param depth1 the number 1 of hidden layers plus output layer.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1.
	 * @param depth2 the number 2 of hidden layers plus output layer.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension inputSize1, Dimension outputSize1, Dimension filterStride1, int depth1, boolean dual1, Dimension outputSize2, int depth2) {
		Filter2D filter = defaultFilter(filterStride1);
		return initialize(inputSize1, outputSize1, filter, depth1, dual1, outputSize2, depth2);
	}
	
	
	@Override
	public MatrixLayer getPrevLayer() {
		return prevLayer;
	}


	@Override
	public MatrixLayer getNextLayer() {
		return nextLayer;
	}


	/**
	 * Adapting this output to input of next layer.
	 * @param thisOutput this output.
	 * @param nextLayer next layer.
	 * @return input of next layer.
	 */
	protected Matrix adaptOutputToNextInput(Matrix thisOutput, MatrixLayer nextLayer) {
		return thisOutput;
	}
	
	
	/**
	 * Adapting this input to output of previous layer.
	 * @param thisInput this input.
	 * @param prevLayer previous layer.
	 * @return output of previous layer.
	 */
	protected Matrix adaptInputToPrevOutput(Matrix thisInput, MatrixLayer prevLayer) {
		return thisInput;
	}
	
	
	@Override
	public Matrix getInput() {
		return getInputLayer().getInput();
	}


	@Override
	public Matrix getOutput() {
		return getOutputLayer().queryOutput();
	}


	/**
	 * Getting size of trainers.
	 * @return size of trainers.
	 */
	int getTrainerSize() {
		return trainers.size();
	}
	
	
	/**
	 * Getting trainer at specified index.
	 * @param index specified index.
	 * @return trainer at specified index.
	 */
	TaskTrainer getTrainer(int index) {
		return trainers.get(index);
	}
	
	
	/**
	 * Getting trainer.
	 * @return the first trainer.
	 */
	public TaskTrainer getTrainer() {
		return trainers.size() > 0 ? trainers.get(0) : null;
	}
	
	
	/**
	 * Adding trainer.
	 * @param trainer specified trainer.
	 * @return adding is successful.
	 */
	boolean addTrainer(TaskTrainer trainer) {
		return trainers.add(trainer);
	}
	
	
	/**
	 * Removing trainer.
	 * @param trainer specified trainer.
	 * @return removal is successful.
	 */
	boolean removeTrainer(TaskTrainer trainer) {
		return trainers.remove(trainer);
	}
	
	
	/**
	 * Clearing trainer.
	 */
	void clearTrainers() {
		trainers.clear();
	}
	
	
	/**
	 * Setting trainer.
	 * @param trainer specified trainer.
	 * @return this network.
	 */
	public MatrixNetworkImpl setTrainer(TaskTrainer trainer) {
		trainers.clear();
		if (trainer != null) trainers.add(trainer);
		return this;
	}
	
	
	@Override
	public Matrix forward(Matrix input) {
		Matrix result = evaluate(input, new Object[] {});
		if (result == null) return result;
		
		MatrixLayer nextLayer = null;
		while ((nextLayer = this.getNextLayer()) != null) {
			result = adaptOutputToNextInput(result, nextLayer);
			Matrix.copy(result, nextLayer.getInput());
			result = nextLayer.evaluate();
		}
		return result;
	}


	@Override
	public Matrix evaluate(Matrix input) throws RemoteException {
		return evaluate(input, new Object[] {});
	}

	
	@Override
	public Matrix evaluate() {
		return evaluate(null, new Object[] {});
	}


	/**
	 * Evaluating matrix neural network.
	 * @param input input matrix for evaluating.
	 * @param params other parameters.
	 * @return array as output.
	 */
	private Matrix evaluate(Matrix input, Object...params) {
		MatrixLayerAbstract inputLayer = getInputLayer();
		if (input != null) Matrix.copy(input, inputLayer.getInput());
		if (inputLayer.getOutput() != inputLayer.getInput()) inputLayer.setOutput(inputLayer.getInput());
		
		for (int i = 1; i < layers.length; i++) layers[i].evaluate();
		return getOutputLayer().queryOutput();
	}
	
	
	@Override
	public Matrix[] learn(Iterable<Matrix[]> inouts) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learn(inouts, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning matrix neural network.
	 * @param inouts sample as collection of input and output whose each element is an 2-component array of input (the first) and output (the second).
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learning errors.
	 */
	private Matrix[] learn(Iterable<Matrix[]> inouts, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Matrix[] outputErrors = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			inouts = resample(inouts, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			if (trainers.size() == 0) {
				List<Matrix> outputErrorList = Util.newList(0);
				for (Matrix[] inout : inouts) {
					Matrix input = inout[0], realOutput = inout[1];
					Matrix output = evaluate(input, new Object[] {});
					Matrix error = calcOutputError(output, realOutput, getOutputLayer());
					outputErrorList.add(error);
				}
				outputErrors = outputErrorList.toArray(new Matrix[] {});
				outputErrors = backward(outputErrors, this, true, lr);
			}
			else {
				for (TaskTrainer trainer : trainers) {
					outputErrors = trainer.train(this, inouts, false, learningRate);
				}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "mane_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (outputErrors == null || outputErrors.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = Matrix.normMean(outputErrors);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}//End while
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "mane_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return outputErrors;
	}

	
	@Override
	public Matrix[] backward(Matrix[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (!validate() || outputErrors == null) return null;
		if (focus == null) learning = true;
		
		outputErrors = Arrays.copyOf(outputErrors, outputErrors.length);
		for (int i = layers.length-1; i >= 0; i--) {
			outputErrors = layers[i].backward(outputErrors, layers[i], true, learningRate);
		}
		if (outputErrors == null || this.prevLayer == null || this == focus) return outputErrors;
		
		Matrix[] backwardErrors = new Matrix[outputErrors.length];
		for (int i = 0; i < outputErrors.length; i++)
			backwardErrors[i] = adaptInputToPrevOutput(outputErrors[i], this.prevLayer);
		return this.prevLayer.backward(backwardErrors, focus, learning, learningRate);
	}

	
	/**
	 * Backward learning.
	 * @param outputErrors output errors.
	 * @param learningRate learning rate.
	 * @return learning errors.
	 */
	public Matrix[] backward(Matrix[] outputErrors, double learningRate) {
		return backward(outputErrors, null, true, learningRate);
	}
	
	
//	@Override
//	public Matrix[] backward(Matrix[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
//		if (!validate()) return null;
//		if (outputErrors == null || outputErrors.length == 0) return null;
//		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
//		if (focus == null) learning = true;
//		if (learning) focus = null;
//
//		Matrix[] errors = new Matrix[outputErrors.length];
//		Matrix[] nextErrors = new Matrix[outputErrors.length];
//		Matrix[] dW1s = new Matrix[outputErrors.length];
//		Matrix[] dW2s = new Matrix[outputErrors.length];
//		NeuronValue[] dFilterErrors = new NeuronValue[outputErrors.length];
//		NeuronValue[][][] dFilterKernels = new NeuronValue[outputErrors.length][][];
//		
//		//Browsing backward layers.
//		for (int i = layers.length-1; i >= 0; i--) {
//			MatrixLayerAbstract layer = layers[i];
//			MatrixLayerAbstract prevLayer = layers[i-1];
//			MatrixLayerAbstract nextLayer = layers[i+1];
//			
//			//Browsing errors.
//			for (int j = 0; j < outputErrors.length; j++) {
//				if (i == layers.length-1) {
//					//Getting errors from environment.
//					errors[j] = outputErrors[j];
//					
//					//Training adapter here.
//				}
//				else {
//					if (nextLayer.getFilter() == null) {
//						Matrix input = layer.getInput(); //X'k-1
//						Matrix output = layer.queryOutput(); //Xk
//						Matrix derivative = input != null ? input.derivativeWise(layer.getActivateRef()) : null;
//						
//						//Updating errors based on weights.
//						Matrix nextW1T = nextLayer.getWeight1();
//						Matrix nextW2 = nextLayer.getWeight2();
//						nextW1T = (nextW1T != null) ? nextW1T.transpose() : output.createIdentity(output.rows());
//						nextW2 = (nextW2 != null) ? nextW2 : output.createIdentity(output.columns());
//						
//						Matrix[] errorArray = new Matrix[nextW2.rows()];
//						Matrix vecNextError = nextErrors[j].vec(); //Please pay attention to this code line.
//						for (int row = 0; row < errorArray.length; row++) {
//							//errorArray[row] = Matrix.kroneckerProductMutilply(nextW2, nextW1T, row, vecNextError);
//							errorArray[row] = nextW2.kroneckerProductRowOf(nextW1T, row).multiply(vecNextError); //Faster.
//						}
//						errors[j] = Matrix.concatV(errorArray);
//						errors[j] = derivative != null ? derivative.multiplyWise(errors[j]) : errors[j];
//					}
//					else {
//						errors[j] = nextErrors[j]; //Please pay attention to this code line.
//					}
//					
//				} //Calculating errors[j]
//
//				//Updating nextErrors[j] by filter.
//				if (layer.getFilter() != null) {
//					ConvLayerSingle2D prevLayer2D = prevLayer.matrixToConvLayer(prevLayer.queryOutput());
//					ConvLayerSingle2D errorj = layer.matrixToConvLayer(errors[j]);
//					NeuronValueRaster dValues = prevLayer2D.dValue(errorj, layer.getFilter());
//					//Please pay attention to this code line to assign current errors to next errors.
//					nextErrors[j] = prevLayer.arrayToMatrix(dValues.getValues(), prevLayer2D.getHeight(), prevLayer2D.getWidth());
//					
//					if (layer.isLearnFilter()) {
//						dFilterErrors[j] = dValues.getCountValues() > 0 ? Matrix.valueSum(nextErrors[j]).divide(dValues.getCountValues()) :
//							nextErrors[j].get(0, 0).zero(); //Filter errors.
//						dFilterKernels[j] = prevLayer2D.dKernel(errorj, layer.getFilter()); //Filter kernel errors.
//					}
//				}
//
//				//Update weight errors[j].
//				if (layer.containsWeights()) {
//					Matrix W1 = layer.getWeight1();
//					Matrix W2 = layer.getWeight2();
//					Matrix prevInput = layer.getPrevInput();
//					prevInput = prevInput != null ? prevInput : prevLayer.queryOutput(); //Xk-1
//					
//					Matrix vecError = errors[j].vec();
//					if (W1 != null) {
//						Matrix XW2 = W2 != null ? prevInput.multiply(W2) : prevInput;
//						Matrix I = W1.createIdentity(W1.rows());
//						Matrix[] W1s = new Matrix[XW2.rows()];
//						for (int row = 0; row < W1s.length; row++) {
//							//W1s[row] = Matrix.kroneckerProductMutilply(XW2, I, row, vecError); //Lower but consuming less memory.
//							W1s[row] = XW2.kroneckerProductRowOf(I, row).multiply(vecError); //Faster.
//						}
//						dW1s[j] = Matrix.concatV(W1s);
//					}
//					
//					if (W2 != null) {
//						Matrix W1XT = W1 != null ? W1.multiply(prevInput) : prevInput;
//						W1XT = W1XT.transpose();
//						Matrix I = W2.createIdentity(W2.columns());
//						Matrix[] W2s = new Matrix[I.rows()];
//						for (int row = 0; row < W2s.length; row++) {
//							//W2s[row] = Matrix.kroneckerProductMutilply(I, W1XT, row, vecError); //Lower but consuming less memory.
//							W2s[row] = I.kroneckerProductRowOf(W1XT, row).multiply(vecError); //Faster.
//						}
//						dW2s[j] = Matrix.concatV(W2s);
//					}
//				} //Updating error of W1 and W2
//				
//			} //End browsing errors.
//			
//			
//			//Update weight bias, first weight, and second weight.
//			if (layer.getBias() != null && learning) {
//				Matrix biasMean = Matrix.mean(errors);
//				Matrix bias = layer.getBias().add(biasMean.multiply0(learningRate));
//				layer.setBias(bias);
//			}
//			if (layer.getWeight1() != null && learning) {
//				Matrix w1Mean = Matrix.mean(dW1s);
//				Matrix w1 = layer.getWeight1().add(w1Mean.multiply0(learningRate));
//				layer.setWeight1(w1);
//			}
//			if (layer.getWeight2() != null && learning) {
//				Matrix w2Mean = Matrix.mean(dW2s);
//				Matrix w2 = layer.getWeight2().add(w2Mean.multiply0(learningRate));
//				layer.setWeight2(w2);
//			}
//			
//			//Update filter and filter bias.
//			if (layer.getFilter() != null && layer.isLearnFilter() && learning) {
//				NeuronValue filterErrorsMean = NeuronValue.valueMean(dFilterErrors);
//				NeuronValue filterBias = layer.getFilterBias().add(filterErrorsMean.multiply(learningRate));
//				layer.setFilterBias(filterBias); //Update filter bias.
//				
//				if (layer.getFilter() instanceof ProductFilter2D) {
//					ProductFilter2D filter = (ProductFilter2D)layer.getFilter();
//					NeuronValue[][] filterKernelsMean = ProductFilter2D.kernelMean(dFilterKernels);
//					filterKernelsMean = NeuronValue.multiply(filterKernelsMean, learningRate);
//					filter = filter.shallowClone();
//					filter.accumKernel(filterKernelsMean);
//					layer.setFilter(filter); //Update filter.
//				}
//			}
//			
//			//Please pay attention to this code line to assign current errors to next errors.
//			if (layer.getFilter() == null) nextErrors = errors;
//		}
//
//		//Returning errors if there is no previous layers;
//		if (nextErrors == null || prevLayer == null) return nextErrors;
//		//Stop at focused layer.
//		if (this == focus) return nextErrors;
//		
//		//Browsing backward layers.
//		Matrix[] backwardErrors = new Matrix[nextErrors.length];
//		for (int i = 0; i < outputErrors.length; i++) {
//			backwardErrors[i] = adaptInputToPrevOutput(outputErrors[i], prevLayer);
//		}
//		return prevLayer.backward(backwardErrors, focus, learning, learningRate);
//	}
	

}
