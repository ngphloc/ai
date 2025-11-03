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
import java.util.Collections;
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
	public final static int BASE_DEFAULT = ZOOMOUT_DEFAULT;
	
	
	/**
	 * Default depth.
	 */
	public final static int DEPTH_DEFAULT = 6;

	
	/**
	 * Default value of minimum width field.
	 */
	public final static int MINSIZE = 32 / BASE_DEFAULT;

	
//	/**
//	 * History size.
//	 */
//	@Deprecated
//	public final static int HISTORY_SIZE = 1000;
//	
//	
//	/**
//	 * Field of history mode.
//	 */
//	@Deprecated
//	public final static String HISTORY_MODE_FIELD = "mane_hist_mode";
	
	
	/**
	 * Default value of history mode.
	 */
	public final static boolean HISTORY_MODE_DEFAULT = false;
	
	
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
	
	
//	/**
//	 * History of evaluations.
//	 */
//	@Deprecated
//	private List<Inout> history = Util.newList(0);
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
//		this.config.put(HISTORY_MODE_FIELD, HISTORY_MODE_DEFAULT);
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
	
	
	@Override
	public void reset() {
		super.reset();
//		history.clear();
	}


	/**
	 * Initializing matrix neural network.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 1, which can be null.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
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
		int[][] numbers = MatrixNetworkInitializer.constructHiddenOutputNeuronNumbers(inputSize1, outputSize1, hBase1, wBase1, depth1);
		if (numbers == null) return false;
		int[] heights = numbers[0];
		int[] widths = numbers[1];
		boolean[] filters = new boolean[heights.length];
		Arrays.fill(filters, filter1 != null);
		
		//Calculating hidden layer number 1.
		if (outputSize2 != null || depth2 > 0) {
			outputSize1 = new Dimension(widths[widths.length-1], heights[heights.length-1]);
			int[][] numbers2 = MatrixNetworkInitializer.constructHiddenOutputNeuronNumbers(outputSize1, outputSize2, hBase1, wBase1, depth2);
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
		if (paramIsVectorized()) {
			newSizes = new Dimension[sizes.length];
			for (int i = 0; i < sizes.length; i++) {
				newSizes[i] = new Dimension(1, sizes[i].height*sizes[i].width);
			}
		}
		
		//Initializing layer.
		List<MatrixLayerAbstract> layers = Util.newList(sizes.length);
		MatrixLayerImpl prevLayer = (MatrixLayerImpl)newLayer();
		if (paramIsVectorized()) prevLayer.setVecRows(sizes[0].height);
		prevLayer.setLearnFilter(paramIsLearnFilter());
		if (!new MatrixLayerInitializer(prevLayer).initialize(newSizes[0]))
			return false;
		layers.add(prevLayer);
		
		Dimension prevSize = prevLayer.getSize();
		if (prevSize.width != newSizes[0].width || prevSize.height != newSizes[0].height) return false;
		Dimension thisSize = prevSize;
		for (int i = 1; i < newSizes.length; i++) {
			int thisVecRows = sizes[i].height;
			MatrixLayerImpl layer = (MatrixLayerImpl)newLayer();
			if (paramIsVectorized()) layer.setVecRows(thisVecRows);
			layer.setLearnFilter(paramIsLearnFilter());
			
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
			if (paramIsVectorized()) dualLayer.setVecRows(thisVecRows);
			dualLayer.setLearnFilter(paramIsLearnFilter());
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
		
		new MatrixNetworkAssoc(this).initParams();
//		history.clear();
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
		Filter2D filter1 = defaultFilter(filterStride1);
		return initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2);
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
	public Matrix forward(Record...inputs) {
		Matrix input = inputs != null && inputs.length > 0 ? inputs[0].input() : null;
		Matrix result = evaluate0(input, inputs[0].params.toArray(new Object[] {}));
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
	public Matrix evaluate(Matrix input, Object...params) throws RemoteException {
		return evaluate0(input, params);
	}

	
	@Override
	public Matrix evaluate(Object...params) {
		return evaluate0(null, params);
	}


	/**
	 * Evaluating matrix neural network.
	 * @param input input matrix for evaluating.
	 * @param params other parameters.
	 * @return array as output.
	 */
	public Matrix evaluate0(Matrix input, Object...params) {
//		if (!paramIsHistoryMode()) history.clear();
		
		MatrixLayerAbstract inputLayer = getInputLayer();
		if (input != null) Matrix.copy(input, inputLayer.getInput());
		if (inputLayer.getOutput() != inputLayer.getInput()) inputLayer.setOutput(inputLayer.getInput());
		
		for (int i = 1; i < layers.length; i++) layers[i].evaluate();
		Matrix output = getOutputLayer().queryOutput();
		
		if (params != null && params.length > 0 && params[0] != null && params[0] instanceof Error) {
			((Error)params[0]).addLayerOInput(this);
		}
		
//		if (output == null || !paramIsHistoryMode()) return output;
//		if (history.size() > HISTORY_SIZE) history.clear();
//		history.add(new Inout(getOutputLayer().queryInput(), output));
		return output;
	}
	
	
//	/**
//	 * Getting history input.
//	 * @param output history output.
//	 * @return history input of specified output.
//	 */
//	@Deprecated
//	Matrix historyInputOf(Matrix output) {
//		for (int i = history.size()-1; i >= 0; i--) {
//			if (Matrix.refEquals(history.get(i).output, output))
//				return history.get(i).input;
//		}
//		return null;
//	}
	
	
	@Override
	public Error[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = paramGetMaxIteration();
		double terminatedThreshold = paramGetTerminatedThreshold();
		double learningRate = paramGetLearningRate();
		int epochs = paramGetPseudoEpochs();

		Error[] outputErrors = null;
		Iterable<Record> newsample = sample;
		for (int epoch = 0; epoch < epochs; epoch++) {
			double lr = calcLearningRate(learningRate, epoch+1);
			if (epoch > 0) {
				if (!(newsample instanceof List<?>)) newsample = net.ea.ann.core.Record.listOf(newsample);
				Collections.shuffle((List<?>)newsample);
			}
			outputErrors = learn(newsample, lr, terminatedThreshold, maxIteration);
		}
		return outputErrors;
	}


	/**
	 * Learning matrix neural network.
	 * @param sample sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learning errors.
	 */
	private Error[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Error[] outputErrors = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			if (trainers.size() == 0) {
				List<Error> outputErrorList = Util.newList(0);
				for (Record record : subsample) {
					Matrix input = record.input(), realOutput = record.output();
					Error error = new Error((Matrix)null);
					Matrix output = evaluate0(input, new Object[] {error});
					Matrix err = calcOutputError(output, realOutput, getOutputLayer());
					if (err != null) {
						error.errorSet(err);
						outputErrorList.add(error);
					}
				}
				outputErrors = backward(outputErrorList.toArray(new Error[] {}), this, true, lr);
			}
			else {
				for (TaskTrainer trainer : trainers) {
					outputErrors = trainer.train(this, subsample, false, learningRate);
				}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "mane_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (outputErrors == null || outputErrors.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = Matrix.normMean(Error.errors(outputErrors));
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
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (!validate() || outputErrors == null || outputErrors.length == 0) return null;
		if (focus == null) learning = true;
		
//		outputErrors = Arrays.copyOf(outputErrors, outputErrors.length);
		for (int i = layers.length-1; i >= 0; i--) {
			if ( (!learning) || (!(layers[i] instanceof MatrixLayerImpl)) ) {
				outputErrors = layers[i].backward(outputErrors, layers[i], learning, learningRate);
				continue;
			}
			MatrixLayerImpl layer = (MatrixLayerImpl)layers[i];
			layer.resetBackwardInfo();
			outputErrors = layer.backwardThisLayerWithoutLearning(outputErrors, learningRate);
		}
		for (int i = layers.length-1; i >= 0; i--) {
			if ( (!learning) || (!(layers[i] instanceof MatrixLayerImpl)) ) continue;
			MatrixLayerImpl layer = (MatrixLayerImpl)layers[i];
			layer.updateParametersFromBackwardInfo(outputErrors.length, learningRate);
		}
		
		if (outputErrors == null || this.prevLayer == null || this == focus) return outputErrors;
		
		//Adapting backward errors.
//		Error[] backwardErrors = Arrays.copyOf(outputErrors, outputErrors.length);
		for (int i = 0; i < outputErrors.length; i++) {
			outputErrors[i].errorSet(adaptInputToPrevOutput(outputErrors[i].error(), this.prevLayer));
		}
		if (!(this.prevLayer instanceof MatrixLayerExt)) return this.prevLayer.backward(outputErrors, focus, learning, learningRate);
		
		Error.adjustErrors(this.prevLayer, outputErrors);
		return this.prevLayer.backward(outputErrors, focus, learning, learningRate);
	}

	
//	/**
//	 * Getting history mode.
//	 * @return history mode.
//	 */
//	@Deprecated
//	boolean paramIsHistoryMode() {
//		if (config.containsKey(HISTORY_MODE_FIELD))
//			return config.getAsBoolean(HISTORY_MODE_FIELD);
//		else
//			return HISTORY_MODE_DEFAULT;
//	}
//
//	
//	/**
//	 * Setting history mode.
//	 * @param historyMode history mode.
//	 */
//	@Deprecated
//	MatrixNetworkAbstract paramSetHistoryMode(boolean historyMode) {
//		config.put(HISTORY_MODE_FIELD, historyMode);
//		return this;
//	}


}
