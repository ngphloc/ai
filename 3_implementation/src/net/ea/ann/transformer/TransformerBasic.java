/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.awt.Dimension;
import java.io.Serializable;
import java.rmi.RemoteException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerExt;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.MatrixNetworkAssoc;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.TaskTrainer;
import net.ea.ann.raster.Raster;

/**
 * This class implements basic transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerBasic extends NetworkAbstract implements Transformer, MatrixLayerExt {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Number of blocks.
	 */
	public static final int BLOCKS_NUMBER_DEFAULT = MatrixNetworkImpl.DEPTH_DEFAULT;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Transformer blocks.
	 */
	protected TransformerBlock[] blocks = null;
	
	
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
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef ID reference.
	 */
	public TransformerBasic(int neuronChannel, Id idRef) {
		super(idRef);
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.config.putAll(TransformerBlock.defaultConfig());
	}

	
	/**
	 * Default constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public TransformerBasic(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	public Id getIdRef() {
		return idRef;
	}


	@Override
	public int id() {
		return idRef.get();
	}

	
	/**
	 * Validating transformer block.
	 * @return true if transformer block is valid.
	 */
	public boolean validate() {
		return size() > 0;
	}

	
	/**
	 * Resetting transformer.
	 */
	public void reset() {
		this.blocks = null;
	}
	
	
	/**
	 * Updating configuration.
	 */
	private void updateConfig() {
		if (blocks == null) return;
		for (int i = 0; i < blocks.length; i++) blocks[i].updateConfig(this.config);
	}
	
	
	/**
	 * Updating configuration.
	 * @param config configuration.
	 */
	void updateConfig(NetworkConfig config) {
		this.config.putAll(config);
		updateConfig();
	}
	
	
	/**
	 * Creating transformer block.
	 * @return transformer block.
	 */
	protected TransformerBlock createBlock() {
		TransformerBlock block = new TransformerBlock(this.neuronChannel, idRef);
		block.updateConfig(this.config);
		return block;
	}
	
	
	/**
	 * Initializing transformer with number of heads, sample size, model dimension, key dimension, value dimension, other sample size, other model dimension, depth of feed forward network, number of blocks, and index of X input block.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param m other sample size.
	 * @param d other model dimension. Default other model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link #BLOCKS_NUMBER_DEFAULT}
	 * @param XBlockIndex index of X input block.
	 * @return true if initialization is successful.
	 */
	boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, int ffnDepth, int nBlocks, int XBlockIndex) {
		nBlocks = nBlocks > 0 ? nBlocks : BLOCKS_NUMBER_DEFAULT;
		if (!(m > 0 && d > 0 && XBlockIndex > 0 && XBlockIndex < nBlocks)) XBlockIndex = -1;
		
		this.blocks = new TransformerBlock[nBlocks];
		for (int i = 0; i < nBlocks; i++) {
			this.blocks[i] = createBlock();
			if (i == XBlockIndex) {
				if (!this.blocks[i].initialize(h, n, dm, dk, dv, m, d, ffnDepth)) return false;
			}
			else {
				if (!this.blocks[i].initialize(h, n, dm, dk, dv, ffnDepth)) return false;
			}
		}
		
		initParams(this, new Random());
		return validate();
	}
	
	
	/**
	 * Initializing transformer with number of heads, sample size, model dimension, key dimension, value dimension, depth of feed forward network, and number of blocks.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network. Default depth of feed forward network is {@link MatrixNetworkImpl#DEPTH_DEFAULT}.
	 * @param nBlocks number of blocks. Default number of blocks is {@link #BLOCKS_NUMBER_DEFAULT}
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
		return initialize(h, n, dm, dk, dv, 0, 0, ffnDepth, nBlocks, -1);
	}
	
	
	/**
	 * Getting number of blocks.
	 * @return number of blocks.
	 */
	public int size() {
		return blocks != null ? blocks.length : 0;
	}
	
	
	/**
	 * Getting block at specified index.
	 * @param blockIndex specified index.
	 * @return block at specified index.
	 */
	public TransformerBlock get(int blockIndex) {
		return size() > 0 ? blocks[blockIndex] : null;
	}
	
	
	/**
	 * Getting X block indices.
	 * @return X block indices.
	 */
	public int[] getXIndices() {
		if (!validate()) return null;
		List<Integer> indexList = Util.newList(0);
		for (int i = 0; i < blocks.length; i++) {
			if (blocks[i].attention.X() != null) indexList.add(i);
		}
		if (indexList.size() == 0) return null;
		
		int[] indices = new int[indexList.size()];
		for (int i = 0; i < indices.length; i++) indices[i] = indexList.get(i);
		return indices;
	}
	
	
	/**
	 * Getting input attached transformers.
	 * @return input attached transformers.
	 */
	Transformer[] getInputAttaches() {
		if (!validate()) return null;
		List<Transformer> transformers = Util.newList(0);
		for (int i = 0; i < size(); i++) {
			Transformer transformer = get(i).getInputAttach();
			if (transformer != null) transformers.add(transformer);
		}
		return transformers.size() > 0 ? transformers.toArray(new Transformer[] {}) : null;
	}
	
	
	/**
	 * Setting input attached transformer.
	 * @param inputAttach input attached transformer.
	 */
	boolean setInputAttach(int blockIndex, Transformer inputAttach) {
		return validate() && blockIndex > 0 &&  blockIndex < size() ? get(blockIndex).setInputAttach(inputAttach) : false;
	}
	
	
	@Override
	public MatrixLayer getPrevLayer() {
		return prevLayer;
	}


	@Override
	public MatrixLayer getNextLayer() {
		return nextLayer;
	}

	
	@Override
	public Matrix getInput() {
		return validate() ? get(0).attention.Y() : null;
	}


	/**
	 * Setting Y input data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	public void enterInputs(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		if (validate()) get(0).attention.enterInputs(inputY, inputX, inputMask);
	}
	
	
	/**
	 * Setting input data and input mask.
	 * @param input input data.
	 * @param inputMask input mask.
	 */
	public void enterInputs(Matrix input, boolean[][] inputMask) {
		enterInputs(input, null, inputMask);
	}

	
	@Override
	public void enterInputs(net.ea.ann.mane.Record record) {
		Matrix inputY = record.input();
		Matrix inputX = record.input2();
		Object extraInput = record.extraInput();
		boolean[][] inputMask = (extraInput != null) && (extraInput instanceof boolean[][]) ? (boolean[][])extraInput : null;
		enterInputs(inputY, inputX, inputMask);
	}

	
	@Override
	public Matrix getOutput() {
		return validate() ? get(size()-1).getOutput() : null;
	}


	@Override
	public MatrixLayerAbstract getOutputLayer() {
		return validate() ? get(size()-1).getOutputLayer() : null;
	}

	
//	@Override
//	public Function getOutputActivateRef() {
//		MatrixLayerAbstract outputLayer = getOutputLayer();
//		return outputLayer != null ? outputLayer.getActivateRef() : null;
//	}


	/**
	 * Getting output adapter.
	 * @return output adapter.
	 */
	public MatrixNetworkImpl getOutputAdapter() {
		return validate() ? get(size()-1).getOutputAdapter() : null;
	}

	
	/**
	 * Setting output adapter.
	 * @param outputAdapter output adapter.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(MatrixNetworkImpl outputAdapter) {
		return validate() ? get(size()-1).setOutputAdapter(outputAdapter) : false;
	}
	
	
	/**
	 * Setting output adapter.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(Dimension outputSize, int outputDepth) {
		return validate() ? get(size()-1).setOutputAdapter(outputSize, outputDepth) : false;
	}

	
	/**
	 * Setting output adapter.
	 * @param middleSize middle size.
	 * @param middleFilter middle filter.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(Dimension middleSize, Filter2D middleFilter, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		return validate() ? get(size()-1).setOutputAdapter(middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth) : false;
	}
	

	/**
	 * Setting output adapter.
	 * @param middleSize middle size.
	 * @param middleFilterStride middle filter stride.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputAdapter(Dimension middleSize, Dimension middleFilterStride, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		return validate() ? get(size()-1).setOutputAdapter(middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth) : false;
	}

		
	/**
	 * Removing output adapter.
	 */
	public TransformerBasic removeOutputAdapter() {
		if (validate()) get(size()-1).removeOutputAdapter();
		return this;
	}
	
	
	/**
	 * Getting feed-forward network.
	 * @return feed-forward network.
	 */
	MatrixNetworkImpl getOutputFFN() {
		return validate() ? get(size()-1).ffn() : null;
	}
	
	
	/**
	 * Setting feed-forward network.
	 * @param ffn feed-forward network.
	 * @return true if setting is successful.
	 */
	boolean setOutputFFN(MatrixNetworkImpl ffn) {
		return validate() ? get(size()-1).setFFN(ffn) : false;
	}
	
	
	/**
	 * Setting feed-forward network.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputFFN(Dimension outputSize, int outputDepth) {
		return validate() ? get(size()-1).setFFN(outputSize, outputDepth) : false;
	}

	
	/**
	 * Setting feed-forward network.
	 * @param middleSize middle size.
	 * @param middleFilter middle filter.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputFFN(Dimension middleSize, Filter2D middleFilter, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		return validate() ? get(size()-1).setFFN(middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth) : false;
	}
	

	/**
	 * Setting feed-forward network.
	 * @param middleSize middle size.
	 * @param middleFilterStride middle filter stride.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputFFN(Dimension middleSize, Dimension middleFilterStride, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		return validate() ? get(size()-1).setFFN(middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth) : false;
	}

		
	/**
	 * Getting size of trainers.
	 * @return size of trainers.
	 */
	int getTrainerSize() {return trainers.size();}
	
	
	/**
	 * Getting trainer at specified index.
	 * @param index specified index.
	 * @return trainer at specified index.
	 */
	TaskTrainer getTrainer(int index) {return trainers.get(index);}
	
	
	/**
	 * Getting trainer.
	 * @return the first trainer.
	 */
	public TaskTrainer getTrainer() {return trainers.size() > 0 ? trainers.get(0) : null;}
	
	
	/**
	 * Adding trainer.
	 * @param trainer specified trainer.
	 * @return adding is successful.
	 */
	boolean addTrainer(TaskTrainer trainer) {return trainers.add(trainer);}
	
	
	/**
	 * Removing trainer.
	 * @param trainer specified trainer.
	 * @return removal is successful.
	 */
	boolean removeTrainer(TaskTrainer trainer) {return trainers.remove(trainer);}
	
	
	/**
	 * Clearing trainer.
	 */
	void clearTrainers() {trainers.clear();}
	
	
	/**
	 * Setting trainer.
	 * @param trainer specified trainer.
	 * @return this network.
	 */
	public TransformerBasic setTrainer(TaskTrainer trainer) {
		trainers.clear();
		if (trainer != null) trainers.add(trainer);
		return this;
	}

	
	@Override
	public Matrix forward(net.ea.ann.mane.Record...inputs) {
		Matrix inputY = inputs != null && inputs.length > 0 ? inputs[0].input() : null;
		Matrix inputX = inputs != null && inputs.length > 0 ? inputs[0].input2() : null;
		Object extraInput = inputs != null && inputs.length > 0 ? inputs[0].extraInput() : null;
		boolean[][] inputMask = (extraInput != null) && (extraInput instanceof boolean[][]) ? (boolean[][])extraInput : null;
		Matrix result = evaluate(inputY, inputX, inputMask, new Object[] {});
		if (result == null) return result;
		
		MatrixLayer nextLayer = null;
		while ((nextLayer = this.getNextLayer()) != null) {
			Matrix.copy(result, nextLayer.getInput());
			result = nextLayer.evaluate();
		}
		return result;
	}


	/**
	 * Evaluating transformer.
	 * @param inputY Y input.
	 * @param inputX X input.
	 * @param inputMask mask input. 
	 * @param params additional parameters.
	 * @return matrix as output.
	 */
	protected Matrix evaluate(Matrix inputY, Matrix inputX, boolean[][] inputMask, Object...params) {
		if (!validate()) return null;
		updateConfig();
			
		Matrix output = blocks[0].evaluate(inputY, null, inputMask, params);
		int vecRows = blocks[0].getVecRows();
		for (int i = 1; i < blocks.length; i++) {
			Matrix Y = vecRows > 0 ? output.vecInverse(vecRows) : output;
			if (blocks[i].containsX())
				output = blocks[i].evaluate(Y, inputX, null, params);
			else
				output = blocks[i].evaluate(Y, null, null, params);
		}
		
		if (params != null && params.length > 0 && params[0] != null && params[0] instanceof Error) {
			((Error)params[0]).addLayerOInput(this);
		}
		return output;
	}
	
	
	/**
	 * Evaluating transformer.
	 * @param input input.
	 * @param inputMask mask input.
	 * @param params additional parameters.
	 * @return matrix as output.
	 */
	public Matrix evaluate(Matrix input, boolean[][] inputMask, Object...params) {
		return evaluate(input, null, inputMask, params);
	}

	
	@Override
	public Matrix evaluate(Record record) throws RemoteException {
		return evaluate(record.inputY(), record.inputX(), record.inputMask(), record.params.toArray(new Object[] {}));
	}

	
	@Override
	public Matrix evaluate(Object...params) {
		return evaluate(null, null, params);
	}


	/**
	 * Back-warding transformer block by errors.
	 * @param errors specified errors.
	 * @param learningRate learning rate.
	 * @return learning errors. The first element is main error and the second element is attached error\\.
	 */
	protected Error[][] backward(Error[] errors, double learningRate) {
		if (!validate()) return null;
		updateConfig();

		Error[] outputErrors = null;
		List<Error[]> attachOutputErrorsList = Util.newList(0);
		for (int i = blocks.length-1; i >= 0; i--) {
			outputErrors = blocks[i].backward(outputErrors == null ? errors : outputErrors, learningRate);
			if (i > 0) Error.adjustErrors(blocks[i-1], outputErrors);
			
			if (outputErrors == null || outputErrors.length == 0) continue;
			if ((blocks[i].inputAttach == null) || !(blocks[i].inputAttach instanceof TransformerBasic)) continue;
			Error[] attachErrors = Error.createByAttach(outputErrors);
			if (attachErrors == null || attachErrors.length == 0) continue;
			
			TransformerBasic attach = (TransformerBasic)blocks[i].inputAttach;
			Error.adjustErrors(attach, attachErrors);
			Error[][] merr = attach.backward(attachErrors, learningRate);
			if (merr != null && merr.length > 0 && merr[0] != null) {
//				if (merr[0].length > 0 && merr[0][0] != null) merr[0][0].tag = attach;
				attachOutputErrorsList.add(merr[0]);
			}
			else {
				System.out.println("Error in training attached transformer at block " + i + " inside method TransformerBasic#backward(Error[], double).");
			}
		}
		
		if (outputErrors == null || outputErrors.length == 0)
			return null;
		else if (attachOutputErrorsList.size() == 0)
			return new Error[][] {outputErrors};
		else {
			Error[][] backwardErrors = new Error[attachOutputErrorsList.size() + 1][];
			backwardErrors[0] = outputErrors;
			for (int i = 1; i < backwardErrors.length; i++) backwardErrors[i] = attachOutputErrorsList.get(i-1);
			return backwardErrors;
		}
	}

	
	@Override
	public net.ea.ann.mane.Error[] backward(net.ea.ann.mane.Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (!learning) throw new IllegalArgumentException("Method Transformer::backward(Matrix[], MatrixLayer, boolean, double) does not support learning = false");
		Error[] errors = Error.create(outputErrors);
		Error[][] learnedErrors = backward(errors, learningRate);
		if (learnedErrors == null || learnedErrors.length == 0 || learnedErrors[0] == null) return null;
		
		net.ea.ann.mane.Error[] backwardErrors = Error.extract(learnedErrors[0]);
		if (this.prevLayer == null || this == focus)
			return backwardErrors;
		else
			return this.prevLayer.backward(backwardErrors, focus, learning, learningRate);
	}


	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	public net.ea.ann.mane.Error[] backward(net.ea.ann.mane.Error[] outputErrors, double learningRate) {
		return backward(outputErrors, null, true, learningRate);
	}
	
	
	@Override
	public Error[][] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = paramGetMaxIteration();
		double terminatedThreshold = paramGetTerminatedThreshold();
		double learningRate = paramGetLearningRate();
		int epochs = paramGetPseudoEpochs();

		Error[][] outputErrors = null;
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
	 * Learning transformer.
	 * @param sample sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learning errors. The first element is main error and the second element is attached error, etc.
	 */
	protected Error[][] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		if (!validate()) return null;
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Error[][] outputErrors = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			if (trainers.size() == 0) {
				List<Error> errorList = Util.newList(0);
				for (Record record : subsample) {
					Error error = new Error((Matrix)null);
					Matrix A = evaluate(record.inputY(), record.inputX(), record.inputMask(), error);
					Matrix err = record.outputA().subtract(A);
					if (err != null) {
						error.errorSet(err);
						errorList.add(error);
					}
				}
				outputErrors = backward(errorList.toArray(new Error[] {}), lr);
			}
			else {
				List<net.ea.ann.mane.Record> maneSample = Record.convert(subsample);
				net.ea.ann.mane.Error[] errors = null;
				for (TaskTrainer trainer : trainers) {
					errors = trainer.train(this, maneSample, false, learningRate);
				}
				outputErrors = new Error[][] {Error.create(errors)};
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "transformer_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (outputErrors == null || outputErrors.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = Matrix.normMean(Error.create(outputErrors[0]));
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "transformer_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return outputErrors;
	}


//	/**
//	 * Checking normalization mode.
//	 * @return normalization mode in rang [0, 1].
//	 */
//	boolean paramIsNorm() {
//		if (config.containsKey(Raster.NORM_FIELD))
//			return config.getAsBoolean(Raster.NORM_FIELD);
//		else
//			return Raster.NORM_DEFAULT;
//	}
//
//
//	/**
//	 * Setting normalization mode.
//	 * @param isNorm normalization mode in rang [0, 1]..
//	 * @return this transformer.
//	 */
//	TransformerBasic paramSetNorm(boolean isNorm) {
//		config.put(Raster.NORM_FIELD, isNorm);
//		return this;
//	}
//
//	
//	/**
//	 * Checking vectorization mode.
//	 * @return vectorization mode.
//	 */
//	boolean paramIsVectorized() {
//		if (config.containsKey(MatrixNetworkAbstract.VECTORIZED_FIELD))
//			return config.getAsBoolean(MatrixNetworkAbstract.VECTORIZED_FIELD);
//		else
//			return MatrixNetworkAbstract.VECTORIZED_DEFAULT;
//	}
//
//
//	/**
//	 * Setting vectorization mode.
//	 * @param vectorized vectorization mode.
//	 * @return this network.
//	 */
//	TransformerBasic paramSetVectorized(boolean vectorized) {
//		config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, vectorized);
//		return this;
//	}

	
	/**
	 * Initializing transformer parameters.
	 * @param block transformer block.
	 * @param rnd randomizer.
	 */
	static void initParams(TransformerBasic transformer, Random rnd) {
		if (transformer.blocks != null) {
			for (TransformerBlock block : transformer.blocks) {
				TransformerBlock.initParams(block, rnd);
			}
		}
	}

	
}



/**
 * This class represents default transformer block.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class TransformerBlock implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
    /**
	 * Internal identifier.
	 */
	protected Id idRef = new Id();

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Internal attention.
	 */
	protected Attention attention = null;
	
	
	/**
	 * Feed forward network.
	 */
	protected MatrixNetworkImpl ffn = null;
	
	
	/**
	 * Output adapter.
	 */
	protected MatrixNetworkImpl outputAdapter = null;
	
	
	/**
	 * Input attached transformer.
	 */
	protected Transformer inputAttach = null;
	
	
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();
	
	
	/**
	 * Constructor with neuron channel, vectorization flag, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef ID reference.
	 */
	TransformerBlock(int neuronChannel, Id idRef) {
		if (idRef != null) this.idRef = idRef;
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.config.putAll(defaultConfig());
	}

	
	/**
	 * Default constructor with neuron channel and vectorization flag.
	 * @param neuronChannel neuron channel.
	 */
	TransformerBlock(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	/**
	 * Creating attention.
	 * @return attention.
	 */
	protected Attention createAttention() {return new Attention();}
	
	
	/**
	 * Creating feed-forward network.
	 * @return feed-forward network.
	 */
	protected MatrixNetworkImpl createFFN() {
		MatrixNetworkImpl ffn = new MatrixNetworkImpl(this.neuronChannel, null, null, idRef);
		try {
			ffn.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		return ffn;
	}
	
	
	/**
	 * Creating output adapter.
	 * @return output adapter.
	 */
	MatrixNetworkImpl createOutputAdapter() {
		MatrixNetworkImpl outputAdapter = new MatrixNetworkImpl(this.neuronChannel, null, null, idRef);
		try {
			outputAdapter.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		return outputAdapter;
	}

	
	/**
	 * Creating default configuration.
	 * @return default configuration.
	 */
	static NetworkConfig defaultConfig() {
		NetworkConfig config = new NetworkConfig();
		config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, MatrixNetworkAbstract.VECTORIZED_DEFAULT);
		return config;
	}
	
	
	/**
	 * Updating configuration.
	 */
	private void updateConfig() {
		try {
			if (ffn != null) ffn.getConfig().putAll(this.config);
			if (outputAdapter != null) outputAdapter.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
	}
	
	
	/**
	 * Updating configuration.
	 * @param config configuration.
	 */
	void updateConfig(NetworkConfig config) {
		try {
			this.config.putAll(config);
		} catch (Throwable e) {Util.trace(e);}
		updateConfig();
	}

	
	/**
	 * Initialize matrix neural network.
	 * @param middleSize middle size.
	 * @param middleFilter middle filter.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	private boolean initializeMANE(MatrixNetworkImpl mane, Dimension inputSize, Dimension middleSize, Filter2D middleFilter, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		if ((inputSize == null) || (middleSize == null && finalSize == null)) return false;
		try {
			mane.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		return mane.paramGetMiddleSize() <= 0 ?
			mane.initializeByDepth(inputSize, middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth) :
			mane.initialize(inputSize, middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth);
	}
	
	
	/**
	 * Initialize matrix neural network.
	 * @param middleSize middle size.
	 * @param middleFilterStride middle filter stride.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	private boolean initializeMANE(MatrixNetworkImpl mane, Dimension inputSize, Dimension middleSize, Dimension middleFilterStride, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		if ((inputSize == null) || (middleSize == null && finalSize == null)) return false;
		try {
			mane.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		return mane.paramGetMiddleSize() <= 0 ?
			mane.initializeByDepth(inputSize, middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth) :
			mane.initialize(inputSize, middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth);
	}

	
	/**
	 * Initializing attention with number of heads, sample size, model dimension, key dimension, value dimension, other sample size, other model dimension, and depth of feed forward network.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param m other sample size.
	 * @param d other model dimension. Default other model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network.
	 * @return true if initialization is successful.
	 */
	boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, int ffnDepth) {
		this.ffn = createFFN();
		NeuronValue zero = this.ffn.newNeuronValue().zero();
		this.attention = createAttention();
		if (!this.attention.initialize(h, n, dm, dk, dv, m, d, zero)) return false;
		
		ffnDepth = ffnDepth < 1 ? MatrixNetworkImpl.DEPTH_DEFAULT : ffnDepth;
		Dimension size = new Dimension(this.attention.dm(), this.attention.n());
		if (!new MatrixNetworkInitializer(this.ffn).initialize(size, size, ffnDepth)) return false;
		
		return true;
	}
	
	
	/**
	 * Initializing attention with number of heads, sample size, model dimension, key dimension, value dimension, and depth of feed forward network.
	 * @param h number of heads. Default number of heads is {@link Attention0#HEADS_NUMBER_DEFAULT}.
	 * @param n sample size.
	 * @param dm model dimension. Default model dimension is {@link Attention0#MODEL_DIMENSION_DEFAULT}.
	 * @param dk key dimension. Default key dimension is {@link Attention0#KEY_DIMENSION_DEFAULT}.
	 * @param dv value dimension. Default value dimension is {@link Attention0#VALUE_DIMENSION_DEFAULT}.
	 * @param ffnDepth depth of feed forward network.
	 * @return true if initialization is successful.
	 */
	boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth) {
		return initialize(h, n, dm, dk, dv, 0, 0, ffnDepth);
	}
	
	
	/**
	 * Checking whether this block has X input.
	 * @return whether this block has X input.
	 */
	boolean containsX() {
		return validate() ? attention.X() != null : false;
	}

	
	/**
	 * Getting attention.
	 * @return attention.
	 */
	public Attention attention() {
		return this.attention;
	}
	
	
	/**
	 * Getting feed-forward network.
	 * @return feed-forward network.
	 */
	public MatrixNetworkImpl ffn() {
		return this.ffn;
	}
	
	
	/**
	 * Setting feed-forward network.
	 * @param ffn feed-forward network.
	 * @return true if setting is true.
	 */
	boolean setFFN(MatrixNetworkImpl ffn) {
		if (!validate() || ffn == null || ffn.paramIsNorm() != this.paramIsNorm() || ffn.paramIsVectorized() != this.paramIsVectorized()) return false;
		Dimension ffnSize = ffn.getInputLayer().getSizeByVecRows();
		if (this.attention.n() != ffnSize.height || this.attention.dm() != ffnSize.width) return false;
		if (this.outputAdapter == null) {
			this.ffn = ffn;
			return true;
		}
		
		Dimension ffnOutputSize = ffn.getOutputLayer().getSize();
		Dimension adapterInputSize = this.outputAdapter.getInputLayer().getSize();
		if (ffnOutputSize.height != adapterInputSize.height || ffnOutputSize.width != adapterInputSize.width) return false;
		this.ffn = ffn;
		return true;
	}
	
	
	/**
	 * Setting feed-forward network.
	 * @param inputSize input size.
	 * @param ffnOutputSize output size of feed-forward network.
	 * @param ffnDepth depth of feed-forward network.
	 * @return true if initialization is successful.
	 */
	public boolean setFFN(Dimension ffnOutputSize, int ffnDepth) {
		return setFFN(null, (Filter2D)null, 0, false, ffnOutputSize, ffnDepth);
	}

	
	/**
	 * Setting feed-forward network.
	 * @param middleSize middle size.
	 * @param middleFilter middle filter.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	boolean setFFN(Dimension middleSize, Filter2D middleFilter, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		MatrixNetworkImpl ffn = createFFN();
		Dimension attentionSize = new Dimension(this.attention.dm(), this.attention.n());
		if (!initializeMANE(ffn, attentionSize, middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth)) return false;
		return setFFN(ffn);
	}

	
	/**
	 * Setting feed-forward network.
	 * @param middleSize middle size.
	 * @param middleFilterStride middle filter stride.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	boolean setFFN(Dimension middleSize, Dimension middleFilterStride, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		MatrixNetworkImpl ffn = createFFN();
		Dimension attentionSize = new Dimension(this.attention.dm(), this.attention.n());
		if (!initializeMANE(ffn, attentionSize, middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth)) return false;
		return setFFN(ffn);
	}

	
	/**
	 * Removing feed-forward network.
	 */
	public void removeFFN() {
		ffn = null;
		outputAdapter = null;
	}
	
	
	/**
	 * Getting output adapter.
	 * @return output adapter.
	 */
	MatrixNetworkImpl getOutputAdapter() {return outputAdapter;}
	
	
	/**
	 * Setting output adapter.
	 * @param outputAdapter output adapter.
	 * @return true if setting is successful.
	 */
	boolean setOutputAdapter(MatrixNetworkImpl outputAdapter) {
		if (!validate() || outputAdapter == null || outputAdapter.paramIsNorm() != this.paramIsNorm() || outputAdapter.paramIsVectorized() != this.paramIsVectorized()) return false;
		Dimension ffnOutputSize = this.ffn.getOutputLayer().getSize();
		Dimension adapterInputSize = outputAdapter.getInputLayer().getSize();
		if (ffnOutputSize.height != adapterInputSize.height || ffnOutputSize.width != adapterInputSize.width) return false;
		
		this.outputAdapter = outputAdapter;
		return true;
	}

	
	/**
	 * Setting output adapter.
	 * @param outputSize output size.
	 * @param outputDepth output depth.
	 * @return true if setting is successful.
	 */
	boolean setOutputAdapter(Dimension outputSize, int outputDepth) {
		return setOutputAdapter(null, (Filter2D)null, 0, false, outputSize, outputDepth);
	}
	
	
	/**
	 * Setting output adapter.
	 * @param middleSize middle size.
	 * @param middleFilter middle filter.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	boolean setOutputAdapter(Dimension middleSize, Filter2D middleFilter, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		Dimension ffnOutputSize = this.ffn.getOutputLayer().getSize();
		MatrixNetworkImpl outputAdapter = createOutputAdapter();
		if (!initializeMANE(outputAdapter, ffnOutputSize, middleSize, middleFilter, middleDepth, middleDual, finalSize, finalDepth)) return false;
		return setOutputAdapter(outputAdapter);
	}

	
	/**
	 * Setting output adapter.
	 * @param middleSize middle size.
	 * @param middleFilterStride middle filter stride.
	 * @param middleDepth middle depth.
	 * @param middleDual middle dual mode.
	 * @param finalSize final size.
	 * @param finalDepth final depth.
	 * @return true if setting is successful.
	 */
	boolean setOutputAdapter(Dimension middleSize, Dimension middleFilterStride, int middleDepth, boolean middleDual, Dimension finalSize, int finalDepth) {
		Dimension ffnOutputSize = this.ffn.getOutputLayer().getSize();
		MatrixNetworkImpl outputAdapter = new MatrixNetworkImpl(this.neuronChannel, null, null, idRef);
		if (!initializeMANE(outputAdapter, ffnOutputSize, middleSize, middleFilterStride, middleDepth, middleDual, finalSize, finalDepth)) return false;
		return setOutputAdapter(outputAdapter);
	}

	
	/**
	 * Removing output adapter.
	 */
	void removeOutputAdapter() {outputAdapter = null;}
	

	/**
	 * Getting input attached transformer.
	 * @return input attached transformer.
	 */
	public Transformer getInputAttach() {return inputAttach;}
	
	
	/**
	 * Setting input attached transformer.
	 * @param inputAttach attached transformer.
	 * @return true if setting is successful.
	 */
	boolean setInputAttach(Transformer inputAttach) {
		if (inputAttach == null || this.attention.X() == null) return false;
		this.inputAttach = inputAttach;
		return true;
	}
	
	
	/**
	 * Removing input attached transformer.
	 */
	void removeInputAttach() {this.inputAttach = null;}
	
	
	/**
	 * Getting output.
	 * @return output.
	 */
	Matrix getOutput() {
		if (!validate())
			return null;
		else if (outputAdapter != null)
			return outputAdapter.getOutput();
		else if (ffn != null)
			return ffn.getOutput();
		else
			return attention.A();
	}
	
	
	/**
	 * Getting output layer.
	 * @return output layer.
	 */
	MatrixLayerAbstract getOutputLayer() {
		if (!validate())
			return null;
		else if (outputAdapter != null)
			return outputAdapter.getOutputLayer();
		else if (ffn != null)
			return ffn.getOutputLayer();
		else
			return null;
	}


	/**
	 * Validating transformer block.
	 * @return true if transformer block is valid.
	 */
	boolean validate() {
		return attention != null && attention.validate() /*&& ffn != null*/;
	}
	
	
	/**
	 * Resetting transformer.
	 */
	void reset() {
		this.attention = null;
		this.ffn = null;
		this.outputAdapter = null;
		this.inputAttach = null;
	}
	
	
	/**
	 * Getting vectorization rows.
	 * @return vectorization rows.
	 */
	int getVecRows() {
		if (!validate())
			return 0;
		else if ((paramIsVectorized()) && (ffn != null || outputAdapter != null))
			return ffn != null ? ffn.getInputLayer().getVecRows() : (outputAdapter != null ? outputAdapter.getInputLayer().getVecRows() : 0);
		else
			return 0;
	}
	
	
	/**
	 * Evaluating transformer block.
	 * @param inputY the Y input.
	 * @param inputX the X input.
	 * @param inputMask input mask.
	 * @param params additional parameters.
	 * @return array as output.
	 */
	Matrix evaluate(Matrix inputY, Matrix inputX, boolean[][] inputMask, Object...params) {
		if (!validate()) return null;
		updateConfig();
		
		Matrix A = attention.evaluate(inputY, inputX, inputMask, params);
		if (ffn == null && outputAdapter == null) return A;
		
		int vecRows = getVecRows();
		if (vecRows > 0) A = A.vec();
		if (ffn != null) A = ffn.evaluate0(A, params);
		if (outputAdapter != null) A = outputAdapter.evaluate0(A, params);
		return A;
	}

	
	/**
	 * Back-warding transformer block by errors.
	 * @param errors specified errors.
	 * @param learningRate learning rate.
	 * @return learning errors. The first element is Y error matrix and the second element is X error matrix.
	 */
	Error[] backward(Error[] errors, double learningRate) {
		if (!validate() || errors == null || errors.length == 0) return null;
		updateConfig();
		
		//Back-warding output adapter. 
		net.ea.ann.mane.Error[] outputAdapterErrors = Error.extract(errors);
		if (outputAdapter != null) {
			outputAdapterErrors = outputAdapter.backward(outputAdapterErrors, learningRate);
			if (outputAdapterErrors == null || outputAdapterErrors.length == 0) return null;
			if (ffn != null) net.ea.ann.mane.Error.adjustErrors(ffn, outputAdapterErrors);
		}
		
		//Back-warding feed-forward network.
		net.ea.ann.mane.Error[] ffnErrors = ffn != null ? ffn.backward(outputAdapterErrors, learningRate) : outputAdapterErrors;
		if (ffnErrors == null || ffnErrors.length == 0) return null;
		
		//Back-warding attention.
		int vecRows = getVecRows();
		if (vecRows > 0) {
			for (net.ea.ann.mane.Error ffnError : ffnErrors) {
				ffnError.errorSet(ffnError.error().vecInverse(vecRows));
			}
		}
		return attention.backward(Error.create(ffnErrors), learningRate);
	}
	
	
	/**
	 * Checking normalization mode.
	 * @return normalization mode in rang [0, 1].
	 */
	boolean paramIsNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Setting normalization mode.
	 * @param isNorm normalization mode in rang [0, 1]..
	 * @return this transformer block.
	 */
	TransformerBlock paramSetNorm(boolean isNorm) {
		config.put(Raster.NORM_FIELD, isNorm);
		return this;
	}

	
	/**
	 * Checking vectorization mode.
	 * @return vectorization mode.
	 */
	boolean paramIsVectorized() {
		if (config.containsKey(MatrixNetworkAbstract.VECTORIZED_FIELD))
			return config.getAsBoolean(MatrixNetworkAbstract.VECTORIZED_FIELD);
		else
			return MatrixNetworkAbstract.VECTORIZED_DEFAULT;
	}


	/**
	 * Setting vectorization mode.
	 * @param vectorized vectorization mode.
	 * @return this network.
	 */
	TransformerBlock paramSetVectorized(boolean vectorized) {
		config.put(MatrixNetworkAbstract.VECTORIZED_FIELD, vectorized);
		return this;
	}
	
	
	/**
	 * Initializing block parameters.
	 * @param block transformer block.
	 * @param rnd randomizer.
	 */
	static void initParams(TransformerBlock block, Random rnd) {
		if (block.attention != null) Attention.initParams(block.attention, rnd);
		if (block.ffn != null) new MatrixNetworkAssoc(block.ffn).initParams(rnd);
		if (block.outputAdapter != null) new MatrixNetworkAssoc(block.outputAdapter).initParams(rnd);
	}


}
