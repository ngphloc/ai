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

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerExt;
import net.ea.ann.mane.MatrixNetworkAssoc;
import net.ea.ann.mane.MatrixNetworkCore;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.TaskTrainer;

/**
 * This class implements basic transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerBasic extends NetworkAbstract implements Transformer, MatrixLayerExt, MatrixNetworkCore {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Number of blocks.
	 */
	public static final int BLOCKS_NUMBER_DEFAULT = 2; //MatrixNetworkImpl.DEPTH_DEFAULT;

	
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
	 * Creating transformer block.
	 * @return transformer block.
	 */
	protected TransformerBlock createBlock() {
		return new TransformerBlock(this.neuronChannel, this.idRef);
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
	 * @param index specified index.
	 * @return block at specified index.
	 */
	public TransformerBlock get(int index) {
		return size() > 0 ? blocks[index] : null;
	}
	
	
	/**
	 * Getting X block index.
	 * @return X block index.
	 */
	public int getXBlockIndex() {
		if (blocks == null) return -1;
		for (int i = 0; i < blocks.length; i++) {
			if (blocks[i].attention.X() != null) return i;
		}
		return -1;
	}
	
	
	/**
	 * Setting transformer as attached object of X block.
	 * @return transformer as attached object of X block.
	 */
	public Transformer getXBlockAttach() {
		int XBlockIndex = getXBlockIndex();
		if (XBlockIndex < 0) return null;
		Object attach = blocks[XBlockIndex].getAttach();
		return attach instanceof Transformer ? (Transformer)attach : null;
	}
	
	
	/**
	 * Setting transformer as attached object of X block.
	 * @param transformer as attached object of X block.
	 */
	protected void setXBlockAttach(Transformer transformer) {
		int XBlockIndex = getXBlockIndex();
		if (XBlockIndex >= 0) blocks[XBlockIndex].setAttach(transformer);
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
	protected void enterInputs(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		if (validate()) get(0).attention.enterInputs(inputY, inputX, inputMask);
	}
	
	
	/**
	 * Setting input data and input mask.
	 * @param input input data.
	 * @param inputMask input mask.
	 */
	protected void enterInputs(Matrix input, boolean[][] inputMask) {
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

	
	/**
	 * Getting output head.
	 * @return output head.
	 */
	public MatrixNetworkImpl getOutputHead() {
		return validate() ? get(size()-1).head : null;
	}

	
	/**
	 * Setting output head.
	 * @param head output head.
	 * @return true if setting is successful.
	 */
	public boolean setOutputHead(MatrixNetworkImpl head) {
		return validate() ? get(size()-1).setHead(head) : false;
	}
	
	
	/**
	 * Setting output head.
	 * @param headOutputSize head output size.
	 * @param headDepth head depth.
	 * @return true if setting is successful.
	 */
	public boolean setOutputHead(Dimension headOutputSize, int headDepth) {
		return validate() ? get(size()-1).setHead(headOutputSize, headDepth) : false;
	}

	
	/**
	 * Removing output head.
	 */
	public void removeOutputHead() {
		if (validate()) get(size()-1).removeHead();
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
	public TransformerBasic setTrainer(TaskTrainer trainer) {
		trainers.clear();
		if (trainer != null) trainers.add(trainer);
		return this;
	}

	
	@Override
	public Matrix forward(Matrix...inputs) {
		Matrix inputY = inputs != null && inputs.length > 0 ? inputs[0] : null;
		Matrix inputX = inputs != null && inputs.length > 1 ? inputs[1] : null;
		Matrix result = evaluate(inputY, inputX, null, new Object[] {});
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
	 * @param inputMatrix mask input. 
	 * @param params additional parameters.
	 * @return matrix as output.
	 */
	protected Matrix evaluate(Matrix inputY, Matrix inputX, boolean[][] inputMatrix, Object...params) {
		if (!validate()) return null;
		Matrix output = blocks[0].evaluate(inputY, null, inputMatrix);
		int XBlockIndex = getXBlockIndex();
		for (int i = 1; i < blocks.length; i++) {
			output = i == XBlockIndex ? blocks[i].evaluate(output, inputX, null) : blocks[i].evaluate(output, null, null);
		}
		return output;
	}
	
	
	/**
	 * Evaluating transformer.
	 * @param input input.
	 * @param inputMatrix mask input. 
	 * @return matrix as output.
	 */
	protected Matrix evaluate(Matrix input, boolean[][] inputMatrix) {
		return evaluate(input, null, inputMatrix, new Object[] {});
	}

	
	@Override
	public Matrix evaluate(Record record) throws RemoteException {
		return evaluate(record.inputY, record.inputX, record.inputMask, new Object[] {});
	}

	
	@Override
	public Matrix evaluate() {
		return evaluate(null, null, null);
	}


	/**
	 * Back-warding transformer block by errors.
	 * @param errors specified errors.
	 * @param learningRate learning rate.
	 * @return learning errors. The first element is main error and the second element is attached error\\.
	 */
	protected Error[][] backward(Error[] errors, double learningRate) {
		if (!validate()) return null;
		Error[] outputErrors = null;
		Error[] attachOutputErrors = null;
		for (int i = blocks.length-1; i >= 0; i--) {
			outputErrors = blocks[i].backward(outputErrors == null ? errors : outputErrors, learningRate);
			
			if (outputErrors == null || outputErrors.length == 0) continue;
			if ((blocks[i].attach == null) || !(blocks[i].attach instanceof TransformerBasic)) continue;
			Error[] attachErrors = Error.extractX(outputErrors);
			if (attachErrors == null || attachErrors.length == 0) continue;
			
			TransformerBasic attach = (TransformerBasic)blocks[i].attach;
			Error[][] merr = attach.backward(attachErrors, learningRate);
			attachOutputErrors = merr != null && merr.length > 0 ? merr[0] : null;
		}
		
		if (outputErrors == null || outputErrors.length == 0)
			return null;
		else if (attachOutputErrors == null || attachOutputErrors.length == 0)
			return new Error[][] {outputErrors};
		else
			return new Error[][] {outputErrors, attachOutputErrors};
	}

	
	@Override
	public Matrix[] backward(Matrix[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (!learning) throw new IllegalArgumentException("Method Transformer::backward(Matrix[], MatrixLayer, boolean, double) does not support learning = false");
		Error[] errors = Error.create(outputErrors);
		Error[][] learnedErrors = backward(errors, learningRate);
		if (learnedErrors == null || learnedErrors.length == 0 || learnedErrors[0] == null) return null;
		
		Matrix[] backwardErrors = Error.create(learnedErrors[0]);
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
	public Matrix[] backward(Matrix[] outputErrors, double learningRate) {
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
	 * @return learning errors. The first element is main error and the second element is attached error\\.
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
				List<Matrix> errorList = Util.newList(0);
				for (Record record : subsample) {
					Matrix A = evaluate(record.inputY, record.inputX, record.inputMask, new Object[] {});
					Matrix error = record.outputA.subtract(A);
					if (error != null) errorList.add(error);
				}
				outputErrors = backward(Error.create(errorList.toArray(new Matrix[] {})), lr);
			}
			else {
				List<net.ea.ann.mane.Record> subinouts = Util.newList(0);
				for (Record record : subsample) {
					net.ea.ann.mane.Record mr = null;
					if (record.inputX == null)
						mr = new net.ea.ann.mane.Record(record.inputY, record.outputA);
					else
						mr = new net.ea.ann.mane.Record(record.inputY, record.outputA, record.inputX, null);
					subinouts.add(mr);
				}
				Matrix[] errors = null;
				for (TaskTrainer trainer : trainers) {
					errors = trainer.train(this, subinouts, false, learningRate);
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

	
	/**
	 * This class represents transformer encoder.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class Encoder extends TransformerBasic {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel and ID reference.
		 * @param neuronChannel neuron channel.
		 * @param idRef ID reference.
		 */
		public Encoder(int neuronChannel, Id idRef) {
			super(neuronChannel, idRef);
		}

		/**
		 * Default constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		public Encoder(int neuronChannel) {
			this(neuronChannel, null);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, int ffnDepth, int nBlocks, int XBlockIndex) {
			return super.initialize(h, n, dm, dk, dv, 0, 0, ffnDepth, nBlocks, -1);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
			return super.initialize(h, n, dm, dk, dv, 0, 0, ffnDepth, nBlocks, -1);
		}
		
	}
	
	
	/**
	 * This class represents transformer decoder.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class Decoder extends TransformerBasic {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel and ID reference.
		 * @param neuronChannel neuron channel.
		 * @param idRef ID reference.
		 */
		public Decoder(int neuronChannel, Id idRef) {
			super(neuronChannel, idRef);
		}

		/**
		 * Default constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		public Decoder(int neuronChannel) {
			this(neuronChannel, null);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int m, int d, int ffnDepth, int nBlocks, int XBlockIndex) {
			return super.initialize(h, n, dm, dk, dv, m, d, ffnDepth, nBlocks, XBlockIndex);
		}

		@Override
		public boolean initialize(int h, int n, int dm, int dk, int dv, int ffnDepth, int nBlocks) {
			return super.initialize(h, n, dm, dk, dv, n, dm, ffnDepth, nBlocks, 1);
		}
		
	}


}



/**
 * This class represents a transformer block.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class TransformerBlock implements MatrixNetworkCore, Cloneable, Serializable {


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
	 * Internal head.
	 */
	protected MatrixNetworkImpl head = null;
	
	
	/**
	 * Internal attention.
	 */
	protected Attention attention = null;
	
	
	/**
	 * Feed forward network.
	 */
	protected MatrixNetworkImpl ffn = null;
	
	
	/**
	 * Attached transformer.
	 */
	protected Transformer attach = null;
	
	
	/**
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef ID reference.
	 */
	TransformerBlock(int neuronChannel, Id idRef) {
		if (idRef != null) this.idRef = idRef;
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
	}

	
	/**
	 * Default constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	TransformerBlock(int neuronChannel) {
		this(neuronChannel, null);
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
	 * Getting head.
	 * @return head.
	 */
	MatrixNetworkImpl getHead() {return head;}
	
	
	/**
	 * Setting head.
	 * @param head head.
	 * @return true if setting is successful.
	 */
	boolean setHead(MatrixNetworkImpl head) {
		if (head == null || !validate()) return false;
		Dimension ffnOutputSize = this.ffn.getOutputLayer().getSize();
		Dimension headInputSize = head.getInputLayer().getSize();
		if (ffnOutputSize.height != headInputSize.height || ffnOutputSize.width != headInputSize.width) return false;
		
		this.head = head;
		return true;
	}

	
	/**
	 * Setting head.
	 * @param headOutputSize head output size.
	 * @param headDepth head depth.
	 * @return true if setting is successful.
	 */
	boolean setHead(Dimension headOutputSize, int headDepth) {
		Dimension ffnOutputSize = this.ffn.getOutputLayer().getSize();
		MatrixNetworkImpl head = new MatrixNetworkImpl(this.neuronChannel, null, null, idRef);
		if (!new MatrixNetworkInitializer(head).initialize(ffnOutputSize, headOutputSize, headDepth)) return false;
		return setHead(head);
	}
	
	
	/**
	 * Removing head.
	 */
	void removeHead() {
		head = null;
	}
	
	
	/**
	 * Getting attached object.
	 * @return attached object.
	 */
	Transformer getAttach() {return attach;}
	
	
	/**
	 * Setting attached object.
	 * @param attach attached object.
	 */
	void setAttach(Transformer attach) {this.attach = attach;}
	
	
	/**
	 * Getting output.
	 * @return output.
	 */
	Matrix getOutput() {
		if (!validate())
			return null;
		else if (head != null)
			return head.getOutput();
		else
			return ffn.getOutput();
	}
	
	
	@Override
	public MatrixLayerAbstract getOutputLayer() {
		if (!validate())
			return null;
		else if (head != null)
			return head.getOutputLayer();
		else
			return ffn.getOutputLayer();
	}


	/**
	 * Validating transformer block.
	 * @return true if transformer block is valid.
	 */
	boolean validate() {
		return attention != null && ffn != null;
	}
	
	
	/**
	 * Resetting transformer.
	 */
	void reset() {
		this.attention = null;
		this.ffn = null;
	}
	
	
	/**
	 * Creating feed-forward network.
	 * @return feed-forward network.
	 */
	protected MatrixNetworkImpl createFFN() {
		return new MatrixNetworkImpl(this.neuronChannel, null, null, idRef);
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
		this.attention = new Attention();
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
	 * Evaluating transformer block.
	 * @param inputY the Y input.
	 * @param inputX the X input.
	 * @param inputMask input mask.
	 * @return array as output.
	 */
	Matrix evaluate(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		if (!validate()) return null;
		Matrix A = attention.evaluate(inputY, inputX, inputMask);
		try {
			A = ffn.evaluate(A);
			if (head != null) return head.evaluate(A);
		} catch (Throwable e) {Util.trace(e);}
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
		Matrix[] headErrors = head != null ? head.backward(Error.create(errors), learningRate) : Error.create(errors); 
		Matrix[] ffnErrors = ffn.backward(headErrors, learningRate);
		if (ffnErrors == null || ffnErrors.length == 0) return null;
		return attention.backward(Error.create(ffnErrors), learningRate);
	}
	
	
	/**
	 * Initializing block parameters.
	 * @param block transformer block.
	 * @param rnd randomizer.
	 */
	static void initParams(TransformerBlock block, Random rnd) {
		if (block.attention != null) Attention.initParams(block.attention, rnd);
		if (block.ffn != null) new MatrixNetworkAssoc(block.ffn).initParams(rnd);
	}


}
