/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.awt.Dimension;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Softmax;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.raster.Raster;
import net.ea.ann.transformer.TransformerImpl;
import net.ea.ann.transformer.TransformerInitializer;

/**
 * This class is default implementation of classifier within context of transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerClassifier extends TransformerClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier nut.
	 */
	protected MatrixNetworkImpl adjuster = null;
	
	
	/**
	 * Adjusting baseline.
	 */
	protected Matrix adjustBaseline = null;

	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	public TransformerClassifier(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public TransformerClassifier(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	public void reset() {
		super.reset();
		adjuster = null;
		adjustBaseline = null;
	}


	@Override
	protected boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension nCoreClasses2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2)) return false;
		if (!paramIsAdjust() || !paramIsBaseline()) return true;

		MatrixNetworkImpl nut = transformer.getOutputAdapter();
		nut = nut != null ? nut : transformer.getOutputFFN();
		int minAdjustDepth = Math.max((int)(Math.log(nut.size())/Math.log(MatrixNetworkImpl.BASE_DEFAULT)), 1);
		int maxAdjustDepth = Math.max(Math.max(nut.size()-1, 1), minAdjustDepth);
		int adjustDepth = Math.max(Math.max(depth1, depth2), minAdjustDepth);
		adjustDepth = Math.min(adjustDepth, maxAdjustDepth);
		Dimension size = nut.getOutputLayer().getSize();
		this.adjustBaseline = null;
		this.adjuster = new MatrixNetworkImpl(this.neuronChannel, nut.getActivateRef(), nut.getConvActivateRef(), this.idRef);
		this.adjuster.paramSetInclude(this);
		if (paramIsEntropyTrainer()) this.adjuster.setTrainer(new TaskTrainerLossEntropy());
		return new MatrixNetworkInitializer(adjuster).initialize(size, size, adjustDepth);
	}


	@Override
	double[] weightsOfOutput(Matrix output, int groupIndex) {
		if (adjuster == null) return super.weightsOfOutput(output, groupIndex);
		NeuronValue[] values = getOutput(output, groupIndex);
		values = paramIsEntropyTrainer() ? Softmax.softmax(values) : values;
		if (this.baseline == null || this.adjustBaseline == null) return super.weightsOfOutput(output, groupIndex);
		
		for (int classIndex = 0; classIndex < values.length; classIndex++) {
			NeuronValue base = paramIsByColumn() ? this.baseline.get(classIndex, groupIndex) : this.baseline.get(groupIndex, classIndex);
			NeuronValue adjustBase = paramIsByColumn() ? this.adjustBaseline.get(classIndex, groupIndex) : this.adjustBaseline.get(groupIndex, classIndex);
			//Following code lines are important due to apply baseline into determining class.
			NeuronValue sim = values[classIndex].subtract(base);
			sim = sim.multiply(adjustBase);
			values[classIndex] = sim;
		}
		return weightsOfOutput(values);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		List<Record> newsample = prelearn(sample);
		Error[] errors = learn(newsample);
		learnVerify(newsample);
		
		if (this.adjuster != null) {
			this.adjuster.paramSetInclude(this);
			List<Record> adjustSample = Util.newList(0);
			for (Record record : newsample) {
				Matrix output = evaluate(record.input());
				if (output != null) adjustSample.add(new Record(output, record.output()));
			}
			errors = adjustSample.size() > 0 ? this.adjuster.learn(adjustSample) : errors;
		}
		
		NeuronValue[] errorArray = null;
		for (Error error : errors) {
			NeuronValue[] values = Matrix.extractValues(error.error());
			errorArray = errorArray == null ? values : NeuronValue.concatArray(errorArray, values);
		}
		return errorArray;
	}

	
	@Override
	public void learnVerify(Iterable<Record> sample) {
		if (adjuster == null) {
			super.learnVerify(sample);
			return;
		}
		this.baseline = null;
		this.adjustBaseline = null;
		if (!paramIsBaseline()) return;
		
		this.baseline = calcBaseline(sample);

//		List<Matrix> outputList = Util.newList(0);
//		for (Record inout : inouts) {
//			Matrix output = evaluate(inout.input());
//			if (output != null) outputList.add(output);
//		}
//		if (outputList.size() == 0) return;
//		this.baseline = calcBaseline(outputList.toArray(new Matrix[] {}));
//		
//		List<Matrix> adjustOutputList = Util.newList(0);
//		for (Matrix output : outputList) {
//			try {
//				Matrix adjustOutput = adjuster.evaluate(output);
//				if (adjustOutput != null) adjustOutputList.add(adjustOutput);
//			} catch (Throwable e) {Util.trace(e);}
//		}
//		if (adjustOutputList.size() == 0) return;
//		this.adjustBaseline = calcBaseline(adjustOutputList.toArray(new Matrix[] {}));
	}

	
	/**
	 * Creating matrix neural classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static TransformerClassifier create(int neuronChannel, boolean isNorm) {
		TransformerClassifier tramac = new TransformerClassifier(neuronChannel);
		tramac.paramSetNorm(isNorm);
		return tramac;
	}


}



/**
 * This class is abstract implementation of classifier within context of transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class TransformerClassifierAbstract extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for number of blocks.
	 */
	public static final String BLOCKS_NUMBER_FIELD = "tramac_blocks";
	
	
	/**
	 * Field for number of blocks.
	 */
	public static final int BLOCKS_NUMBER_DEFAULT = 1; //TransformerBasic.BLOCKS_NUMBER_DEFAULT;

	
	/**
	 * Internal transformer.
	 */
	protected TransformerImpl transformer = null;
	
	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	public TransformerClassifierAbstract(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
		this.config.put(BLOCKS_NUMBER_FIELD, BLOCKS_NUMBER_DEFAULT);
		
		this.transformer = new TransformerImpl(this.neuronChannel, idRef);
		try {
			this.config.putAll(this.transformer.getConfig());
			this.transformer.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		if (paramIsEntropyTrainer()) this.transformer.setTrainer(new TaskTrainerLossEntropy());
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public TransformerClassifierAbstract(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	protected boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension outputSize2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2)) return false;
		TransformerInitializer initializer = new TransformerInitializer(this.transformer);
		if (!initializer.initializeOnlyEncoder(inputSize1.height, inputSize1.width, depth1, paramGetBlocksNumber())) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = getNumberOfGroups();
		Dimension outputCombSize = paramIsByColumn() ? new Dimension(groupCount, outputCount) : new Dimension(outputCount, groupCount);
		if (outputSize2 == null) {
			int depth = depth1 + depth2;
			depth = depth > 1 ? depth-1 : depth;
			if (!transformer.setOutputAdapter(outputCombSize, filter1, depth, dual1, null, 0))
				return false;
		}
		else {
			depth1 = depth1 > 1 ? depth1-1 : depth1;
			if (!transformer.setOutputAdapter(outputSize1, filter1, depth1, dual1, outputCombSize, depth2))
				return false;
		}
		
		Matrix output = getOutput();
		if (paramIsByColumn()) {
			if (output.rows() != this.outputClassMaps.get(0).size() ||
				output.columns() != this.outputClassMaps.size()) return false;
		}
		else {
			if (output.rows() != this.outputClassMaps.size() ||
				output.columns() != this.outputClassMaps.get(0).size()) return false;
		}
		
		MatrixNetworkImpl outputAdapter = transformer.getOutputAdapter();
		if (outputAdapter != null) {
			transformer.removeOutputAdapter();
			transformer.setOutputFFN(outputAdapter);
		}
		return true;
	}

	
	@Override
	protected Matrix getOutput() {
		return transformer.getOutput();
	}

	
	@Override
	protected Matrix toMatrix(Raster raster) {
		Matrix input = transformer.getInput();
		return Matrix.toMatrix(input.rows(), input.columns(), raster, neuronChannel, transformer.paramIsNorm());
	}

	
	@Override
	protected Matrix evaluate(Matrix input, Object...params) {
		try {
			transformer.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		return transformer.evaluate(input, params);
	}

	
	@Override
	protected Error[] learn(Iterable<Record> sample) {
		try {
			transformer.getConfig().putAll(this.config);
			Iterable<net.ea.ann.transformer.Record> transformerSample = net.ea.ann.transformer.Record.create(sample);
			net.ea.ann.transformer.Error[][] transformerErrors = transformer.learn(transformerSample);
			if (transformerErrors == null || transformerErrors.length == 0 || transformerErrors[0] == null)
				return null;
			else
				return net.ea.ann.transformer.Error.extract(transformerErrors[0]);
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
	/**
	 * Getting the number of blocks.
	 * @return the number of blocks.
	 */
	int paramGetBlocksNumber() {
		int blocksNumber = config.getAsInt(BLOCKS_NUMBER_FIELD);
		return blocksNumber < 1 ? BLOCKS_NUMBER_DEFAULT : blocksNumber;
	}
	
	
	/**
	 * Setting the number of blocks.
	 * @param blockNumber the number of blocks.
	 * @return this classifier.
	 */
	TransformerClassifierAbstract paramSetBlocksNumber(int blockNumber) {
		blockNumber = blockNumber < 1 ? BLOCKS_NUMBER_DEFAULT : blockNumber;
		config.put(BLOCKS_NUMBER_FIELD, blockNumber);
		return this;
	}


}
