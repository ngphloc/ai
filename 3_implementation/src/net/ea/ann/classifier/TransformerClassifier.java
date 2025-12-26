/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;
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
	MatrixNetworkImpl adjuster = null;
	
	
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
	}


	@Override
	void updateConfig() {
		super.updateConfig();
		if (adjuster != null) adjuster.paramSetInclude(this);
	}

	
	@Override
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size nCoreClasses2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2)) return false;
		this.adjuster = null;
		if (!paramIsAdjust() || !paramIsBaseline() || !paramIsCreateAdjuster()) return true;

		MatrixNetworkImpl nut = transformer.getOutputAdapter();
		nut = nut != null ? nut : transformer.getOutputFFN();
		int minAdjustDepth = Math.max((int)(Math.log(nut.size())/Math.log(NetworkAbstract.ZOOMOUT_DEFAULT)), 1);
		int maxAdjustDepth = Math.max(Math.max(nut.size()-1, 1), minAdjustDepth);
		int adjustDepth = Math.max(Math.max(depth1, depth2), minAdjustDepth);
		adjustDepth = Math.min(adjustDepth, maxAdjustDepth);
		Size size = nut.getOutputLayer().getSize();
		this.adjuster = new MatrixNetworkImpl(this.neuronChannel, nut.getActivateRef(), nut.getConvActivateRef(), this.idRef);
		this.adjuster.paramSetInclude(this);
		if (paramIsEntropyTrainer()) this.adjuster.setTrainer(new TaskTrainerLossEntropy());
		return new MatrixNetworkInitializer(adjuster).initialize(size, size, adjustDepth);
	}


	@Override
	MatrixNetworkImpl paramGetAdjuster() {return adjuster;}

	
	/**
	 * Creating matrix neural classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param rasterChannel raster channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static TransformerClassifier create(int neuronChannel, int rasterChannel, boolean isNorm) {
		TransformerClassifier tramac = new TransformerClassifier(neuronChannel);
		tramac.paramSetRasterChannel(rasterChannel);
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
	public static final String BLOCKS_NUMBER_FIELD = net.ea.ann.mane.beans.VGG.BLOCKS_NUMBER_FIELD;
	
	
	/**
	 * Field for number of blocks.
	 */
	public static final int BLOCKS_NUMBER_DEFAULT = net.ea.ann.mane.beans.VGG.BLOCKS_NUMBER_DEFAULT;

	
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
	public void reset() {
		super.reset();
		transformer.reset();
	}


	@Override
	void updateConfig() {
		super.updateConfig();
		transformer.updateConfig(this.config);
	}


	@Override
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2)) return false;
		
		TransformerInitializer initializer = new TransformerInitializer(this.transformer);
		if (!initializer.initializeOnlyEncoder(inputSize1.height, inputSize1.width, depth1, paramGetBlocksNumber())) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = getNumberOfGroups();
		Size outputCombSize = paramIsByColumn() ? new Size(groupCount, outputCount, 1) : new Size(outputCount, groupCount, 1);
		depth1 = depth1 < 0 ? 0 : depth1;
		depth2 = depth2 < 0 ? 0 : depth2;
		if (outputSize2 == null) {
			int depth = depth1 + depth2;
			depth = depth > 1 ? depth-1 : depth;
			depth = depth < 0 ? 0 : depth;
			if (!transformer.setOutputFFN(outputCombSize, filter1, depth, dual1, null, 0)) return false;
		}
		else {
//			if (outputSize1 != null && (outputSize1.height != inputSize1.height || outputSize1.width != inputSize1.width)) {
//				outputSize1.height = inputSize1.height;
//				outputSize1.width = inputSize1.width;
//			}
			
			if (depth1 <= 0 && depth2 <= 0) {
				if (!transformer.setOutputFFN(outputCombSize, filter1, 0, dual1, null, 0)) return false;
			}
			else if (depth1 > 0 && depth2 <= 0) {
				if (!transformer.setOutputFFN(outputCombSize, filter1, depth1-1, dual1, null, 0)) return false;
			}
			else if (depth1 <= 0 && depth2 > 0) {
				if (!transformer.setOutputFFN(outputCombSize, filter1, depth2-1, dual1, null, 0)) return false;
			}
			else if (depth1 == 1) {
				if (depth2 == 1) {
					if (!transformer.setOutputFFN(outputCombSize, filter1, 1, dual1, null, 0)) return false;
				}
				else {
					if (!transformer.setOutputFFN(outputSize1, filter1, 1, dual1, outputCombSize, depth2-1)) return false;
				}
			}
			else {
				if (!transformer.setOutputFFN(outputSize1, filter1, depth1-1, dual1, outputCombSize, depth2)) return false;
			}
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
		
		return true;
	}

	
	@Override
	protected Matrix getOutput() {
		return transformer.getOutput();
	}

	
	@Override
	protected Matrix toMatrix(Raster raster) {
		Matrix input = transformer.getInput();
		return MatrixUtil.toMatrix(new Size(input.columns(), input.rows()), raster, neuronChannel, paramGetRasterChannel(), transformer.paramIsNorm());
	}

	
	@Override
	protected Matrix evaluate(Matrix input, Object...params) {
		updateConfig();
		return transformer.evaluate(input, params);
	}

	
	@Override
	protected Error[] learn(Iterable<Record> sample) {
		try {
			updateConfig();
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
