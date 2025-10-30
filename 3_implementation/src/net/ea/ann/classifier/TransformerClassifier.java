/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.awt.Dimension;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.raster.Raster;
import net.ea.ann.transformer.TransformerBasic;
import net.ea.ann.transformer.TransformerImpl;
import net.ea.ann.transformer.TransformerInitializer;

/**
 * This class is default implementation of classifier within context of transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerClassifier extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for number of blocks.
	 */
	public static final String BLOCKS_NUMBER_FIELD = "tramac_blocks";
	
	
	/**
	 * Internal transformer.
	 */
	protected TransformerImpl transformer = null;
	
	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	public TransformerClassifier(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
		this.config.put(BLOCKS_NUMBER_FIELD, TransformerBasic.BLOCKS_NUMBER_DEFAULT);
		
		this.transformer = new TransformerImpl(this.neuronChannel, idRef);
		try {
			this.config.putAll(this.transformer.getConfig());
		} catch (Throwable e) {Util.trace(e);}
		this.transformer.setTrainer(new TaskTrainerLossEntropy());
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public TransformerClassifier(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	protected boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension nCoreClasses2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2)) return false;
		TransformerInitializer initializer = new TransformerInitializer(this.transformer);
		if (!initializer.initializeOnlyEncoder(inputSize1.height, inputSize1.width, depth1, paramGetBlocksNumber())) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = paramIsByColumn() ? nCoreClasses2.width : nCoreClasses2.height;
		Dimension outputSize2 = paramIsByColumn() ? new Dimension(groupCount, outputCount) : new Dimension(outputCount, groupCount);
		boolean initialized = false;
		if (outputSize1 != null || depth1 > 0) {
			int halfDepth = depth2/2;
			if (paramIsConv())
				initialized = transformer.setOutputAdapter(outputSize1, filter1, halfDepth, dual1, outputSize2, halfDepth);
			else
				initialized = transformer.setOutputAdapter(outputSize1, (Filter2D)null, halfDepth, false, outputSize2, halfDepth);
		}
		else
			initialized = transformer.setOutputAdapter(outputSize2, depth2);
		if (!initialized) return false;
		
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
		return Matrix.toMatrix(input.rows(), input.columns(), raster, neuronChannel, transformer.paramIsNorm());
	}

	
	@Override
	protected Matrix evaluate(Matrix input) {
		try {
			transformer.getConfig().putAll(this.config);
			return transformer.evaluate(input);
		} catch (Throwable e) {Util.trace(e);}
		return null;
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
				return net.ea.ann.transformer.Error.create2(transformerErrors[0]);
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
	/**
	 * Getting the number of blocks.
	 * @return the number of blocks.
	 */
	int paramGetBlocksNumber() {
		int blocksNumber = config.getAsInt(BLOCKS_NUMBER_FIELD);
		return blocksNumber < 1 ? TransformerBasic.BLOCKS_NUMBER_DEFAULT : blocksNumber;
	}
	
	
	/**
	 * Setting the number of blocks.
	 * @param blockNumber the number of blocks.
	 * @return this classifier.
	 */
	TransformerClassifier paramSetBlocksNumber(int blockNumber) {
		blockNumber = blockNumber < 1 ? TransformerBasic.BLOCKS_NUMBER_DEFAULT : blockNumber;
		config.put(BLOCKS_NUMBER_FIELD, blockNumber);
		return this;
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
