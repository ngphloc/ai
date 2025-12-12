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
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.mane.beans.VGGBlocks;
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is an implementation of classifier within VGG model developed by Simonyan and Zisserman.
 * 
 * @author Simonyan and Zisserman, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class VGG extends VGGAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier nut.
	 */
	MatrixNetworkImpl adjuster = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGG(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public VGG(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public VGG(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public VGG(int neuronChannel) {
		this(neuronChannel, null, null, null);
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
		
		int minAdjustDepth = Math.max((int)(Math.log(this.nut.size())/Math.log(NetworkAbstract.ZOOMOUT_DEFAULT)), 1);
		int maxAdjustDepth = Math.max(Math.max(this.nut.size()-1, 1), minAdjustDepth);
		int adjustDepth = Math.max(Math.max(depth1, depth2), minAdjustDepth);
		adjustDepth = Math.min(adjustDepth, maxAdjustDepth);
		Size size = this.nut.getOutputLayer().getSize();
		this.adjuster = new MatrixNetworkImpl(this.neuronChannel, this.nut.getActivateRef(), this.nut.getConvActivateRef(), this.idRef);
		this.adjuster.paramSetInclude(this);
		if (paramIsEntropyTrainer()) this.adjuster.setTrainer(new TaskTrainerLossEntropy());
		return new MatrixNetworkInitializer(adjuster).initialize(size, size, adjustDepth);
	}


	@Override
	MatrixNetworkImpl paramGetAdjuster() {return adjuster;}


	/**
	 * Creating matrix neural classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static VGG create(int neuronChannel, boolean isNorm) {
		Function activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		VGG vgg = new VGG(neuronChannel, activateRef, contentActivateRef, null);
		vgg.paramSetNorm(isNorm);
		return vgg;
	}


}



/**
 * This class is abstract implementation of classifier within context of matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class VGGAbstract extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier nut.
	 */
	protected VGGBlocks nut = null;


	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGGAbstract(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, idRef);
		
		this.nut = new VGGBlocks(this.neuronChannel, activateRef, convActivateRef, idRef);
		try {
			this.config.putAll(this.nut.getConfig());
			this.nut.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		if (paramIsEntropyTrainer()) this.nut.setTrainer(new TaskTrainerLossEntropy());
	}
	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public VGGAbstract(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public VGGAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public VGGAbstract(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	@Override
	public void reset() {
		super.reset();
		nut.reset();
	}

	
	@Override
	void updateConfig() {
		super.updateConfig();
		nut.paramSetInclude(this);
	}

	
	@Override
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2)) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = getNumberOfGroups();
		Size outputCombSize = paramIsByColumn() ? new Size(groupCount, outputCount, 1) : new Size(outputCount, groupCount, 1);
		if (outputSize2 == null) {
			if (paramGetMiddleSize() <= 0) {
				if (!nut.initializeByDepth(inputSize1, outputCombSize, filter1, depth1, dual1, outputSize2, depth2))
					return false;
			}
			else {
				if (!nut.initialize(inputSize1, outputCombSize, filter1, depth1, dual1, outputSize2, depth2))
					return false;
			}
		}
		else {
			if (paramGetMiddleSize() <= 0) {
				if (!nut.initializeByDepth(inputSize1, outputSize1, filter1, depth1, dual1, outputCombSize, depth2))
					return false;
			}
			else {
				if (!nut.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputCombSize, depth2))
					return false;
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
		return nut.getOutput();
	}

	
	@Override
	protected Matrix toMatrix(Raster raster) {
		return nut.getInputLayer().toMatrix(raster);
	}

	
	@Override
	protected Matrix evaluate(Matrix input, Object...params) {
		updateConfig();
		return nut.evaluate0(input, params);
	}

	
	@Override
	protected Error[] learn(Iterable<Record> sample) {
		try {
			updateConfig();
			return nut.learn(sample);
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}
	

}
