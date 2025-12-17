/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.util.List;
import java.util.Map;

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
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterProperty.Label;
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
	 * @param rasterChannel raster channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static VGG create(int neuronChannel, int rasterChannel, boolean isNorm) {
		Function activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		VGG vgg = new VGG(neuronChannel, activateRef, contentActivateRef, null);
		vgg.paramSetRasterChannel(rasterChannel);
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
	 * Field for number of blocks.
	 */
	public static final String BLOCKS_NUMBER_FIELD = net.ea.ann.mane.beans.VGG.BLOCKS_NUMBER_FIELD;
	
	
	/**
	 * Field for number of blocks.
	 */
	public static final int BLOCKS_NUMBER_DEFAULT = net.ea.ann.mane.beans.VGG.BLOCKS_NUMBER_DEFAULT;

	
	/**
	 * Classifier nut.
	 */
	protected net.ea.ann.mane.beans.VGG nut = null;


	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGGAbstract(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, idRef);
		this.config.put(BLOCKS_NUMBER_FIELD, BLOCKS_NUMBER_DEFAULT);
		
		this.nut = new net.ea.ann.mane.beans.VGG(this.neuronChannel, activateRef, convActivateRef, idRef);
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
		return initialize(inputSize1, outputSize2 != null ? outputSize2 : outputSize1);
	}


	/**
	 * Initializing classifier.
	 * @param inputSize1 input size 1.
	 * @param outputSize1 output size 1, which can be null.
	 * @param filter1 filter 1, which can be null.
	 * @param depth1 the number 1 of hidden layers plus output layer, which can be 0.
	 * @param dual1 dual mode 1.
	 * @param outputSize2 output size 2.
	 * @param depth2 the number 2 of hidden layers plus output layer, which can be 0.
	 * @return true if initialization is successful.
	 */
	protected boolean initialize(Size inputSize, Size nCoreClasses) {
		updateConfig();
		this.baseline = null;
		this.adjustline = null;
		
		if (!configClassInfo(nCoreClasses)) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = getNumberOfGroups();
		Size outputCombSize = paramIsByColumn() ? new Size(groupCount, outputCount, 1) : new Size(outputCount, groupCount, 1);
		if (!nut.initializeWithImplicitMiddleSize(inputSize, outputCombSize)) return false;
		
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
	public boolean initialize(List<List<Label>> labelGroups, Size averageSize) {
		//Removing empty labels and sorting labels.
		List<List<Label>> tempLabelGroups = Util.newList(labelGroups.size());
		tempLabelGroups.addAll(labelGroups);
		labelGroups.clear();
		for (List<Label> labels : tempLabelGroups) {
			if (labels.size() == 0) continue;
			Label.sort(labels, true);
			labelGroups.add(labels);
		}
		if (labelGroups.size() == 0) return false;

		//Adjusting the group label list so that every its element has the same class count.
		int minClassCount = labelGroups.get(0).size();
		for (List<Label> labels : labelGroups) {
			if (minClassCount > labels.size()) minClassCount = labels.size();
		}
		for (List<Label> labels : labelGroups) {
			if (labels.size() > minClassCount) {
				List<Label> temp = Util.newList(0);
				temp.addAll(labels);
				List<Label> sub = temp.subList(0, minClassCount);
				labels.clear();
				labels.addAll(sub);
			}
		}

		//Initializing matrix network.
		int groupCount = labelGroups.size();
		Size inputSize = new Size(averageSize.width, averageSize.height, averageSize.depth); //Flattening later in matrix neural network.
		Size nCoreClasses = paramIsByColumn() ? new Size(groupCount, minClassCount, 1) : new Size(minClassCount, groupCount, 1);
		if (!initialize(inputSize, nCoreClasses)) return false;
		
		//Main task: setting up class maps.
		this.classMaps.clear();
		for (int group = 0; group < groupCount; group++) {
			Map<Integer, Label> classMap = Util.newMap(0);
			int classCount = getNumberOfClasses(group);
			List<Label> labels = labelGroups.get(group);
			for (int classNumber = 0; classNumber < classCount; classNumber++) {
				Label label = classNumber < labels.size() ? labels.get(classNumber) : labels.get(labels.size()-1);
				if (label != null) classMap.put(classNumber, label);
			}
			if (classMap.size() > 0) this.classMaps.add(classMap);
		}
		return this.classMaps.size() > 0;
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
	VGGAbstract paramSetBlocksNumber(int blockNumber) {
		blockNumber = blockNumber < 1 ? BLOCKS_NUMBER_DEFAULT : blockNumber;
		config.put(BLOCKS_NUMBER_FIELD, blockNumber);
		return this;
	}


}
