/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.WeightSpec;
import net.ea.ann.mane.WeightSpec.KernelType;
import net.ea.ann.mane.WeightSpec.Type;
import net.ea.ann.mane.beans.VGG;
import net.ea.ann.raster.Size;

/**
 * This class is an implementation of network weight based on matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightNetworkImpl extends WeightNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for weight kernel type.
	 */
	public final static String WEIGHT_KERNEL_TYPE_FIELD = VGG.WEIGHT_KERNEL_TYPE_FIELD;
	
	
	/**
	 * Default value for weight kernel type.
	 */
	public final static KernelType WEIGHT_KERNEL_TYPE_DEFAULT = VGG.WEIGHT_KERNEL_TYPE_DEFAULT;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public WeightNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public WeightNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public WeightNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public WeightNetworkImpl(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	@Override
	public Kernel kernel() {return new Kernel.NullKernel();}

	
	@Override
	public boolean initialize(Size inputSize, Size outputSize, int length, KernelType kernelType) {
		if (inputSize == null || outputSize == null) throw new IllegalArgumentException();
		
		LayerSpec layerSpec0 = new MatrixLayerAbstract.LayerSpec(inputSize);
		List<LayerSpec> layerSpecs = Util.newList(0);
		layerSpecs.add(layerSpec0);
		for (int i = 0; i < length; i++) {
			LayerSpec layerSpec = new LayerSpec(outputSize);
			if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
			layerSpec.weightSpec = new WeightSpec(Type.kernel);
			layerSpec.weightSpec.kernelType = kernelType;
			layerSpecs.add(layerSpec);
		}
		return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
	}

	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @param length length (number) of weight layers (network depth).
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize, int length) {
		return initialize(inputSize, outputSize, length, paramGetWeightType());
	}

	
	/**
	 * Getting weight kernel type.
	 * @return weight kernel type.
	 */
	KernelType paramGetWeightType() {
		if (config.containsKey(WEIGHT_KERNEL_TYPE_FIELD))
			return net.ea.ann.mane.WeightSpec.intToKernelType(config.getAsInt(WEIGHT_KERNEL_TYPE_FIELD));
		else
			return WEIGHT_KERNEL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting weight kernel type.
	 * @param kernelType weight kernel type.
	 * @return this model.
	 */
	WeightNetworkImpl paramSetWeightKernelType(KernelType kernelType) {
		config.put(WEIGHT_KERNEL_TYPE_FIELD, WeightSpec.kernelTypeToInt(kernelType));
		return this;
	}


	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @param length length (number) of weight layers (network depth).
	 * @param kernelType weight kernel type.
	 * @return network as weight.
	 */
	public static WeightNetworkImpl create(Size inputSize, Size outputSize, int length, KernelType kernelType, int neuronChannel) {
		length = length < 1 ? 1 : length;
		WeightNetworkImpl weight = new WeightNetworkImpl(neuronChannel);
		if (!weight.initialize(inputSize, outputSize, length, kernelType)) return null;
		return weight;
	}
	
	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @param length length (number) of weight layers (network depth).
	 * @param kernelType weight kernel type.
	 * @return network as weight.
	 */
	public static WeightNetworkImpl create(Size inputSize, Size outputSize, int length, int neuronChannel) {
		length = length < 1 ? 1 : length;
		WeightNetworkImpl weight = new WeightNetworkImpl(neuronChannel);
		if (!weight.initialize(inputSize, outputSize, length, KernelType.normal)) return null;
		return weight;
	}


}
