/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.awt.Dimension;
import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.FilterSpec.KernelType;
import net.ea.ann.mane.FilterSpec.Type;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.beans.VGG;
import net.ea.ann.raster.Size;

/**
 * This class implements default filter network (network filter).
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FilterNetworkImpl extends FilterNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for kernel type.
	 */
	public final static String KERNEL_TYPE_FIELD = VGG.FILTER_KERNEL_TYPE_FIELD;
	
	
	/**
	 * Default value for kernel type.
	 */
	public final static KernelType KERNEL_TYPE_DEFAULT = VGG.FILTER_KERNEL_TYPE_DEFAULT;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public FilterNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public FilterNetworkImpl(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public FilterNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public FilterNetworkImpl(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size in which in which width and height are size of filter itself and
	 * depth is the number of matrices (length of matrix stack) in each layer.
	 * @param length length (number) of filter layers (network depth).
	 * @param kernelType kernel type.
	 * @return true if initialization is successful.
	 */
	@Deprecated
	boolean initialize(Size inputSize, Size kernelSize, int length, KernelType kernelType) {
		if (inputSize == null) return false;
		
		LayerSpec layerSpec0 = new MatrixLayerAbstract.LayerSpec(inputSize);
		List<LayerSpec> layerSpecs = Util.newList(0);
		layerSpecs.add(layerSpec0);
		for (int i = 0; i < length; i++) {
			LayerSpec layerSpec = new LayerSpec(new Size(inputSize.width, inputSize.height, kernelSize.depth));
			if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
			layerSpec.filterSpec = new FilterSpec(kernelSize.width, kernelSize.height, Type.kernel);
			layerSpec.filterSpec.kernelType = kernelType; //It should be set as KernelType.product_max
			layerSpec.filterSpec.moveStride = false;
			layerSpecs.add(layerSpec);
		}
		return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
	}

	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size in which in which width and height are size of filter itself and
	 * depth is the number of matrices (length of matrix stack) in each layer.
	 * @param length length (number) of filter layers (network depth).
	 * @return true if initialization is successful.
	 */
	@Deprecated
	boolean initialize(Size inputSize, Size kernelSize, int length) {
		return initialize(inputSize, kernelSize, length, paramGetKernelType());
	}

	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size in which in which width and height are size of filter itself.
	 * @param kernelDepth output depth is also kernel depth. Output depth is the number of matrices (length of matrix stack) in each layer.
	 * @param length the number of (filter) layers in network (network depth).
	 * @param kernelType kernel type.
	 * @return true if initialization is successful.
	 */
	@Deprecated
	boolean initialize(Size inputSize, Size kernelSize, int outputDepth, int length, KernelType kernelType) {
		return initialize(inputSize, new Size(kernelSize.width, kernelSize.height, outputDepth), length, kernelType);
	}
	
	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size in which in which width and height are size of filter itself.
	 * @param kernelDepth output depth is also kernel depth. Output depth is the number of matrices (length of matrix stack) in each layer.
	 * @param length the number of (filter) layers in network (network depth).
	 * @param kernelType kernel type.
	 * @return true if initialization is successful.
	 */
	@Deprecated
	boolean initialize(Size inputSize, Size kernelSize, int outputDepth, int length) {
		return initialize(inputSize, kernelSize, outputDepth, length, paramGetKernelType());
	}


	@Override
	public boolean initialize(Size inputSize, Size outputSize, int length, Dimension kernelSize, KernelType kernelType) {
		if (inputSize == null || outputSize == null) throw new IllegalArgumentException();
		
		LayerSpec layerSpec0 = new MatrixLayerAbstract.LayerSpec(inputSize);
		List<LayerSpec> layerSpecs = Util.newList(0);
		layerSpecs.add(layerSpec0);
		for (int i = 0; i < length; i++) {
			LayerSpec layerSpec = new LayerSpec(outputSize);
			if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
			layerSpec.filterSpec = new FilterSpec(kernelSize.width, kernelSize.height, Type.kernel);
			layerSpec.filterSpec.kernelType = kernelType; //It should be set as KernelType.product_max
			layerSpec.filterSpec.moveStride = false;
			layerSpecs.add(layerSpec);
		}
		return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
	}
	
	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @param length length (number) of filter layers (network depth).
	 * @param kernelSize kernel size in which in which width and height are size of filter itself.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize, int length, Dimension kernelSize) {
		return initialize(inputSize, outputSize, length, kernelSize, paramGetKernelType());
	}
	
	
	/**
	 * Checking kernel filter type.
	 * @return kernel filter type.
	 */
	KernelType paramGetKernelType() {
		if (config.containsKey(KERNEL_TYPE_FIELD))
			return FilterSpec.intToKernelType(config.getAsInt(KERNEL_TYPE_FIELD));
		else
			return KERNEL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting kernel filter type.
	 * @param kernelType kernel filter type.
	 * @return this classifier.
	 */
	FilterNetworkImpl paramSetKernelType(KernelType kernelType) {
		config.put(KERNEL_TYPE_FIELD, FilterSpec.kernelTypeToInt(kernelType));
		return this;
	}

	
	/**
	 * Creating network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size is different, in which in which width and height are size of filter itself and
	 * depth is the number of matrices (length of matrix stack) in each layer whereas time is the number of (filter) layers in network (network depth).
	 * @param neuronChannel neuronChannel.
	 * @param kernelType kernel type.
	 * @return network as filter.
	 */
	@Deprecated
	static FilterNetworkImpl create(Size inputSize, Size kernelSize, int neuronChannel, KernelType kernelType) {
		int length = kernelSize.time < 1 ? 1 : kernelSize.time;
		FilterNetworkImpl filter = new FilterNetworkImpl(neuronChannel);
		if (!filter.initialize(inputSize, kernelSize, length, kernelType)) return null;
		return filter;
	}
	
	
	/**
	 * Creating network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size is different, in which in which width and height are size of filter itself and
	 * depth is the number of matrices (length of matrix stack) in each layer whereas time is the number of (filter) layers in network (network depth).
	 * @param neuronChannel neuronChannel.
	 * @return network as filter.
	 */
	@Deprecated
	static FilterNetworkImpl create(Size inputSize, Size kernelSize, int neuronChannel) {
		return create(inputSize, kernelSize, neuronChannel, KernelType.product);
	}

	
	/**
	 * Creating network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size is different, in which in which width and height are size of filter itself.
	 * @param kernelDepth kernel depth is also output depth. Kernel depth is the number of matrices (length of matrix stack) in each layer.
	 * @param length the number of (filter) layers in network (network depth).
	 * @param neuronChannel neuronChannel.
	 * @param kernelType kernel type.
	 * @return network as filter.
	 */
	@Deprecated
	static FilterNetworkImpl create(Dimension inputSize, Dimension kernelSize, int kernelDepth, int length, int neuronChannel, KernelType kernelType) {
		return create(
			new Size(inputSize.width, inputSize.height, kernelDepth),
			new Size(kernelSize.width, kernelSize.height, kernelDepth, length), 
			neuronChannel,
			kernelType);
	}
	
	
	/**
	 * Creating network as filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size is different, in which in which width and height are size of filter itself.
	 * @param kernelDepth kernel depth is also output depth. Kernel depth is the number of matrices (length of matrix stack) in each layer.
	 * @param length the number of (filter) layers in network (network depth).
	 * @param neuronChannel neuronChannel.
	 * @return network as filter.
	 */
	@Deprecated
	static FilterNetworkImpl create(Dimension inputSize, Dimension kernelSize, int kernelDepth, int length, int neuronChannel) {
		return create(inputSize, kernelSize, kernelDepth, length, neuronChannel, KernelType.product);
	}
	
	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @param length length (number) of filter layers (network depth).
	 * @param kernelSize kernel size in which in which width and height are size of filter itself.
	 * @param kernelType kernel type.
	 * @param neuronChannel neuronChannel.
	 * @return network as filter.
	 */
	public static FilterNetworkImpl create(Size inputSize, Size outputSize, int length, Dimension kernelSize, KernelType kernelType, int neuronChannel) {
		length = length < 1 ? 1 : length;
		FilterNetworkImpl filter = new FilterNetworkImpl(neuronChannel);
		if (!filter.initialize(inputSize, outputSize, length, kernelSize, kernelType)) return null;
		return filter;
	}
	
	
	/**
	 * Initializing network as filter.
	 * @param inputSize input size.
	 * @param outputSize output size.
	 * @param length length (number) of filter layers (network depth).
	 * @param kernelSize kernel size in which in which width and height are size of filter itself.
	 * @param neuronChannel neuronChannel.
	 * @return network as filter.
	 */
	public static FilterNetworkImpl create(Size inputSize, Size outputSize, int length, Dimension kernelSize, int neuronChannel) {
		return create(inputSize, outputSize, length, kernelSize, KernelType.product, neuronChannel);
	}


}
