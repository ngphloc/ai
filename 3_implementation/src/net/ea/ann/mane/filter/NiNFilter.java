/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.FilterSpec.KernelType;
import net.ea.ann.mane.FilterSpec.Type;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.raster.Size;

/**
 * This class implements network-in-network (NiN) filter developed by Min Lin, Qiang Chen, Shuicheng Yan.
 * @author Min Lin, Qiang Chen, Shuicheng Yan, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class NiNFilter extends FilterNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public NiNFilter(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public NiNFilter(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public NiNFilter(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public NiNFilter(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	/**
	 * Initializing NiN filter.
	 * @param inputSize input size.
	 * @param kernelSize kernel size.
	 * @param length length of filter layers.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size kernelSize, int length) {
		if (inputSize == null) return false;
		
		int rasterChannel = paramGetRasterChannel();
		boolean flatten = MatrixUtil.isFlatten(inputSize.depth, this.neuronChannel, rasterChannel); //inputSize.depth is actually raster depth.
		LayerSpec layerSpec0 = new MatrixLayerAbstract.LayerSpec(new Size(inputSize.width, inputSize.height, flatten?rasterChannel:1));
		List<LayerSpec> layerSpecs = Util.newList(0);
		layerSpecs.add(layerSpec0);
		for (int i = 0; i < length; i++) {
			LayerSpec layerSpec = new LayerSpec(new Size(inputSize.width, inputSize.height, kernelSize.depth));
			if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
			layerSpec.filterSpec = new FilterSpec(kernelSize.width, kernelSize.height, Type.kernel);
			layerSpec.filterSpec.kernelType = KernelType.max;
			layerSpec.filterSpec.moveStride = false;
			layerSpecs.add(layerSpec);
		}
		return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
	}


}
