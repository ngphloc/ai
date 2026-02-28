/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.FilterSpec.NetworkType;
import net.ea.ann.mane.FilterSpec.PoolType;
import net.ea.ann.mane.FilterSpec.Type;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.raster.Size;

/**
 * This class implements network-in-network (NiN) developed by Min Lin, Qiang Chen, Shuicheng Yan.
 * @author Min Lin, Qiang Chen, Shuicheng Yan, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class NiN extends MatrixNetworkImpl {


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
	public NiN(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(VGG.MIDDLE_SIZE_FIELD, VGG.MIDDLE_SIZE_DEFAULT_TEXT);
		config.put(VGG.BLOCKS_NUMBER_FIELD, VGG.BLOCKS_NUMBER_DEFAULT);
		config.put(VGG.LAYERS_NUMBER_FIELD, VGG.LAYERS_NUMBER_DEFAULT);
		config.put(VGG.FILTERS_NUMBER_FIELD, VGG.FILTERS_NUMBER_DEFAULT);
		config.put(VGG.FILTER_SIZE_FIELD, VGG.FILTER_SIZE_DEFAULT);
		config.put(VGG.FFN_LENGTH_FIELD, VGG.FFN_LENGTH_DEFAULT);
		config.put(VGG.FFN_FLATTEN_FIELD, VGG.FFN_FLATTEN_DEFAULT);
		config.put(VGG.POOL_TYPE_FIELD, FilterSpec.poolTypeToInt(VGG.POOL_TYPE_DEFAULT));
		config.put(VGG.NETWORK_TYPE_FIELD, FilterSpec.networkTypeToInt(VGG.NETWORK_TYPE_DEFAULT));
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public NiN(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public NiN(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public NiN(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	/**
	 * Initializing NiN model.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	protected boolean initialize(Size inputSize, Size middleSize, Size outputSize) {
		List<Size> blockSizes = VGG.calcBlockSizes(inputSize, middleSize, paramGetBlocksNumber(), paramGetFiltersNumber());
		if (blockSizes.size() == 0) return false;
		
		int base = VGG.BASE;
		int layersNumberPerBlock = paramGetLayersNumber();
		int filterSize = paramGetFilterSize();
		int ffnLength = paramGetFFNLength();

		int rasterChannel = paramGetRasterChannel();
		boolean flatten = MatrixUtil.isFlatten(inputSize.depth, this.neuronChannel, rasterChannel); //inputSize.depth is actually raster depth.
		LayerSpec layerSpec0 = new MatrixLayerAbstract.LayerSpec(new Size(inputSize.width, inputSize.height, flatten?rasterChannel:1));
		List<LayerSpec> layerSpecs = Util.newList(0);
		layerSpecs.add(layerSpec0);
		for (int i = 0; i < blockSizes.size(); i++) {
			Size blockSize = blockSizes.get(i);
			for (int j = 0; j < layersNumberPerBlock; j++) {
				LayerSpec layerSpec = new LayerSpec(new Size(blockSize.width, blockSize.height, blockSize.depth));
				if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
				layerSpec.filterSpec = new FilterSpec(filterSize, filterSize, Type.network);
				layerSpec.filterSpec.networkType = paramGetNetworkType();
				layerSpec.filterSpec.moveStride = false;
				layerSpecs.add(layerSpec);
			}
			if (i < blockSizes.size()-1) {
				Size poolSize = new Size(blockSizes.get(i+1).width, blockSizes.get(i+1).height, blockSize.depth);
				LayerSpec layerSpec = new LayerSpec(poolSize);
				if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
				layerSpec.filterSpec = new FilterSpec(base, base, Type.pool);
				layerSpec.filterSpec.poolType = paramGetPoolType();
				layerSpec.filterSpec.moveStride = true;
				layerSpecs.add(layerSpec);
			}
		}
		if (outputSize == null || ffnLength < 1) return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
		
		Size lastSize = layerSpecs.get(layerSpecs.size()-1).size;
		Size ffnSize = paramIsFFNFlatten() ?
			new Size(middleSize.width, lastSize.depth*middleSize.height, 1) :
			new Size(middleSize.width, middleSize.height, lastSize.depth);
		List<LayerSpec> ffnlLayerSpecs = Util.newList(0);
		for (int i = 0; i < ffnLength-1; i++) {
			ffnlLayerSpecs.add(new LayerSpec(new Size(ffnSize)));
		}
		int outputDepth = outputSize.depth < 1 ? 1 : outputSize.depth;
		ffnlLayerSpecs.add(new LayerSpec(new Size(outputSize.width, outputSize.height, outputDepth)));
		
		for (int i = 0; i < ffnlLayerSpecs.size(); i++) {
			if (i > 0) ffnlLayerSpecs.get(i).prevSize = ffnlLayerSpecs.get(i-1).size;
			if (paramIsVectorized()) {
				ffnlLayerSpecs.get(i).vecRows = ffnlLayerSpecs.get(i).size.height;
				ffnlLayerSpecs.get(i).size = new Size(1, ffnlLayerSpecs.get(i).size.width*ffnlLayerSpecs.get(i).size.height, ffnlLayerSpecs.get(i).size.depth);
			}
		}
		layerSpecs.addAll(ffnlLayerSpecs);
		return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
	}
	
	
	/**
	 * Initializing NiN model.
	 * @param inputSize input size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize) {
		return initialize(inputSize, outputSize, null);
	}
	
	
	/**
	 * Initializing NiN model.
	 * @param inputSize input size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Size inputSize, Size outputSize) {
		return initialize(inputSize, paramGetVGGMiddleSize(), outputSize);
	}

	
	/**
	 * Getting VGG middle size.
	 * @return VGG middle size.
	 */
	public Size paramGetVGGMiddleSize() {
		String sizeText = config.containsKey(VGG.MIDDLE_SIZE_FIELD) ? config.getAsString(VGG.MIDDLE_SIZE_FIELD) : VGG.MIDDLE_SIZE_DEFAULT_TEXT;
		return VGG.paramGetVGGMiddleSize(sizeText);
	}
	
	
	/**
	 * Setting VGG middle size.
	 * @param middleSize VGG middle size.
	 * @return this NiN.
	 */
	public NiN paramSetVGGMiddleSize(Size middleSize) {
		int width = middleSize.width < 1 ? MINSIZE : middleSize.width;
		int height = middleSize.height < 1 ? MINSIZE : middleSize.height;
		config.put(VGG.MIDDLE_SIZE_FIELD, width + ", " + height);
		return this;
	}

	
	/**
	 * Getting number of blocks.
	 * @return number of blocks.
	 */
	public int paramGetBlocksNumber() {
		if (config.containsKey(VGG.BLOCKS_NUMBER_FIELD))
			return config.getAsInt(VGG.BLOCKS_NUMBER_FIELD);
		else
			return VGG.BLOCKS_NUMBER_DEFAULT;
	}
	
	
	/**
	 * Setting number of blocks.
	 * @param blocks number of blocks.
	 * @return this NiN.
	 */
	public NiN paramSetBlocksNumber(int blocks) {
		blocks = blocks < 1 ? VGG.BLOCKS_NUMBER_DEFAULT : blocks;
		config.put(VGG.BLOCKS_NUMBER_FIELD, blocks);
		return this;
	}


	/**
	 * Getting number of layers per block.
	 * @return number of layers per block.
	 */
	public int paramGetLayersNumber() {
		if (config.containsKey(VGG.LAYERS_NUMBER_FIELD))
			return config.getAsInt(VGG.LAYERS_NUMBER_FIELD);
		else
			return VGG.LAYERS_NUMBER_DEFAULT;
	}
	
	
	/**
	 * Setting number of layers per block.
	 * @param layersNumber number of layers per block.
	 * @return this VGG.
	 */
	public NiN paramSetLayersNumber(int layersNumber) {
		layersNumber = layersNumber < 1 ? VGG.LAYERS_NUMBER_DEFAULT : layersNumber;
		config.put(VGG.LAYERS_NUMBER_FIELD, layersNumber);
		return this;
	}


	/**
	 * Getting number of filters per layer.
	 * @return number of filters per layer.
	 */
	public int paramGetFiltersNumber() {
		if (config.containsKey(VGG.FILTERS_NUMBER_FIELD))
			return config.getAsInt(VGG.FILTERS_NUMBER_FIELD);
		else
			return VGG.FILTERS_NUMBER_DEFAULT;
	}
	
	
	/**
	 * Setting number of filters per layer.
	 * @param filtersNumber number of filters per layer.
	 * @return this NiN.
	 */
	public NiN paramSetFiltersNumber(int filtersNumber) {
		filtersNumber = filtersNumber < 1 ? VGG.FILTERS_NUMBER_DEFAULT : filtersNumber;
		config.put(VGG.FILTERS_NUMBER_FIELD, filtersNumber);
		return this;
	}


	/**
	 * Getting filter size.
	 * @return filter size.
	 */
	int paramGetFilterSize() {
		int filterSize = config.getAsInt(VGG.FILTER_SIZE_FIELD);
		return filterSize < 1 ? VGG.FILTER_SIZE_DEFAULT : filterSize;
	}
	
	
	/**
	 * Setting filter size.
	 * @param filterSize filter size.
	 * @return this classifier.
	 */
	NiN paramSetFilterSize(int filterSize) {
		filterSize = filterSize < 1 ? VGG.FILTER_SIZE_DEFAULT : filterSize;
		config.put(VGG.FILTER_SIZE_FIELD, filterSize);
		return this;
	}


	/**
	 * Getting length of feed-forward network.
	 * @return length of feed-forward network.
	 */
	int paramGetFFNLength() {
		if (config.containsKey(VGG.FFN_LENGTH_FIELD))
			return config.getAsInt(VGG.FFN_LENGTH_FIELD);
		else
			return VGG.FFN_LENGTH_DEFAULT;
	}
	
	
	/**
	 * Setting length of feed-forward network.
	 * @param ffnLength length of feed-forward network.
	 * @return this VGG.
	 */
	NiN paramSetFFNLength(int ffnLength) {
		ffnLength = ffnLength < 1 ? VGG.FFN_LENGTH_DEFAULT : ffnLength;
		config.put(VGG.FFN_LENGTH_FIELD, ffnLength);
		return this;
	}


	/**
	 * Checking flattening feed-forward network mode.
	 * @return flattening feed-forward network mode.
	 */
	boolean paramIsFFNFlatten() {
		if (config.containsKey(VGG.FFN_FLATTEN_FIELD))
			return config.getAsBoolean(VGG.FFN_FLATTEN_FIELD);
		else
			return VGG.FFN_FLATTEN_DEFAULT;
	}
	
	
	/**
	 * Setting flattening feed-forward network mode.
	 * @param ffnFlatten flattening feed-forward network mode.
	 * @return this classifier.
	 */
	NiN paramSetFFNFlatten(boolean ffnFlatten) {
		config.put(VGG.FFN_FLATTEN_FIELD, ffnFlatten);
		return this;
	}


	/**
	 * Checking pooling filter type.
	 * @return pooling filter type.
	 */
	PoolType paramGetPoolType() {
		if (config.containsKey(VGG.POOL_TYPE_FIELD))
			return FilterSpec.intToPoolType(config.getAsInt(VGG.POOL_TYPE_FIELD));
		else
			return VGG.POOL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting pooling filter type.
	 * @param poolType pooling filter type.
	 * @return this classifier.
	 */
	NiN paramSetPoolType(PoolType poolType) {
		config.put(VGG.POOL_TYPE_FIELD, FilterSpec.poolTypeToInt(poolType));
		return this;
	}


	/**
	 * Checking network filter type.
	 * @return network filter type.
	 */
	NetworkType paramGetNetworkType() {
		if (config.containsKey(VGG.NETWORK_TYPE_FIELD))
			return FilterSpec.intToNetworkType(config.getAsInt(VGG.NETWORK_TYPE_FIELD));
		else
			return VGG.NETWORK_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting network filter type.
	 * @param networkType network filter type.
	 * @return this classifier.
	 */
	NiN paramSetNetworkType(NetworkType networkType) {
		config.put(VGG.NETWORK_TYPE_FIELD, FilterSpec.networkTypeToInt(networkType));
		return this;
	}

	
}
