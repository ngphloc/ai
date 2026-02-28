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
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.WeightSpec;
import net.ea.ann.raster.Size;
import net.ea.ann.transformer.TransformerBasic;
import net.hudup.core.parser.TextParserUtil;

/**
 * This class is an implementation of VGG blocks developed by Simonyan and Zisserman with support of matrix.
 * 
 * @author Simonyan and Zisserman, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class VGG extends MatrixNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default base.
	 */
	final static int BASE = ZOOMOUT_DEFAULT;
	
	
	/**
	 * Field for middle size.
	 */
	public final static String MIDDLE_SIZE_FIELD = "vgg_midsize";
	
	
	/**
	 * Default value for middle size.
	 */
	public final static Size MIDDLE_SIZE_DEFAULT = new Size(MINSIZE, MINSIZE);

	
	/**
	 * Default text value for middle size.
	 */
	public final static String MIDDLE_SIZE_DEFAULT_TEXT = MINSIZE + ", " + MINSIZE;
	
	
	/**
	 * Field for convolutional filter.
	 */
	public final static String CONV_FIELD = "mane_conv";
	
	
	/**
	 * Default value for convolutional filter.
	 */
	public final static boolean CONV_DEFAULT = true;

	
	/**
	 * Field for number of blocks.
	 */
	public final static String BLOCKS_NUMBER_FIELD = "mane_blocks";
	
	
	/**
	 * Default value for number of blocks.
	 */
	public final static int BLOCKS_NUMBER_DEFAULT = TransformerBasic.BLOCKS_NUMBER_DEFAULT;
	
	
	/**
	 * Field for number of layers per block.
	 */
	public final static String LAYERS_NUMBER_FIELD = "mane_depth";
	
	
	/**
	 * Default value for number of layers per block.
	 */
	public final static int LAYERS_NUMBER_DEFAULT = DEPTH_DEFAULT;

	
	/**
	 * Field for number of filters.
	 */
	public final static String FILTERS_NUMBER_FIELD = "vgg_filters";
	
	
	/**
	 * Default value for number of filters.
	 */
	public final static int FILTERS_NUMBER_DEFAULT = DEPTH_DEFAULT;

	
	/**
	 * Field for filter size.
	 */
	public static final String FILTER_SIZE_FIELD = "mane_filter_size";
	
	
	/**
	 * Default value for filter size.
	 */
	public static final int FILTER_SIZE_DEFAULT = BASE_DEFAULT;

	
	/**
	 * Field for length of feed-forward network.
	 */
	public final static String FFN_LENGTH_FIELD = "vgg_ffn_length";
	
	
	/**
	 * Default value for length of feed-forward network.
	 */
	public final static int FFN_LENGTH_DEFAULT = DEPTH_DEFAULT;

	
	/**
	 * Field for flattening feed-forward network mode.
	 */
	public final static String FFN_FLATTEN_FIELD = "vgg_ffn_flatten";
	
	
	/**
	 * Default value for flattening feed-forward network mode.
	 */
	public final static boolean FFN_FLATTEN_DEFAULT = false;

	
	/**
	 * Field for co-weight mode.
	 */
	public final static String COWEIGHT_FIELD = "vgg_coweight";
	
	
	/**
	 * Default value for co-weight mode.
	 */
	public final static boolean COWEIGHT_DEFAULT = FilterSpec.COWEIGHT;

	
	/**
	 * Field for pool type.
	 */
	public final static String POOL_TYPE_FIELD = "mane_pool_type";
	
	
	/**
	 * Default value for pool type.
	 */
	public final static PoolType POOL_TYPE_DEFAULT = PoolType.max;

	
	/**
	 * Field for network type.
	 */
	public final static String NETWORK_TYPE_FIELD = "mane_network_type";

	
	/**
	 * Default value for network type.
	 */
	public final static NetworkType NETWORK_TYPE_DEFAULT = NetworkType.nin;

	
	/**
	 * Field for weight type.
	 */
	public final static String WEIGHT_TYPE_FIELD = "mane_weight_type";
	
	
	/**
	 * Default value for weight type.
	 */
	public final static net.ea.ann.mane.WeightSpec.Type WEIGHT_TYPE_DEFAULT = net.ea.ann.mane.WeightSpec.Type.normal;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGG(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(MIDDLE_SIZE_FIELD, MIDDLE_SIZE_DEFAULT_TEXT);
		config.put(CONV_FIELD, CONV_DEFAULT);
		config.put(BLOCKS_NUMBER_FIELD, BLOCKS_NUMBER_DEFAULT);
		config.put(LAYERS_NUMBER_FIELD, LAYERS_NUMBER_DEFAULT);
		config.put(FILTERS_NUMBER_FIELD, FILTERS_NUMBER_DEFAULT);
		config.put(FILTER_SIZE_FIELD, FILTER_SIZE_DEFAULT);
		config.put(FFN_LENGTH_FIELD, FFN_LENGTH_DEFAULT);
		config.put(FFN_FLATTEN_FIELD, FFN_FLATTEN_DEFAULT);
		config.put(COWEIGHT_FIELD, COWEIGHT_DEFAULT);
		config.put(POOL_TYPE_FIELD, FilterSpec.poolTypeToInt(POOL_TYPE_DEFAULT));
		config.put(WEIGHT_TYPE_FIELD, WeightSpec.typeToInt(WEIGHT_TYPE_DEFAULT));
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

	
	/**
	 * Calculating block size.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param blocksNumber number of blocks.
	 * @param filtersNumberPerLayer number of filters per layer.
	 * @return true if initialization is successful.
	 */
	static List<Size> calcBlockSizes(Size inputSize, Size middleSize, int blocksNumber, int filtersNumberPerLayer) {
		if (inputSize == null) return Util.newList(0);
		int base = BASE;
		int r = Math.min(inputSize.width/middleSize.width, inputSize.height/middleSize.height);
		if (r < 1)
			middleSize = new Size(inputSize.width, inputSize.height);
		else
			middleSize = new Size(inputSize.width/r, inputSize.height/r);
		int[][] numbers = MatrixNetworkInitializer.constructHiddenOutputNeuronNumbers(inputSize, middleSize, base, base, blocksNumber);
		if (numbers == null) return Util.newList(0);
		
		int[] heights = numbers[0];
		int[] widths = numbers[1];
		int[] tempHeights = new int[heights.length+1], tempWidths = new int[widths.length+1];
		tempHeights[0] = inputSize.height;
		tempWidths[0] = inputSize.width;
		for (int i = 0; i < heights.length; i++) {
			tempHeights[i+1] = heights[i];
			tempWidths[i+1] = widths[i];
		}
		heights = tempHeights;
		widths = tempWidths;

		List<Size> blockSizes = Util.newList(0);
		int power = 1;
		for (int i = 0; i < heights.length; i++) {
			if (blockSizes.size() == 0) {
				blockSizes.add(new Size(widths[i], heights[i], (int)Math.pow(filtersNumberPerLayer, power)));
				power++;
				continue;
			}
			
			int prevWidth = blockSizes.get(blockSizes.size()-1).width;
			int prevHeight = blockSizes.get(blockSizes.size()-1).height;
			if (widths[i] != prevWidth || heights[i] != prevHeight) {
				blockSizes.add(new Size(widths[i], heights[i], (int)Math.pow(filtersNumberPerLayer, power)));
				power++;
			}
		}
		return blockSizes;
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	protected boolean initialize(Size inputSize, Size middleSize, Size outputSize) {
		List<Size> blockSizes = calcBlockSizes(inputSize, middleSize, paramGetBlocksNumber(), paramGetFiltersNumber());
		if (blockSizes.size() == 0) return false;
		
		int base = BASE;
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
				layerSpec.weightSpec = new WeightSpec(paramGetWeightType());
				if (paramIsConv()) {
					layerSpec.filterSpec = new FilterSpec(filterSize, filterSize, Type.kernel);
					layerSpec.filterSpec.coweight = paramIsCoweight();
					layerSpec.filterSpec.moveStride = false;
				}
				layerSpecs.add(layerSpec);
			}
			if (i < blockSizes.size()-1) {
				Size poolSize = new Size(blockSizes.get(i+1).width, blockSizes.get(i+1).height, blockSize.depth);
				LayerSpec layerSpec = new LayerSpec(poolSize);
				if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
				layerSpec.filterSpec = new FilterSpec(base, base, Type.pool);
				layerSpec.filterSpec.poolType = paramGetPoolType();
				layerSpec.filterSpec.coweight = paramIsCoweight();
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
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize) {
		return initialize(inputSize, outputSize, null);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initializeWithImplicitMiddleSize(Size inputSize, Size outputSize) {
		return initialize(inputSize, paramGetVGGMiddleSize(), outputSize);
	}

	
	/**
	 * Getting VGG middle size.
	 * @param sizeText size text.
	 * @return VGG middle size.
	 */
	public static Size paramGetVGGMiddleSize(String sizeText) {
		List<Integer> lsize = TextParserUtil.parseListByClass(sizeText, Integer.class, ",");
		int width = MINSIZE, height = MINSIZE;
		if (lsize.size() == 1)
			width = height = lsize.get(0);
		else if (lsize.size() > 1) {
			width = lsize.get(0);
			height = lsize.get(1);
		}
		
		width = width < 1 ? MINSIZE : width;
		height = height < 1 ? MINSIZE : height;
		return new Size(width, height);
	}
	
	
	/**
	 * Getting VGG middle size.
	 * @return VGG middle size.
	 */
	public Size paramGetVGGMiddleSize() {
		String sizeText = config.containsKey(MIDDLE_SIZE_FIELD) ? config.getAsString(MIDDLE_SIZE_FIELD) : MIDDLE_SIZE_DEFAULT_TEXT;
		return paramGetVGGMiddleSize(sizeText);
	}
	
	
	/**
	 * Setting VGG middle size.
	 * @param middleSize VGG middle size.
	 * @return this VGG.
	 */
	public VGG paramSetVGGMiddleSize(Size middleSize) {
		int width = middleSize.width < 1 ? MINSIZE : middleSize.width;
		int height = middleSize.height < 1 ? MINSIZE : middleSize.height;
		config.put(MIDDLE_SIZE_FIELD, width + ", " + height);
		return this;
	}
	
	
	/**
	 * Checking convolutional network mode.
	 * @return convolutional network mode.
	 */
	public boolean paramIsConv() {
		if (config.containsKey(CONV_FIELD))
			return config.getAsBoolean(CONV_FIELD);
		else
			return CONV_DEFAULT;
	}
	
	
	/**
	 * Setting convolutional network mode.
	 * @param conv convolutional network mode.
	 * @return this classifier.
	 */
	public VGG paramSetConv(boolean conv) {
		config.put(CONV_FIELD, conv);
		return this;
	}

	
	/**
	 * Getting number of blocks.
	 * @return number of blocks.
	 */
	public int paramGetBlocksNumber() {
		if (config.containsKey(BLOCKS_NUMBER_FIELD))
			return config.getAsInt(BLOCKS_NUMBER_FIELD);
		else
			return BLOCKS_NUMBER_DEFAULT;
	}
	
	
	/**
	 * Setting number of blocks.
	 * @param blocks number of blocks.
	 * @return this VGG.
	 */
	public VGG paramSetBlocksNumber(int blocks) {
		blocks = blocks < 1 ? BLOCKS_NUMBER_DEFAULT : blocks;
		config.put(BLOCKS_NUMBER_FIELD, blocks);
		return this;
	}
	
	
	/**
	 * Getting number of layers per block.
	 * @return number of layers per block.
	 */
	public int paramGetLayersNumber() {
		if (config.containsKey(LAYERS_NUMBER_FIELD))
			return config.getAsInt(LAYERS_NUMBER_FIELD);
		else
			return LAYERS_NUMBER_DEFAULT;
	}
	
	
	/**
	 * Setting number of layers per block.
	 * @param layersNumber number of layers per block.
	 * @return this VGG.
	 */
	public VGG paramSetLayersNumber(int layersNumber) {
		layersNumber = layersNumber < 1 ? LAYERS_NUMBER_DEFAULT : layersNumber;
		config.put(LAYERS_NUMBER_FIELD, layersNumber);
		return this;
	}

	
	/**
	 * Getting number of filters per layer.
	 * @return number of filters per layer.
	 */
	public int paramGetFiltersNumber() {
		if (config.containsKey(FILTERS_NUMBER_FIELD))
			return config.getAsInt(FILTERS_NUMBER_FIELD);
		else
			return FILTERS_NUMBER_DEFAULT;
	}
	
	
	/**
	 * Setting number of filters per layer.
	 * @param filtersNumber number of filters per layer.
	 * @return this VGG.
	 */
	public VGG paramSetFiltersNumber(int filtersNumber) {
		filtersNumber = filtersNumber < 1 ? FILTERS_NUMBER_DEFAULT : filtersNumber;
		config.put(FILTERS_NUMBER_FIELD, filtersNumber);
		return this;
	}
	
	
	/**
	 * Getting filter size.
	 * @return filter size.
	 */
	int paramGetFilterSize() {
		int filterSize = config.getAsInt(FILTER_SIZE_FIELD);
		return filterSize < 1 ? FILTER_SIZE_DEFAULT : filterSize;
	}
	
	
	/**
	 * Setting filter size.
	 * @param filterSize filter size.
	 * @return this classifier.
	 */
	VGG paramSetFilterSize(int filterSize) {
		filterSize = filterSize < 1 ? FILTER_SIZE_DEFAULT : filterSize;
		config.put(FILTER_SIZE_FIELD, filterSize);
		return this;
	}

	
	/**
	 * Getting length of feed-forward network.
	 * @return length of feed-forward network.
	 */
	int paramGetFFNLength() {
		if (config.containsKey(FFN_LENGTH_FIELD))
			return config.getAsInt(FFN_LENGTH_FIELD);
		else
			return FFN_LENGTH_DEFAULT;
	}
	
	
	/**
	 * Setting length of feed-forward network.
	 * @param ffnLength length of feed-forward network.
	 * @return this VGG.
	 */
	VGG paramSetFFNLength(int ffnLength) {
		ffnLength = ffnLength < 1 ? FFN_LENGTH_DEFAULT : ffnLength;
		config.put(FFN_LENGTH_FIELD, ffnLength);
		return this;
	}


	/**
	 * Checking flattening feed-forward network mode.
	 * @return flattening feed-forward network mode.
	 */
	boolean paramIsFFNFlatten() {
		if (config.containsKey(FFN_FLATTEN_FIELD))
			return config.getAsBoolean(FFN_FLATTEN_FIELD);
		else
			return FFN_FLATTEN_DEFAULT;
	}
	
	
	/**
	 * Setting flattening feed-forward network mode.
	 * @param ffnFlatten flattening feed-forward network mode.
	 * @return this classifier.
	 */
	VGG paramSetFFNFlatten(boolean ffnFlatten) {
		config.put(FFN_FLATTEN_FIELD, ffnFlatten);
		return this;
	}

	
	/**
	 * Checking co-weight mode.
	 * @return co-weight mode.
	 */
	boolean paramIsCoweight() {
		if (config.containsKey(COWEIGHT_FIELD))
			return config.getAsBoolean(COWEIGHT_FIELD);
		else
			return COWEIGHT_DEFAULT;
	}
	
	
	/**
	 * Setting co-weight mode.
	 * @param coweight co-weight mode.
	 * @return this VGG.
	 */
	VGG paramSetCoweight(boolean coweight) {
		config.put(COWEIGHT_FIELD, coweight);
		return this;
	}


	/**
	 * Checking pooling filter type.
	 * @return pooling filter type.
	 */
	PoolType paramGetPoolType() {
		if (config.containsKey(POOL_TYPE_FIELD))
			return FilterSpec.intToPoolType(config.getAsInt(POOL_TYPE_FIELD));
		else
			return POOL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting pooling filter type.
	 * @param poolType pooling filter type.
	 * @return this classifier.
	 */
	VGG paramSetPoolType(PoolType poolType) {
		config.put(POOL_TYPE_FIELD, FilterSpec.poolTypeToInt(poolType));
		return this;
	}

	
	/**
	 * Checking network filter type.
	 * @return network filter type.
	 */
	NetworkType paramGetNetworkType() {
		if (config.containsKey(NETWORK_TYPE_FIELD))
			return FilterSpec.intToNetworkType(config.getAsInt(NETWORK_TYPE_FIELD));
		else
			return NETWORK_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting network filter type.
	 * @param networkType network filter type.
	 * @return this classifier.
	 */
	VGG paramSetNetworkType(NetworkType networkType) {
		config.put(NETWORK_TYPE_FIELD, FilterSpec.networkTypeToInt(networkType));
		return this;
	}

	
	/**
	 * Getting weight type.
	 * @return weight type.
	 */
	net.ea.ann.mane.WeightSpec.Type paramGetWeightType() {
		if (config.containsKey(WEIGHT_TYPE_FIELD))
			return net.ea.ann.mane.WeightSpec.intToType(config.getAsInt(WEIGHT_TYPE_FIELD));
		else
			return WEIGHT_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting weight type.
	 * @param weightType weight type.
	 * @return this classifier.
	 */
	VGG paramSetWeightType(net.ea.ann.mane.WeightSpec.Type weightType) {
		config.put(WEIGHT_TYPE_FIELD, net.ea.ann.mane.WeightSpec.typeToInt(weightType));
		return this;
	}


}
