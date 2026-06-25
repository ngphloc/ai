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
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.DropoutNetwork;
import net.ea.ann.mane.FilterSpec;
import net.ea.ann.mane.FilterSpec.KernelType;
import net.ea.ann.mane.FilterSpec.NetworkType;
import net.ea.ann.mane.FilterSpec.PoolType;
import net.ea.ann.mane.FilterSpec.Type;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.WeightSpec;
import net.ea.ann.raster.Size;
import net.ea.ann.transformer.TransformerBasic;
import net.hudup.core.parser.TextParserUtil;

/**
 * This class is an implementation of VGG blocks developed by Simonyan and Zisserman with support of matrix.
 * VGG should not be residual network because it is implemented here as a network bean.
 * 
 * @author Simonyan and Zisserman, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class VGG extends DropoutNetwork /*MatrixNetworkImpl*/ {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default base.
	 */
	private final static int BASE = ZOOMOUT_DEFAULT;
	
	
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
	 * Field for filter mode. If false, filtering is not applied.
	 */
	public final static String FILTER_MODE_FIELD = "mane_filter_mode";
	
	
	/**
	 * Default value for filter mode. If false, filtering is not applied.
	 */
	public final static boolean FILTER_MODE_DEFAULT = true;

	
	/**
	 * Field for ending pooling filter mode. If false, pooling filtering is not applied into the last layer of each block.
	 */
	public final static String FILTER_MODE_ENDPOOL_FIELD = "mane_filter_mode_endpool";
	
	
	/**
	 * Default value for ending pooling filter mode. If false, pooling filtering is not applied into the last layer of each block.
	 */
	public final static boolean FILTER_MODE_ENDPOOL_DEFAULT = true;

	
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
	 * Field for initial number of filters.
	 */
	public final static String FILTERS_NUMBER_INIT_FIELD = "vgg_filter_number_init";
	
	
	/**
	 * Default value for initial number of filters. It is also the initial depth of layer.
	 */
	public final static int FILTERS_NUMBER_INIT_DEFAULT = BASE*BASE;

	
	/**
	 * Field for maximum number of filters.
	 */
	public final static String FILTERS_NUMBER_MAX_FIELD = "vgg_filter_number_max";
	
	
	/**
	 * Default value for maximum number of filters. Zero value indicates unlimited number.
	 */
	public final static int FILTERS_NUMBER_MAX_DEFAULT = 0;

	
	/**
	 * Field for field to increase filter number. If false, filter number is not increase.
	 */
	public final static String FILTERS_NUMBER_INCREASE_FIELD = "vgg_filter_number_increase";
	
	
	/**
	 * Default value for field to increase filter number. If false, filter number is not increase.
	 */
	public final static boolean FILTERS_NUMBER_INCREASE_DEFAULT = true;

	
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
	 * Field for co-weight mode. If true, weight matrix and filter kernel co-exist in the same layer.
	 */
	public final static String COWEIGHT_FIELD = "vgg_coweight";
	
	
	/**
	 * Default value for co-weight mode. If true, weight matrix and filter kernel co-exist in the same layer.
	 */
	public final static boolean COWEIGHT_DEFAULT = FilterSpec.COWEIGHT;

	
	/**
	 * Field for weight type.
	 */
	public final static String FILTER_TYPE_FIELD = "mane_filter_type";
	
	
	/**
	 * Default value for weight type.
	 */
	public final static Type FILTER_TYPE_DEFAULT = Type.kernel;

	
	/**
	 * Field for filter kernel type.
	 */
	public final static String FILTER_KERNEL_TYPE_FIELD = "mane_filter_type_kernel";
	
	
	/**
	 * Default value for filter kernel type.
	 */
	public final static KernelType FILTER_KERNEL_TYPE_DEFAULT = KernelType.product;

	
	/**
	 * Field for filter pool type.
	 */
	public final static String FILTER_POOL_TYPE_FIELD = "mane_filter_type_pool";
	
	
	/**
	 * Default value for filter pool type.
	 */
	public final static PoolType FILTER_POOL_TYPE_DEFAULT = PoolType.max;

	
	/**
	 * Field for filter network type.
	 */
	public final static String FILTER_NETWORK_TYPE_FIELD = "mane_filter_type_network";

	
	/**
	 * Default value for filter network type.
	 */
	public final static NetworkType FILTER_NETWORK_TYPE_DEFAULT = NetworkType.basic;

	
	/**
	 * Field for weight type.
	 */
	public final static String WEIGHT_TYPE_FIELD = "mane_weight_type";
	
	
	/**
	 * Default value for weight type.
	 */
	public final static net.ea.ann.mane.WeightSpec.Type WEIGHT_TYPE_DEFAULT = net.ea.ann.mane.WeightSpec.Type.kernel;

	
	/**
	 * Field for weight kernel type.
	 */
	public final static String WEIGHT_KERNEL_TYPE_FIELD = "mane_weight_type_kernel";
	
	
	/**
	 * Default value for weight kernel type.
	 */
	public final static net.ea.ann.mane.WeightSpec.KernelType WEIGHT_KERNEL_TYPE_DEFAULT = net.ea.ann.mane.WeightSpec.KernelType.normal;

	
	/**
	 * Field for weight network type.
	 */
	public final static String WEIGHT_NETWORK_TYPE_FIELD = "mane_weight_type_network";
	
	
	/**
	 * Default value for weight network type.
	 */
	public final static net.ea.ann.mane.WeightSpec.NetworkType WEIGHT_NETWORK_TYPE_DEFAULT = net.ea.ann.mane.WeightSpec.NetworkType.basic;

	
	/**
	 * Field for length of filter network.
	 */
	public final static String SUB_NETWORK_LENGTH_FIELD = "mane_subnetwork_length";
	
	
	/**
	 * Default value for length of filter network.
	 */
	public final static int SUB_NETWORK_LENGTH_DEFAULT = 1;

	
	/**
	 * Setting configuration.
	 * @param config configuration.
	 */
	private static void config(NetworkConfig config) {
		config.put(MIDDLE_SIZE_FIELD, MIDDLE_SIZE_DEFAULT_TEXT);
		config.put(FILTER_MODE_FIELD, FILTER_MODE_DEFAULT);
		config.put(FILTER_MODE_ENDPOOL_FIELD, FILTER_MODE_ENDPOOL_DEFAULT);
		config.put(BLOCKS_NUMBER_FIELD, BLOCKS_NUMBER_DEFAULT);
		config.put(LAYERS_NUMBER_FIELD, LAYERS_NUMBER_DEFAULT);
		config.put(FILTERS_NUMBER_INIT_FIELD, FILTERS_NUMBER_INIT_DEFAULT);
		config.put(FILTERS_NUMBER_MAX_FIELD, FILTERS_NUMBER_MAX_DEFAULT);
		config.put(FILTERS_NUMBER_INCREASE_FIELD, FILTERS_NUMBER_INCREASE_DEFAULT);
		config.put(FILTER_SIZE_FIELD, FILTER_SIZE_DEFAULT);
		config.put(FFN_LENGTH_FIELD, FFN_LENGTH_DEFAULT);
		config.put(FFN_FLATTEN_FIELD, FFN_FLATTEN_DEFAULT);
		config.put(COWEIGHT_FIELD, COWEIGHT_DEFAULT);
		config.put(FILTER_TYPE_FIELD, FilterSpec.typeToInt(FILTER_TYPE_DEFAULT));
		config.put(FILTER_KERNEL_TYPE_FIELD, FilterSpec.kernelTypeToInt(FILTER_KERNEL_TYPE_DEFAULT));
		config.put(FILTER_POOL_TYPE_FIELD, FilterSpec.poolTypeToInt(FILTER_POOL_TYPE_DEFAULT));
		config.put(FILTER_NETWORK_TYPE_FIELD, FilterSpec.networkTypeToInt(FILTER_NETWORK_TYPE_DEFAULT));
		config.put(WEIGHT_TYPE_FIELD, WeightSpec.typeToInt(WEIGHT_TYPE_DEFAULT));
		config.put(WEIGHT_KERNEL_TYPE_FIELD, WeightSpec.kernelTypeToInt(WEIGHT_KERNEL_TYPE_DEFAULT));
		config.put(WEIGHT_NETWORK_TYPE_FIELD, WeightSpec.networkTypeToInt(WEIGHT_NETWORK_TYPE_DEFAULT));
		config.put(SUB_NETWORK_LENGTH_FIELD, SUB_NETWORK_LENGTH_DEFAULT);
	}
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public VGG(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config(this.config);
		
//		//Removing following lines after debugging.
//		paramSetVGGMiddleSize(new Size(32, 32));
//		paramSetFiltersNumberMax(1);
//		System.out.println("VGG: Removing following lines after debugging.");
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
	public VGG(int neuronChannel, Function activateRef) {this(neuronChannel, activateRef, null, null);}

	
	/**
	 * Constructor with neuron channel.
	 * As usual neuron channel is set to be 1 and raster channel is set to be 3 so that the first layer is split as stack of matrices.
	 * However please pay attention that if neuron channel is set as same as raster channel then the first layer is the singular matrix whose each element is a vector.
	 * Please see {@link MatrixUtil#toMatrix(Size, net.ea.ann.raster.Raster, int, int, boolean, net.ea.ann.core.value.Matrix)}. 
	 * @param neuronChannel neuron channel.
	 */
	public VGG(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	/**
	 * Calculating block size.
	 * @param inputSize input size.
	 * @param middleSize middle size.
	 * @param blocksNumber number of blocks.
	 * @param filtersNumberInitPerLayer initial number of filters per layer (initial) which can be zero for automatic calculation.
	 * It is also the initial depth of layer.
	 * @param filterNumberMax maximum number of filters.
	 * @param increaseFiltersNumber flag to increase filter number.
	 * @return true if initialization is successful.
	 */
	static List<Size> calcBlockSizes(Size inputSize, Size middleSize, int blocksNumber, int filtersNumberInitPerLayer, int filterNumberMax, boolean increaseFiltersNumber) {
		if (inputSize == null) return Util.newList(0);
		int base = Math.max(BASE, 1);
		
		int r = Math.min(inputSize.width/middleSize.width, inputSize.height/middleSize.height);
		if (r < 1)
			middleSize = new Size(inputSize.width, inputSize.height);
		else
			middleSize = new Size(inputSize.width/r, inputSize.height/r);
		int[][] numbers = MatrixNetworkInitializer.constructHiddenOutputNeuronNumbers(inputSize, middleSize, base, base, blocksNumber);
		if (numbers == null) return Util.newList(0);
		
		int factor = 2; //The factor 2 implies the basic 2-dimension matrix is decreased base*base each time.
		filtersNumberInitPerLayer = filtersNumberInitPerLayer < 1 ? (int)Math.pow(base, factor) : filtersNumberInitPerLayer;
		
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
		int power = 0;
		for (int i = 0; i < heights.length; i++) {
			int filterNumber = increaseFiltersNumber ? (int)(filtersNumberInitPerLayer*Math.pow(base, factor*power)) : filtersNumberInitPerLayer;
			if (filterNumberMax > 0) filterNumber = Math.min(filterNumber, filterNumberMax);
			if (blockSizes.size() == 0) {
				blockSizes.add(new Size(widths[i], heights[i], filterNumber));
				power++;
				continue;
			}
			
			int prevWidth = blockSizes.get(blockSizes.size()-1).width;
			int prevHeight = blockSizes.get(blockSizes.size()-1).height;
			if (widths[i] != prevWidth || heights[i] != prevHeight) {
				blockSizes.add(new Size(widths[i], heights[i], filterNumber));
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
		List<Size> blockSizes = calcBlockSizes(inputSize, middleSize, paramGetBlocksNumber(), paramGetFiltersNumberInit(), paramGetFiltersNumberMax(), paramIsFiltersNumberIncrease());
		if (blockSizes.size() == 0) return false;
		
		int base = Math.max(BASE, 1);
		int layersNumberPerBlock = paramGetLayersNumber();
		int filterSize = paramGetFilterSize();
		int ffnLength = paramGetFFNLength();

		int rasterChannel = paramGetRasterChannel();
		boolean flatten = MatrixUtil.isFlatten(inputSize.depth, this.neuronChannel, rasterChannel); //inputSize.depth is actually raster depth.
		LayerSpec layerSpec0 = null;
		if (flatten)
			layerSpec0 = new MatrixLayerAbstract.LayerSpec(new Size(inputSize.width, inputSize.height, rasterChannel, 1));
		else
			layerSpec0 = new MatrixLayerAbstract.LayerSpec(new Size(inputSize.width, inputSize.height,
				inputSize.depth < 1 ? 1 : inputSize.depth,
				1/*inputSize.time < 1 ? 1 : inputSize.time*/)); //In current version, time is 1, which means that the model supports until 3-dimension layers.
		List<LayerSpec> layerSpecs = Util.newList(0);
		layerSpecs.add(layerSpec0);
		for (int i = 0; i < blockSizes.size(); i++) {
			Size blockSize = blockSizes.get(i);
			for (int j = 0; j < layersNumberPerBlock; j++) {
				LayerSpec layerSpec = new LayerSpec(new Size(blockSize.width, blockSize.height, blockSize.depth, 1));
				if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
				
				if (paramIsFilterMode()) {
					Type filterType = paramGetFilterType();
					layerSpec.filterSpec = null;
					if (filterType == Type.kernel) {
						layerSpec.filterSpec = new FilterSpec(filterSize, filterSize, filterType);
						layerSpec.filterSpec.kernelType = paramGetFilterKernelType();
					}
					else if (filterType == Type.pool) {
						if (layerSpec.prevSize != null && layerSpec.prevSize.depth == layerSpec.size.depth) {
							layerSpec.filterSpec = new FilterSpec(filterSize, filterSize, filterType);
							layerSpec.filterSpec.poolType = paramGetFilterPoolType();
						}
					}
					else if (filterType == Type.network) {
						layerSpec.filterSpec = new FilterSpec(filterSize, filterSize, filterType);
						layerSpec.filterSpec.networkType = paramGetFilterNetworkType();
						layerSpec.filterSpec.kernelType = paramGetFilterKernelType(); //This is kernel type of network filter.
						layerSpec.fifthLength = Math.max(paramGetSubNetworkLength(), 1); //The fifth length of layer specification is now the depth (length) of network filter.
					}
					
					if (layerSpec.filterSpec != null) {
						layerSpec.filterSpec.moveStride = false;
						layerSpec.filterSpec.coweight = paramIsCoweight();
					}
				} //End setting filter specification.
				
				net.ea.ann.mane.WeightSpec.Type weightType = paramGetWeightType();
				layerSpec.weightSpec = null;
				if (weightType == net.ea.ann.mane.WeightSpec.Type.kernel) {
					layerSpec.weightSpec = new WeightSpec(weightType);
					layerSpec.weightSpec.kernelType = paramGetWeightKernelType();
				}
				else if (weightType == net.ea.ann.mane.WeightSpec.Type.network) {
					if (layerSpec.filterSpec != null && layerSpec.filterSpec.coweight)
						layerSpec.weightSpec = new WeightSpec(weightType);
					else if (layerSpec.filterSpec == null && layerSpec.prevSize != null && layerSpec.prevSize.depth == layerSpec.size.depth)
						layerSpec.weightSpec = new WeightSpec(weightType);
					
					if (layerSpec.weightSpec != null) {
						layerSpec.weightSpec.networkType = paramGetWeightNetworkType();
						layerSpec.weightSpec.kernelType = paramGetWeightKernelType();
						layerSpec.fifthLength = paramGetSubNetworkLength(); //The fifth length of layer specification is now the depth (length) of network weight.
					}
				} //End setting weight specification.
				
				assert (layerSpec.filterSpec != null || layerSpec.weightSpec != null); //This assertion is not important, which can be removed.
				layerSpecs.add(layerSpec);
			}
			
			if (i < blockSizes.size()-1 && paramIsFilterEndPoolMode()) {
				Size poolSize = new Size(blockSizes.get(i+1).width, blockSizes.get(i+1).height, blockSize.depth, 1);
				LayerSpec layerSpec = new LayerSpec(poolSize);
				if (layerSpecs.size() > 0) layerSpec.prevSize = layerSpecs.get(layerSpecs.size()-1).size;
				
				if (layerSpec.prevSize != null && layerSpec.prevSize.depth == layerSpec.size.depth) {
					layerSpec.filterSpec = new FilterSpec(base, base, Type.pool);
					layerSpec.filterSpec.poolType = paramGetFilterPoolType();
					layerSpec.filterSpec.moveStride = true; //It means zooming out to be smaller.
					layerSpec.filterSpec.coweight = false; //This code line is not necessary because weight specification is not specified but confirming that by default pooling filter is not associated with weight matrix because of reduced layer size (width and height).
					layerSpecs.add(layerSpec);
				}
			}
		}
		
		//Adding more layers to layer specifications.
		addMoreLayers(layerSpecs);
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
		
		//Adding more FFN layers to layer specifications.
		addMoreFFNLayers(layerSpecs);
		return initialize(layerSpecs.toArray(new LayerSpec[] {}), false);
	}
	
	
	/**
	 * Initializing VGG model.
	 * @param inputSize input size.
	 * @param outputSize output size which can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size inputSize, Size outputSize) {return initialize(inputSize, outputSize, null);}
	
	
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
	 * Adding more layers to layer specifications.
	 * @param layerSpecs layer specifications.
	 */
	protected void addMoreLayers(List<LayerSpec> layerSpecs) {}
	
	
	/**
	 * Adding more FFN layers to layer specifications.
	 * @param layerSpecs layer specifications.
	 */
	protected void addMoreFFNLayers(List<LayerSpec> layerSpecs) {}
	
	
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
	 * Checking filter mode. If false, filtering is not applied.
	 * @return filter mode.
	 */
	boolean paramIsFilterMode() {
		if (config.containsKey(FILTER_MODE_FIELD))
			return config.getAsBoolean(FILTER_MODE_FIELD);
		else
			return FILTER_MODE_DEFAULT;
	}
	
	
	/**
	 * Setting filter mode.
	 * @param filterMode filter mode. If false, filtering is not applied.
	 * @return this VGG.
	 */
	VGG paramSetFilterMode(boolean filterMode) {
		config.put(FILTER_MODE_FIELD, filterMode);
		return this;
	}

	
	/**
	 * Checking ending pooling filter mode. If false, pooling filtering is not applied into the last layer of each block.
	 * @return ending pooling filter mode.
	 */
	boolean paramIsFilterEndPoolMode() {
		if (config.containsKey(FILTER_MODE_ENDPOOL_FIELD))
			return config.getAsBoolean(FILTER_MODE_ENDPOOL_FIELD);
		else
			return FILTER_MODE_ENDPOOL_DEFAULT;
	}

	
	/**
	 * Setting ending pooling filter mode.
	 * @param filterEndPoolMode ending pooling filter mode. If false, pooling filtering is not applied into the last layer of each block.
	 * @return this VGG.
	 */
	VGG paramSetFilterEndPoolMode(boolean filterEndPoolMode) {
		config.put(FILTER_MODE_ENDPOOL_FIELD, filterEndPoolMode);
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
	 * Getting initial number of filters per layer which is also the initial depth of layer.
	 * @return initial number of filters per layer.
	 */
	public int paramGetFiltersNumberInit() {
		if (config.containsKey(FILTERS_NUMBER_INIT_FIELD))
			return config.getAsInt(FILTERS_NUMBER_INIT_FIELD);
		else
			return FILTERS_NUMBER_INIT_DEFAULT;
	}
	
	
	/**
	 * Setting initial number of filters per layer.
	 * @param filtersNumberInit initial number of filters per layer which is also the initial depth of layer.
	 * @return this VGG.
	 */
	public VGG paramSetFiltersNumberInit(int filtersNumberInit) {
		filtersNumberInit = filtersNumberInit < 0 ? FILTERS_NUMBER_INIT_DEFAULT : filtersNumberInit;
		config.put(FILTERS_NUMBER_INIT_FIELD, filtersNumberInit);
		return this;
	}
	
	
	/**
	 * Getting maximum number of filters per layer (also layer depth).
	 * @return maximum number of filters per layer.
	 */
	int paramGetFiltersNumberMax() {
		if (config.containsKey(FILTERS_NUMBER_MAX_FIELD))
			return config.getAsInt(FILTERS_NUMBER_MAX_FIELD);
		else
			return FILTERS_NUMBER_MAX_DEFAULT;
	}
	
	
	/**
	 * Setting maximum number of filters per layer (also layer depth).
	 * @param maxFiltersNumber maximum number of filters per layer.
	 * @return this VGG.
	 */
	VGG paramSetFiltersNumberMax(int maxFiltersNumber) {
		maxFiltersNumber = maxFiltersNumber < 0 ? FILTERS_NUMBER_MAX_DEFAULT : maxFiltersNumber;
		config.put(FILTERS_NUMBER_MAX_FIELD, maxFiltersNumber);
		return this;
	}
	
	
	/**
	 * Checking flag to increase filter number. If false, filter number is not increase.
	 * @return flag to increase filter number.
	 */
	boolean paramIsFiltersNumberIncrease() {
		if (config.containsKey(FILTERS_NUMBER_INCREASE_FIELD))
			return config.getAsBoolean(FILTERS_NUMBER_INCREASE_FIELD);
		else
			return FILTERS_NUMBER_INCREASE_DEFAULT;
	}
	
	
	/**
	 * Setting flag to increase filter number.
	 * @param increase flag to increase filter number. If false, filter number is not increase.
	 * @return this VGG.
	 */
	VGG paramSetFiltersNumberIncrease(boolean increase) {
		config.put(FILTERS_NUMBER_INCREASE_FIELD, increase);
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
	 * Checking co-weight mode. If true, weight matrix and filter kernel co-exist in the same layer.
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
	 * @param coweight co-weight mode. If true, weight matrix and filter kernel co-exist in the same layer.
	 * @return this VGG.
	 */
	VGG paramSetCoweight(boolean coweight) {
		config.put(COWEIGHT_FIELD, coweight);
		return this;
	}


	/**
	 * Getting filter type.
	 * @return filter type.
	 */
	Type paramGetFilterType() {
		if (config.containsKey(FILTER_TYPE_FIELD))
			return FilterSpec.intToType(config.getAsInt(FILTER_TYPE_FIELD));
		else
			return FILTER_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting filter type.
	 * @param filterType filter type.
	 * @return this model.
	 */
	VGG paramSetFilterType(Type filterType) {
		config.put(FILTER_TYPE_FIELD, FilterSpec.typeToInt(filterType));
		return this;
	}

	
	/**
	 * Checking filter kernel type.
	 * @return filter kernel type.
	 */
	KernelType paramGetFilterKernelType() {
		if (config.containsKey(FILTER_KERNEL_TYPE_FIELD))
			return FilterSpec.intToKernelType(config.getAsInt(FILTER_KERNEL_TYPE_FIELD));
		else
			return FILTER_KERNEL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting filter kernel type.
	 * @param kernelType filter kernel type.
	 * @return this classifier.
	 */
	VGG paramSetFilterKernelType(KernelType kernelType) {
		config.put(FILTER_KERNEL_TYPE_FIELD, FilterSpec.kernelTypeToInt(kernelType));
		return this;
	}

	
	/**
	 * Checking filter pooling type.
	 * @return filter pooling type.
	 */
	PoolType paramGetFilterPoolType() {
		if (config.containsKey(FILTER_POOL_TYPE_FIELD))
			return FilterSpec.intToPoolType(config.getAsInt(FILTER_POOL_TYPE_FIELD));
		else
			return FILTER_POOL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting filter pooling type.
	 * @param poolType filter pooling type.
	 * @return this classifier.
	 */
	VGG paramSetFilterPoolType(PoolType poolType) {
		config.put(FILTER_POOL_TYPE_FIELD, FilterSpec.poolTypeToInt(poolType));
		return this;
	}

	
	/**
	 * Checking filter network type.
	 * @return filter network type.
	 */
	NetworkType paramGetFilterNetworkType() {
		if (config.containsKey(FILTER_NETWORK_TYPE_FIELD))
			return FilterSpec.intToNetworkType(config.getAsInt(FILTER_NETWORK_TYPE_FIELD));
		else
			return FILTER_NETWORK_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting filter network type.
	 * @param networkType filter network type.
	 * @return this classifier.
	 */
	VGG paramSetFilterNetworkType(NetworkType networkType) {
		config.put(FILTER_NETWORK_TYPE_FIELD, FilterSpec.networkTypeToInt(networkType));
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
	 * @return this model.
	 */
	VGG paramSetWeightType(net.ea.ann.mane.WeightSpec.Type weightType) {
		config.put(WEIGHT_TYPE_FIELD, net.ea.ann.mane.WeightSpec.typeToInt(weightType));
		return this;
	}


	/**
	 * Getting length of sub-network.
	 * @return length of sub-network.
	 */
	int paramGetSubNetworkLength() {
		if (config.containsKey(SUB_NETWORK_LENGTH_FIELD))
			return config.getAsInt(SUB_NETWORK_LENGTH_FIELD);
		else
			return SUB_NETWORK_LENGTH_DEFAULT;
	}
	
	
	/**
	 * Checking weight kernel type.
	 * @return weight kernel type.
	 */
	net.ea.ann.mane.WeightSpec.KernelType paramGetWeightKernelType() {
		if (config.containsKey(WEIGHT_KERNEL_TYPE_FIELD))
			return WeightSpec.intToKernelType(config.getAsInt(WEIGHT_KERNEL_TYPE_FIELD));
		else
			return WEIGHT_KERNEL_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting weight kernel type.
	 * @param kernelType weight kernel type.
	 * @return this model.
	 */
	VGG paramSetWeightKernelType(net.ea.ann.mane.WeightSpec.KernelType kernelType) {
		config.put(WEIGHT_KERNEL_TYPE_FIELD, WeightSpec.kernelTypeToInt(kernelType));
		return this;
	}

	
	/**
	 * Checking weight network type.
	 * @return weight network type.
	 */
	net.ea.ann.mane.WeightSpec.NetworkType paramGetWeightNetworkType() {
		if (config.containsKey(WEIGHT_NETWORK_TYPE_FIELD))
			return WeightSpec.intToNetworkType(config.getAsInt(WEIGHT_NETWORK_TYPE_FIELD));
		else
			return WEIGHT_NETWORK_TYPE_DEFAULT;
	}
	
	
	/**
	 * Setting weight network type.
	 * @param networkType weight network type.
	 * @return this model.
	 */
	VGG paramSetWeightNetworkType(net.ea.ann.mane.WeightSpec.NetworkType networkType) {
		config.put(WEIGHT_NETWORK_TYPE_FIELD, WeightSpec.networkTypeToInt(networkType));
		return this;
	}

	
	/**
	 * Setting length of sub-network.
	 * @param length length of sub-network.
	 * @return this VGG.
	 */
	VGG paramSetSubNetworkLength(int length) {
		length = length < 1 ? SUB_NETWORK_LENGTH_DEFAULT : length;
		config.put(SUB_NETWORK_LENGTH_FIELD, length);
		return this;
	}


}
