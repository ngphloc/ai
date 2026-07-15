/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;
import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.filter.FilterNetworkImpl;
import net.ea.ann.mane.filter.GAP;
import net.ea.ann.mane.filter.KernelFilterMax;
import net.ea.ann.mane.filter.KernelFilterProduct;
import net.ea.ann.mane.filter.MicroFilter;
import net.ea.ann.mane.filter.NullFilter;
import net.ea.ann.mane.filter.PoolFilterAverage;
import net.ea.ann.mane.filter.PoolFilterMax;
import net.ea.ann.raster.Size;

/**
 * This class represents filter specification.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FilterSpec implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default co-weight mode.
	 */
	public final static boolean COWEIGHT = false;
	
	
	/**
	 * This enum specifies filter type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum Type {
		
		/**
		 * Kernel filter.
		 */
		kernel,
		
		/**
		 * Pooling filter.
		 */
		pool,

		/**
		 * Network-based filter.
		 */
		network,
	}
	
	
	/**
	 * This enum represents kernel filter type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum KernelType {
		
		/**
		 * Product filter type.
		 */
		product,
		
		/**
		 * Max filter type.
		 */
		product_max,
		
		/**
		 * Micro filter type.
		 */
		micro,
		
		/**
		 * Null filter.
		 */
		nil,
	}

	
	/**
	 * This enum represents pooling filter type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum PoolType {
		
		/**
		 * Max-pooling filter type.
		 */
		max,
		
		/**
		 * Average pooling filter type.
		 */
		average,
		
		/**
		 * Global Pooling Average (GAP) filer type.
		 */
		gap,
		
	}
	
	
	/**
	 * This enum represents pooling filter type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum NetworkType {
		
		/**
		 * Basic type.
		 */
		basic,
		
	}

	
	/**
	 * Filter type.
	 */
	public Type type = Type.kernel;
	
	
	/**
	 * Kernel type.
	 */
	public KernelType kernelType = KernelType.product;
	
	
	/**
	 * Pooling type.
	 */
	public PoolType poolType = PoolType.max;
	
	
	/**
	 * Network-based type.
	 */
	public NetworkType networkType = NetworkType.basic;

	
	/**
	 * Size of filter.
	 */
	public Size size = new Size(1, 1, 1, 1);
	
	
	/**
	 * This flag indicates whether this filter is associated with weight.
	 */
	public boolean coweight = COWEIGHT;
	
	
	/**
	 * Flag to move by stride.
	 */
	public boolean moveStride = Filter.MOVE_STRIDE;
	
	
	/**
	 * Constructor with size and type.
	 * @param size size.
	 * @param type type.
	 */
	public FilterSpec(Size size, Type type) {
		this.size = size;
		this.type = type;
	}

	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	public FilterSpec(Size size) {
		this(size, Type.kernel);
	}
	
	
	/**
	 * Constructor with type.
	 * @param type type.
	 */
	public FilterSpec(Type type) {
		this(Size.unit(), type);
	}

	
	/**
	 * Constructor with size and type.
	 * @param size size.
	 * @param type type
	 */
	public FilterSpec(Dimension size, Type type) {
		this(new Size(size.width, size.height, 1, 1), type);
	}

	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	public FilterSpec(Dimension size) {
		this(new Size(size.width, size.height, 1, 1));
	}
	

	/**
	 * Constructor with width, height, and type.
	 * @param width width.
	 * @param height height.
	 * @param type type.
	 */
	public FilterSpec(int width, int height, Type type) {
		this(new Dimension(width, height), type);
	}

	
	/**
	 * Constructor with width and height.
	 * @param width width.
	 * @param height height.
	 */
	public FilterSpec(int width, int height) {
		this(new Dimension(width, height));
	}

	
	/**
	 * Getting width.
	 * @return width.
	 */
	public int width() {return size.width;}


	/**
	 * Getting height.
	 * @return weight.
	 */
	public int height() {return size.height;}


	/**
	 * Getting rows.
	 * @return rows.
	 */
	public int rows() {return height();}
	
	
	/**
	 * Getting columns.
	 * @return columns.
	 */
	public int columns() {return width();}


	/**
	 * Converting type to integer number.
	 * @param type type.
	 * @return integer number.
	 */
	public static int typeToInt(Type type) {
		return type.ordinal();
	}
	

	/**
	 * Converting integer number to type.
	 * @param typeOrdinal integer number.
	 * @return type.
	 */
	public static Type intToType(int typeOrdinal) {
		Type type = Type.kernel;
		switch (typeOrdinal) {
		case 0:
			type = Type.kernel;
			break;
		case 1:
			type = Type.pool;
			break;
		case 2:
			type = Type.network;
			break;
		default:
			type = Type.kernel;
			break;
		}
		return type;
	}
	
	
	/**
	 * Converting string to type.
	 * @param typeText string.
	 * @return type.
	 */
	public static Type stringToType(String typeText) {
		return Type.valueOf(typeText);
	}
	
	
	/**
	 * Converting kernel type to integer number.
	 * @param kernelType kernel type.
	 * @return integer number.
	 */
	public static int kernelTypeToInt(KernelType kernelType) {
		return kernelType.ordinal();
	}

	
	/**
	 * Converting integer number to kernel type.
	 * @param kernelTypeOrdinal integer number.
	 * @return kernel type.
	 */
	public static KernelType intToKernelType(int kernelTypeOrdinal) {
		KernelType kernelType = KernelType.product;
		switch (kernelTypeOrdinal) {
		case 0:
			kernelType = KernelType.product;
			break;
		case 1:
			kernelType = KernelType.product_max;
			break;
		case 2:
			kernelType = KernelType.micro;
			break;
		case 3:
			kernelType = KernelType.nil;
			break;
		default:
			kernelType = KernelType.product;
			break;
		}
		return kernelType;
	}
	
	
	/**
	 * Converting string to kernel type.
	 * @param kernelTypeText string.
	 * @return kernel type.
	 */
	public static KernelType stringToKernelType(String kernelTypeText) {
		return KernelType.valueOf(kernelTypeText);
	}

	
	/**
	 * Converting pooling type to integer number.
	 * @param poolType pooling type.
	 * @return integer number.
	 */
	public static int poolTypeToInt(PoolType poolType) {
		return poolType.ordinal();
	}
	

	/**
	 * Converting integer number to pooling type.
	 * @param poolTypeOrdinal integer number.
	 * @return type.
	 */
	public static PoolType intToPoolType(int poolTypeOrdinal) {
		PoolType poolType = PoolType.max;
		switch (poolTypeOrdinal) {
		case 0:
			poolType = PoolType.max;
			break;
		case 1:
			poolType = PoolType.average;
			break;
		case 2:
			poolType = PoolType.gap;
			break;
		default:
			poolType = PoolType.max;
			break;
		}
		return poolType;
	}
	
	
	/**
	 * Converting string to pooling type.
	 * @param poolTypeText string.
	 * @return pooling type.
	 */
	public static PoolType stringToPoolType(String poolTypeText) {
		return PoolType.valueOf(poolTypeText);
	}
	
	
	/**
	 * Converting network type to integer number.
	 * @param networkType network type.
	 * @return integer number.
	 */
	public static int networkTypeToInt(NetworkType networkType) {
		return networkType.ordinal();
	}
	

	/**
	 * Converting integer number to network type.
	 * @param networkTypeOrdinal integer number.
	 * @return type.
	 */
	public static NetworkType intToNetworkType(int networkTypeOrdinal) {
		NetworkType networkType = NetworkType.basic;
		switch (networkTypeOrdinal) {
		case 0:
			networkType = NetworkType.basic;
			break;
		default:
			networkType = NetworkType.basic;
			break;
		}
		return networkType;
	}
	
	
	/**
	 * Converting string to network type.
	 * @param networkTypeText string.
	 * @return network type.
	 */
	public static NetworkType stringToNetworkType(String networkTypeText) {
		return NetworkType.valueOf(networkTypeText);
	}
	
	
	/**
	 * Creating filter.
	 * @param filterSize filter size in which width and height are size of filter itself
	 * whereas depth is the number of previous matrices (length of matrix stack in previous layer) and
	 * time is the number of current matrices (length of matrix stack in current layer).<br/>
	 * For pooling filter, depth is the number of current matrices (length of matrix stack in current layer) and time is 1 because
	 * all previous matrices needs only one pooling filter.<br/>
	 * For network filter, the specification is slight different, in which width and height are size of filter itself.
	 * @param hint value hint.
	 * @param layerSpec layer specification which can be null.
	 * @param neuronChannel neuron channel which is only applied to network filter.
	 * @return filter.
	 */
	public static Filter newFilter(Size filterSize, NeuronValue hint, LayerSpec layerSpec, int neuronChannel) {
		if (filterSize == null || filterSize.width <= 0 || filterSize.height <= 0) return null;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		hint = hint != null ? hint : NeuronValueCreator.newNeuronValue(neuronChannel);

		double factor = 1.0 / (double)(filterSize.width*filterSize.height);
		Filter filter = null;
		FilterSpec filterSpec = layerSpec != null ? layerSpec.filterSpec : new FilterSpec(filterSize);
		switch (filterSpec.type) {
			case kernel:
				switch (filterSpec.kernelType) {
				case product:
					filter = KernelFilterProduct.create(factor, filterSize, hint);
					break;
				case product_max:
					filter = KernelFilterMax.create(factor, filterSize, hint);
					break;
				case micro:
					filterSize = new Size(layerSpec.size.width, layerSpec.size.height, filterSize.depth, filterSize.time);
					filter = MicroFilter.create(factor, filterSize, hint);
					break;
				case nil:
					filter = new NullFilter();
					break;
				default:
					filter = KernelFilterProduct.create(factor, filterSize, hint);
					break;
				}
				break;
			case pool:
				if ( (filterSpec.poolType != PoolType.gap) && (filterSize.depth != filterSize.time || filterSize.time <= 0) ) throw new IllegalArgumentException();
				Size adjustedSize = new Size(filterSize.width, filterSize.height, filterSize.time, 1);
				switch (filterSpec.poolType) {
				case max:
					filter = PoolFilterMax.create(adjustedSize);
					break;
				case average:
					filter = PoolFilterAverage.create(adjustedSize);
					break;
				case gap:
					filter = new GAP();
					break;
				default:
					filter = PoolFilterMax.create(adjustedSize);
					break;
				}
				break;
			case network:
				if (layerSpec == null) return null;
				KernelType kernelType = layerSpec.filterSpec != null ? layerSpec.filterSpec.kernelType : KernelType.product;
				switch (filterSpec.networkType) {
				case basic:
					filter = FilterNetworkImpl.create(layerSpec.prevSize, layerSpec.size, Math.max(layerSpec.fifthLength, 1), filterSize, kernelType, neuronChannel);
					break;
				default:
					filter = FilterNetworkImpl.create(layerSpec.prevSize, layerSpec.size, Math.max(layerSpec.fifthLength, 1), filterSize, kernelType, neuronChannel);
					break;
				}
				break;
			default:
				filter = KernelFilterProduct.create(factor, filterSize, hint);
				break;
		}
		
		filter.setMoveStride(filterSpec.moveStride);
		return filter;
	}

	
	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param hint value hint.
	 * @param layerSpec layer specification.
	 * @return filter.
	 */
	static Filter newFilter(Size filterSize, NeuronValue hint, LayerSpec layerSpec) {
		return newFilter(filterSize, hint, layerSpec, 0);
	}
	
	
	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param layerSpec layer specification.
	 * @param neuronChannel neuron channel which is only applied to network filter.
	 * @return filter.
	 */
	static Filter newFilter(Size filterSize, LayerSpec layerSpec, int neuronChannel) {
		return newFilter(filterSize, null, layerSpec, neuronChannel);
	}
	
	
	/**
	 * Creating default filter which is product filter as usual.
	 * @param filterSize filter size.
	 * @param hint value hint.
	 * @return default filter which is product filter as usual.
	 */
	static Filter newFilter(Size filterSize, NeuronValue hint) {
		return newFilter(filterSize, hint, (LayerSpec)null, 0);
	}
	
	
	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param neuronChannel neuron channel which is only applied to network filter.
	 * @return filter.
	 */
	static Filter newFilter(Size filterSize, int neuronChannel) {
		return newFilter(filterSize, null, (LayerSpec)null, neuronChannel);
	}

	
	/**
	 * Creating null filter.
	 * @return null filter.
	 */
	static Filter newFilter() {return new NullFilter();}
	
	
	/**
	 * Creating filter.
	 * @param filterSize filter size in which width and height are size of filter itself
	 * whereas depth is the number of previous matrices (length of matrix stack in previous layer) and
	 * time is the number of current matrices (length of matrix stack in current layer).<br/>
	 * For pooling filter, depth is the number of current matrices (length of matrix stack in current layer) and time is 1 because
	 * all previous matrices needs only one pooling filter.
	 * @param hint value hint.
	 * @param filterSpec filter specification which can be null.
	 * @param neuronChannel neuron channel which is only applied to network filter.
	 * @return filter.
	 */
	public static Filter newFilter(Size filterSize, NeuronValue hint, FilterSpec filterSpec, int neuronChannel) {
		if (filterSize == null || filterSize.width <= 0 || filterSize.height <= 0) return null;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		hint = hint != null ? hint : NeuronValueCreator.newNeuronValue(neuronChannel);

		double factor = 1.0 / (filterSize.width*filterSize.height);
		Filter filter = null;
		filterSpec = filterSpec != null ? filterSpec : new FilterSpec(filterSize);
		switch (filterSpec.type) {
			case kernel:
				switch (filterSpec.kernelType) {
				case product:
					filter = KernelFilterProduct.create(factor, filterSize, hint);
					break;
				case product_max:
					filter = KernelFilterMax.create(factor, filterSize, hint);
					break;
				case micro:
					filterSize = new Size(filterSpec.size.width, filterSpec.size.height, filterSize.depth, filterSize.time);
					filter = MicroFilter.create(factor, filterSize, hint);
					throw new RuntimeException("Not validation yet");
				case nil:
					filter = new NullFilter();
					break;
				default:
					filter = KernelFilterProduct.create(factor, filterSize, hint);
					break;
				}
				break;
			case pool:
				if ( (filterSpec.poolType != PoolType.gap) && (filterSize.depth != filterSize.time || filterSize.time <= 0) ) throw new IllegalArgumentException();
				Size adjustedSize = new Size(filterSize.width, filterSize.height, filterSize.time, 1);
				switch (filterSpec.poolType) {
				case max:
					filter = PoolFilterMax.create(adjustedSize);
					break;
				case average:
					filter = PoolFilterAverage.create(adjustedSize);
					break;
				case gap:
					filter = new GAP();
					break;
				default:
					filter = PoolFilterMax.create(adjustedSize);
					break;
				}
				break;
			case network:
				throw new IllegalArgumentException();
			default:
				filter = KernelFilterProduct.create(factor, filterSize, hint);
				break;
		}
		
		filter.setMoveStride(filterSpec.moveStride);
		return filter;
	}


	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param hint value hint.
	 * @param filterSpec filter specification which can be null.
	 * @return filter.
	 */
	static Filter newFilter(Size filterSize, NeuronValue hint, FilterSpec filterSpec) {
		return newFilter(filterSize, hint, filterSpec, 0);
	}
	
	
	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param filterSpec filter specification which can be null.
	 * @param neuronChannel neuron channel which is only applied to network filter.
	 * @return filter.
	 */
	static Filter newFilter(Size filterSize, FilterSpec filterSpec, int neuronChannel) {
		return newFilter(filterSize, null, filterSpec, neuronChannel);
	}
	
	
}
