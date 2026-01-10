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
import net.ea.ann.mane.filter.KernelFilterMax;
import net.ea.ann.mane.filter.KernelFilterProduct;
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
	public int rows() {
		return height();
	}
	
	
	/**
	 * Getting columns.
	 * @return columns.
	 */
	public int columns() {
		return width();
	}


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
	 * Creating default filter.
	 * @param filterSize filter size.
	 * @param hint matrix hint.
	 * @return default filter.
	 */
	public static Filter newFilter(Size filterSize, FilterSpec filterSpec, NeuronValue hint) {
		if (filterSize == null || filterSize.width <= 0 || filterSize.height <= 0) return null;
		double factor = 1.0 / (filterSize.width*filterSize.height);
		Filter filter = null;
		switch (filterSpec.type) {
			case kernel:
				switch (filterSpec.kernelType) {
				case product:
					filter = KernelFilterProduct.create(factor, filterSize, hint);
					break;
				case product_max:
					filter = KernelFilterMax.create(factor, filterSize, hint);
					break;
				default:
					filter = KernelFilterProduct.create(factor, filterSize, hint);
					break;
				}
				break;
			case pool:
				Size adjustedSize = new Size(filterSize.width, filterSize.height, filterSize.time, 1);
				switch (filterSpec.poolType) {
				case max:
					filter = PoolFilterMax.create(adjustedSize);
					break;
				case average:
					filter = PoolFilterAverage.create(adjustedSize);
					break;
				default:
					filter = PoolFilterMax.create(adjustedSize);
					break;
				}
				break;
			default:
				break;
		}
		filter.setMoveStride(filterSpec.moveStride);
		return filter;
	}

	
}
