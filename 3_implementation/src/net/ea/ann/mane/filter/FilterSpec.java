/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.awt.Dimension;
import java.io.Serializable;

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
	 * Filter type.
	 */
	public Type type = Type.kernel;
	
	
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


}
