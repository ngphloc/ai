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
	 * Size of filter.
	 */
	public Size size = new Size(1, 1, 1, 1);
	
	
	/**
	 * This flag indicates whether this filter is associated with normal weight.
	 */
	public boolean coweight = false;
	
	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	public FilterSpec(Size size) {
		this.size = size;
	}
	
	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	public FilterSpec(Dimension size) {
		this(new Size(size.width, size.height, 1, 1));
	}
	

	/**
	 * Constructor with size.
	 * @param size size.
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
