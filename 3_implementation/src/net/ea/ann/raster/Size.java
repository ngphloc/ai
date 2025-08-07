/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.Dimension;

/**
 * This class represents multi-dimension.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Size extends java.awt.Dimension {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Depth.
	 */
	public int depth = 0;
	
	
	/**
	 * Time.
	 */
	public int time = 0;

	
	/**
	 * Constructor with specified width, height, depth, and time.
	 * @param width specified width.
	 * @param height specified height.
	 * @param depth specified depth.
	 * @param time specified time.
	 */
	public Size(int width, int height, int depth, int time) {
		super(width, height);
		this.depth = depth;
		this.time = time;
	}
	
	
	/**
	 * Constructor with specified width, height, and depth.
	 * @param width specified width.
	 * @param height specified height.
	 * @param depth specified depth.
	 */
	public Size(int width, int height, int depth) {
		this(width, height, depth, 0);
	}

	
	/**
	 * Constructor with specified size.
	 * @param size specified size.
	 */
	public Size(Dimension size) {
		this(size.width, size.height, 0, 0);
	}
	
	
	/**
	 * Constructor with specified width and height.
	 * @param width specified width.
	 * @param height specified height.
	 */
	public Size(int width, int height) {
		this(width, height, 0, 0);
	}

	
	/**
	 * Default constructor.
	 */
	public Size() {
		this(0, 0, 0, 0);
	}

	
	/**
	 * Multiplying this size with factor.
	 * @param factor specified factor.
	 * @return new size multiplied with factor.
	 */
	public Size multiply(int factor) {
		return new Size(width*factor, height*factor, depth*factor, time*factor);
	}


	/**
	 * Dividing this size by factor.
	 * @param factor specified factor.
	 * @return new size multiplied with factor.
	 */
	public Size divide(int factor) {
		return factor != 0 ? new Size(width/factor, height/factor, depth/factor, time/factor) : null;
	}
	
	
	/**
	 * Getting size.
	 * @return length.
	 */
	public int length() {
		return width*height*depth*time;
	}


	@Override
	public boolean equals(Object obj) {
		if (obj == null)
			return false;
		else if (obj instanceof Size) {
			Size otherSize = (Size)obj;
			return this.width == otherSize.width && this.height == otherSize.height &&
				this.depth == otherSize.depth && this.time == otherSize.time;
		}
		else
			return super.equals(obj);
	}
	
	
	/**
	 * Getting zero element.
	 * @return zero element.
	 */
	public static Size zero() {
		return new Size(0, 0, 0, 0);
	}

	
	/**
	 * Getting unit element.
	 * @return unit element.
	 */
	public static Size unit() {
		return new Size(1, 1, 1, 1);
	}
	
	
}
