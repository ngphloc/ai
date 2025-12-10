/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

/**
 * This class represents a pooling filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class PoolFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Kernel width.
	 */
	protected int width = 1;
	
	
	/**
	 * Kernel height.
	 */
	protected int height = 1;

	
	/**
	 * Constructor with kernel width and height.
	 * @param width kernel width.
	 * @param height kernel height.
	 */
	protected PoolFilter(int width, int height) {
		super();
		if (width < 1 || height < 1) throw new IllegalArgumentException();
		this.width = width;
		this.height = height;
	}

	
	@Override
	public int width() {
		return width;
	}


	@Override
	public int height() {
		return height;
	}
	
	
}
