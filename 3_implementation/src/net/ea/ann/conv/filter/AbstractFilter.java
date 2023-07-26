/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

/**
 * This class is an abstract implementation of filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractFilter implements Filter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to slide according to block when filtering.
	 */
	protected boolean blockSlide = true;
	
	
	/**
	 * Default constructor.
	 */
	protected AbstractFilter() {
		super();
	}


	@Override
	public boolean isBlockSlide() {
		return blockSlide;
	}


	@Override
	public void setBlockSlide(boolean blockSlide) {
		this.blockSlide = blockSlide;
	}


	@Override
	public int slideWidth() {
		return isBlockSlide() ? width() : 1;
	}


	@Override
	public int slideHeight() {
		return isBlockSlide() ? height() : 1;
	}

	
}
