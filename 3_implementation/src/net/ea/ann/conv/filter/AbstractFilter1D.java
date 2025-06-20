/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

/**
 * This class is an abstract implementation of filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractFilter1D implements Filter1D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to move according to stride when filtering.
	 */
	protected boolean moveStride = true;

	
	/**
	 * Default constructor.
	 */
	protected AbstractFilter1D() {
		super();
	}

	
	@Override
	public boolean isMoveStride() {
		return moveStride;
	}


	@Override
	public void setMoveStride(boolean moveStride) {
		this.moveStride = moveStride;
	}


	@Override
	public int getStrideWidth() {
		return isMoveStride() ? width() : 1;
	}


	@Override
	public int height() {
		return 1;
	}
	
	
	@Override
	public int getStrideHeight() {
		return isMoveStride() ? height() : 1;
	}


	@Override
	public int depth() {
		return 1;
	}


	@Override
	public int getStrideDepth() {
		return isMoveStride() ? depth() : 1;
	}

	
	@Override
	public int time() {
		return 1;
	}


	@Override
	public int getStrideTime() {
		return isMoveStride() ? time() : 1;
	}


}
