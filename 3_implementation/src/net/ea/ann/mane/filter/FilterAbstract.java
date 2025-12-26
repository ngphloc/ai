package net.ea.ann.mane.filter;

import net.ea.ann.mane.Filter;

/**
 * This class is an abstract implementation of filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class FilterAbstract implements Filter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to move according to stride when filtering.
	 */
	protected boolean moveStride = MOVE_STRIDE;

	
	/**
	 * Default constructor.
	 */
	protected FilterAbstract() {
		super();
	}

	
	@Override
	public boolean isMoveStride() {return moveStride;}


	@Override
	public void setMoveStride(boolean moveStride) {this.moveStride = moveStride;}


}

