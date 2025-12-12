/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.io.Serializable;
import java.util.Random;

/**
 * This interface represents a filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter extends Serializable, Cloneable {

	
	/**
	 * Flag to calculate error mean.
	 */
	static boolean CALC_ERROR_MEAN = net.ea.ann.conv.filter.Filter.CALC_ERROR_MEAN;
	
	
	/**
	 * Moving stride mode.
	 */
	static boolean MOVE_STRIDE = false;
	
	
	/**
	 * Getting filter width.
	 * @return filter width.
	 */
	default int width() {return 1;}

	
	/**
	 * Getting stride width.
	 * @return stride width.
	 */
	default int getStrideWidth() {
		return isMoveStride() ? width() : 1;
	}
	
	
	/**
	 * Getting filter height.
	 * @return filter height.
	 */
	default int height() {return 1;}
	
	
	/**
	 * Getting stride height.
	 * @return stride height.
	 */
	default int getStrideHeight() {
		return isMoveStride() ? height() : 1;
	}
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @return whether to move according to stride when filtering.
	 */
	boolean isMoveStride();
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @param moveStride flag to whether to move according to stride when filtering.
	 */
	void setMoveStride(boolean moveStride);
	
	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	default boolean isPadZero() {return true;}

	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	default void initialize(double v) {}
	
	
	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	default void initialize(Random rnd) {}
	
	
	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	default int sizeOfParams() {return 0;}

	
}



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

