/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.io.Serializable;

/**
 * This interface represents a factory to create filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface FilterFactory extends Serializable, Cloneable {

	
	/**
	 * Creating product filter.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @return product filter.
	 */
	Filter productFilter(int width, int height);

		
	/**
	 * Creating mean product filter.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @return mean product filter.
	 */
	Filter meanFilter(int width, int height);
	
	
}
