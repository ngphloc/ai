/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.Filter1D;

/**
 * This interface represents a single convolutional layer in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvLayerSingle1D extends ConvLayerSingle {

	
	/**
	 * Getting 1D filter.
	 * @return 1D filter.
	 */
	Filter1D getFilter1D();


}
