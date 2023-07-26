/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

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
	Filter product(int width, int height);

		
	/**
	 * Creating identity product filter as simple convolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @return identity product filter as simple convolutional filter.
	 */
	Filter zoomOut(int width, int height);

	
	/**
	 * Creating mean product filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @return mean product filter.
	 */
	Filter mean(int width, int height);
	
	
	/**
	 * Creating zoom-in filter as simple deconvolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @return zoom-in filter as simple deconvolutional filter.
	 */
	Filter zoomIn(int width, int height);
	
	
}
