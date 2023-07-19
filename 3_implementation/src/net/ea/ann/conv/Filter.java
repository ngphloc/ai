/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.io.Serializable;

import net.ea.ann.core.NeuronValue;

/**
 * This class represents a filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter extends Serializable, Cloneable {

	
	/**
	 * Getting filter width.
	 * @return filter width.
	 */
	int width();

	
	/**
	 * Getting filter height.
	 * @return filter height.
	 */
	int height();
	
	
	/**
	 * Checking whether to slide according to block when filtering.
	 * @return whether to slide according to block when filtering.
	 */
	boolean isBlockSlide();
	
	
	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, ConvLayer layer);

	
}
