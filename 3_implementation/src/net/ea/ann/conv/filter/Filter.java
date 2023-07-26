/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import java.io.Serializable;

import net.ea.ann.conv.ConvLayer;
import net.ea.ann.core.NeuronValue;

/**
 * This interface represents a filter.
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
	 * Getting sliding width.
	 * @return
	 */
	int slideWidth();
	
	
	/**
	 * Getting filter height.
	 * @return filter height.
	 */
	int height();
	
	
	/**
	 * Getting sliding height.
	 * @return
	 */
	int slideHeight();
	
	
	/**
	 * Checking whether to slide according to block when filtering.
	 * @return whether to slide according to block when filtering.
	 */
	boolean isBlockSlide();
	
	
	/**
	 * Checking whether to slide according to block when filtering.
	 * @param blockSlide flag to slide according to block when filtering.
	 */
	void setBlockSlide(boolean blockSlide);
	
	
	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, ConvLayer layer);


	/**
	 * Calculating zooming ratio of filters.
	 * @param filters specified filters.
	 * @return zooming ratio of filters.
	 */
	static int zoomRatioOf(Filter[] filters) {
		if (filters == null || filters.length == 0)
			return 1;
		else {
			int zoom = 1;
			for (Filter filter : filters) {
				zoom *= Math.max(filter.slideWidth(), filter.slideHeight());
			}
			
			return zoom;
		}
		
	}

	
}
