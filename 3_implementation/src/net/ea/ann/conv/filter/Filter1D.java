/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a filter in 1D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter1D extends Filter {

	
	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param x specified x index.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, ConvLayerSingle1D layer);


}
