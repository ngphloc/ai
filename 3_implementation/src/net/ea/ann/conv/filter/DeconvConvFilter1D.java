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
 * This interface represents a deconvolutional filter based on a convolutional filter in 1S space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface DeconvConvFilter1D extends DeconvFilter1D, DeconvConvFilter {

	
	/**
	 * Applying this filter to specific layer and next layer. Please attention to this important method.
	 * This is interpolation technique where next layer is larger than this layer.
	 * @param x x coordinator.
	 * @param layer specific layer (this layer).
	 * @param nextX x coordinator at next layer.
	 * @param nextLayer next layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, ConvLayerSingle1D layer, int nextX, ConvLayerSingle1D nextLayer);


}
