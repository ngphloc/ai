/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a deconvolutional filter based on a convolutional filter in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface DeconvConvFilter4D extends DeconvFilter4D, DeconvConvFilter3D {


	/**
	 * Applying this filter to specific layer and next layer. Please attention to this important method.
	 * This is interpolation technique where next layer is larger than this layer.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param z z coordinator.
	 * @param t t coordinator.
	 * @param layer specific layer (this layer).
	 * @param nextX x coordinator at next layer.
	 * @param nextY y coordinator at next layer.
	 * @param nextZ z coordinator at next layer.
	 * @param nextT t coordinator at next layer.
	 * @param nextLayer next layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer, int nextX, int nextY, int nextZ, int nextT, ConvLayerSingle4D nextLayer);


}
