/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter2D extends Filter1D {

	
	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, ConvLayerSingle2D layer);


	/**
	 * Calculating derivative of kernel of this layer given next layer as bias layer at specified coordinator.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param thisLayer this layer.
	 * @param nextLayer next layer as bias layer.
	 * @return derivative of kernel of this layer given next layer as bias layer.
	 */
	public NeuronValue[][] dKernel(int nextX, int nextY, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer);

	
	/**
	 * Calculating derivative of this layer given next layer as bias layer at specified coordinator.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param thisLayer this layer.
	 * @param nextLayer next layer as bias layer.
	 * @return derivative of this layer given next layer as bias layer.
	 */
	public NeuronValue[][] dValue(int nextX, int nextY, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer);

	
}
