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
 * This interface represents a filter in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter4D extends Filter3D {

	
	/**
	 * Applying this filter to specific layer.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param z z coordinator.
	 * @param t t coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer);


}
