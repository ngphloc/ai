/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle3D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a filter in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter3D extends Filter2D {

	
	/**
	 * Applying this filter to specific layer.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param z z coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, int z, ConvLayerSingle3D layer);


}
