/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This interface provides raster utility methods.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RasterUtility {

	
	/**
	 * Converting neuron values to raster. This method is only used in case that there is no convolutional network.
	 * @param values neuron values. This is X data, not feature.
	 * @return converted raster.
	 */
	Raster createRaster(NeuronValue[] values);
		
		
}
