/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a single convolutional layer in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvLayerSingle2D extends ConvLayerSingle1D {

	
	/**
	 * Getting 2D filter.
	 * @return 2D filter.
	 */
	Filter2D getFilter2D();

	
	/**
	 * Getting neuron at specific coordination.
	 * @param x horizontal coordination.
	 * @param y vertical coordination.
	 * @return neuron at specific coordination.
	 */
	ConvNeuron get(int x, int y);
	
	
	/**
	 * Setting neuron value at specific coordination.
	 * @param x horizontal coordination.
	 * @param y vertical coordination.
	 * @param value neuron value.
	 * @return previous neuron value.
	 */
	NeuronValue set(int x, int y, NeuronValue value);
	
	
}
