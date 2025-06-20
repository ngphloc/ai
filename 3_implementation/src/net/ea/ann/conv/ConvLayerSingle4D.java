/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.Filter4D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a single convolutional layer in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvLayerSingle4D extends ConvLayerSingle3D {

	
	/**
	 * Getting 4D filter.
	 * @return 4D filter.
	 */
	Filter4D getFilter4D();

	
	/**
	 * Getting neuron at specific coordination.
	 * @param x horizontal coordination.
	 * @param y vertical coordination.
	 * @param z depth coordination.
	 * @param t time coordination.
	 * @return neuron at specific coordination.
	 */
	ConvNeuron get(int x, int y, int z, int t);
	
	
	/**
	 * Setting neuron value at specific coordination.
	 * @param x horizontal coordination.
	 * @param y vertical coordination.
	 * @param z depth coordination.
	 * @param t time coordination.
	 * @param value neuron value.
	 * @return previous neuron value.
	 */
	NeuronValue set(int x, int y, int z, int t, NeuronValue value);
	
	
}
