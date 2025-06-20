/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.Layer;
import net.ea.ann.core.value.NeuronValueCreator;

/**
 * This interface represents a convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvLayer extends Layer, NeuronValueCreator {


	/**
	 * Create neuron.
	 * @return created neuron.
	 */
	ConvNeuron newNeuron();

	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	int getNeuronChannel();
	
	
	/**
	 * Getting previous layer.
	 * @return previous layer.
	 */
	ConvLayer getPrevLayer();

	
	/**
	 * Getting next layer.
	 * @return next layer.
	 */
	ConvLayer getNextLayer();


	/**
	 * Setting next layer.
	 * @param nextLayer next layer. It can be null.
	 * @return true if setting is successful.
	 */
	boolean setNextLayer(ConvLayer nextLayer);

	
	/**
	 * Forwarding to evaluate the next layer.
	 * @return the next layer.
	 */
	ConvLayer forward();
	
	
}
