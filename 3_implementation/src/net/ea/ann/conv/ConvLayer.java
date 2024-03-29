/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Layer;
import net.ea.ann.core.NeuronValue;

/**
 * This interface represents a convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvLayer extends Layer {


	/**
	 * Create neuron.
	 * @return created neuron.
	 */
	ConvNeuron newNeuron();

	
	/**
	 * Getting filter.
	 * @return internal filter.
	 */
	Filter getFilter();
	
	
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
	
	
	/**
	 * Getting raster width.
	 * @return raster width.
	 */
	int getWidth();
	
	
	/**
	 * Getting raster height.
	 * @return raster height.
	 */
	int getHeight();
	
	
	/**
	 * Getting size of neurons.
	 * @return size of neurons.
	 */
	int size();
	
	
	/**
	 * Getting data as array of neurons.
	 * @return data as array of neurons.
	 */
	ConvNeuron[] getNeurons();
	
	
	/**
	 * Getting data as array of neuron value.
	 * @return data as array of neuron value.
	 */
	NeuronValue[] getData();

	
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
	 * @return the data of the next layer.
	 */
	ConvNeuron[] forward();
	
	
}
