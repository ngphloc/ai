/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.NeuronValue1;
import net.ea.ann.core.NeuronValueV;

/**
 * This class is the default implementation of convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvLayerImpl extends ConvLayerAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	public ConvLayerImpl(int neuronChannel, int width, int height, Filter filter, Id idRef) {
		super(neuronChannel, width, height, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 */
	public ConvLayerImpl(int neuronChannel, int width, int height, Filter filter) {
		this(neuronChannel, width, height, filter, null);
	}


	@Override
	public NeuronValue newNeuronValue() {
		if (neuronChannel <= 0)
			return null;
		else if (neuronChannel == 1)
			return new NeuronValue1(0.0).zero();
		else
			return new NeuronValueV(neuronChannel).zero();
	}


	/**
	 * Creating convolutional layer with neuron channel, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return convolutional layer.
	 */
	public static ConvLayer create(int neuronChannel, int width, int height, Filter filter, Id idRef) {
		width = width < 0 ? 0 : width;
		height = height < 0 ? 0 : height;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		
		return new ConvLayerImpl(neuronChannel, width, height, filter, idRef);
	}


	/**
	 * Creating convolutional layer with neuron channel, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @return convolutional layer.
	 */
	public static ConvLayer create(int neuronChannel, int width, int height, Filter filter) {
		return create(neuronChannel, width, height, filter, null);
	}
	
	
	/**
	 * Creating convolutional layer with neuron channel, width, and height.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @return convolutional layer.
	 */
	public static ConvLayer create(int neuronChannel, int width, int height) {
		return create(neuronChannel, width, height, null, null);
	}


}
