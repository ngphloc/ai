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
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;

/**
 * This class is the default implementation of convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvLayer1DImpl extends ConvLayer1DAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, width, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer1DImpl(int neuronChannel, Function activateRef, int width, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, width, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param filter kernel filter.
	 */
	protected ConvLayer1DImpl(int neuronChannel, Function activateRef, int width, Filter filter) {
		this(neuronChannel, activateRef, width, filter, null);
	}


	/**
	 * Constructor with neuron channel, activation function, and width.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 */
	protected ConvLayer1DImpl(int neuronChannel, Function activateRef, int width) {
		this(neuronChannel, activateRef, width, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ConvLayer1DImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}

	
	@Override
	public NeuronValue newNeuronValue() {
		return NeuronValueCreator.newNeuronValue(neuronChannel);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return convolutional layer.
	 */
	public static ConvLayer1DImpl create(int neuronChannel, Function activateRef, int width, Filter filter, Id idRef) {
		width = width < 1 ? 1 : width;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ConvLayer1DImpl(neuronChannel, activateRef, width, filter, idRef);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param filter kernel filter.
	 * @return convolutional layer.
	 */
	public static ConvLayer1DImpl create(int neuronChannel, Function activateRef, int width, Filter filter) {
		return create(neuronChannel, activateRef, width, filter, null);
	}
	
	
	/**
	 * Creating convolutional layer with neuron channel, activation function, and width.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @return convolutional layer.
	 */
	public static ConvLayer1DImpl create(int neuronChannel, Function activateRef, int width) {
		return create(neuronChannel, activateRef, width, null, null);
	}


}
