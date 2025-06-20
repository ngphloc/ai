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
public class ConvLayer2DImpl extends ConvLayer2DAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer2DImpl(int neuronChannel, Function activateRef, int width, int height, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, width, height, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 */
	protected ConvLayer2DImpl(int neuronChannel, Function activateRef, int width, int height, Filter filter) {
		this(neuronChannel, activateRef, width, height, filter, null);
	}


	/**
	 * Constructor with neuron channel, activation function, width, and height.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 */
	protected ConvLayer2DImpl(int neuronChannel, Function activateRef, int width, int height) {
		this(neuronChannel, activateRef, width, height, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ConvLayer2DImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}

	
	@Override
	public NeuronValue newNeuronValue() {
		return NeuronValueCreator.newNeuronValue(neuronChannel);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return convolutional layer.
	 */
	public static ConvLayer2DImpl create(int neuronChannel, Function activateRef, int width, int height, Filter filter, Id idRef) {
		width = width < 1 ? 1 : width;
		height = height < 1 ? 1 : height;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ConvLayer2DImpl(neuronChannel, activateRef, width, height, filter, idRef);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @return convolutional layer.
	 */
	public static ConvLayer2DImpl create(int neuronChannel, Function activateRef, int width, int height, Filter filter) {
		return create(neuronChannel, activateRef, width, height, filter, null);
	}
	
	
	/**
	 * Creating convolutional layer with neuron channel, activation function, width, and height.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @return convolutional layer.
	 */
	public static ConvLayer2DImpl create(int neuronChannel, Function activateRef, int width, int height) {
		return create(neuronChannel, activateRef, width, height, null, null);
	}


}
