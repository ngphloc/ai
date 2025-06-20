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
 * This class is the default implementation of convolutional layer in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvLayer3DImpl extends ConvLayer3DAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer3DImpl(int neuronChannel, Function activateRef, int width, int height, int depth, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, width, height, depth, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param filter kernel filter.
	 */
	protected ConvLayer3DImpl(int neuronChannel, Function activateRef, int width, int height, int depth, Filter filter) {
		this(neuronChannel, activateRef, width, height, depth, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, and depth.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 */
	protected ConvLayer3DImpl(int neuronChannel, Function activateRef, int width, int height, int depth) {
		this(neuronChannel, activateRef, width, height, depth, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer3DImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}

	
	@Override
	public NeuronValue newNeuronValue() {
		return NeuronValueCreator.newNeuronValue(neuronChannel);
	}

	
	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, depth, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return 3D convolutional layer.
	 */
	public static ConvLayer3DImpl create(int neuronChannel, Function activateRef, int width, int height, int depth, Filter filter, Id idRef) {
		width = width < 1 ? 1 : width;
		height = height < 1 ? 1 : height;
		depth = depth < 1 ? 1 : depth;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ConvLayer3DImpl(neuronChannel, activateRef, width, height, depth, filter, idRef);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, depth, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param filter kernel filter.
	 * @return 3D convolutional layer.
	 */
	public static ConvLayer3DImpl create(int neuronChannel, Function activateRef, int width, int height, int depth, Filter filter) {
		return create(neuronChannel, activateRef, width, height, depth, filter, null);
	}
	
	
	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, and depth.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @return 3D convolutional layer.
	 */
	public static ConvLayer3DImpl create(int neuronChannel, Function activateRef, int width, int height, int depth) {
		return create(neuronChannel, activateRef, width, height, depth, null, null);
	}


}
