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
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of convolutional layer in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvLayer4DImpl extends ConvLayer4DAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, time, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer4DImpl(int neuronChannel, Function activateRef, int width, int height, int depth, int time, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, width, height, depth, time, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, time, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @param filter kernel filter.
	 */
	protected ConvLayer4DImpl(int neuronChannel, Function activateRef, int width, int height, int depth, int time, Filter filter) {
		this(neuronChannel, activateRef, width, height, depth, time, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, and time.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 */
	protected ConvLayer4DImpl(int neuronChannel, Function activateRef, int width, int height, int depth, int time) {
		this(neuronChannel, activateRef, width, height, depth, time, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer4DImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}

	
	@Override
	public NeuronValue newNeuronValue() {
		return NeuronValueCreator.newNeuronValue(neuronChannel);
	}

	
	@Override
	public ConvLayerSingle newLayer(Size size) {
		return create(neuronChannel, activateRef, size.width, size.height, size.depth, size.time, filter, idRef);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, depth, time, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return 4D convolutional layer.
	 */
	public static ConvLayer4DImpl create(int neuronChannel, Function activateRef, int width, int height, int depth, int time, Filter filter, Id idRef) {
		width = width < 1 ? 1 : width;
		height = height < 1 ? 1 : height;
		depth = depth < 1 ? 1 : depth;
		time = time < 1 ? 1 : time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ConvLayer4DImpl(neuronChannel, activateRef, width, height, depth, time, filter, idRef);
	}


	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, depth, time, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @param filter kernel filter.
	 * @return 4D convolutional layer.
	 */
	public static ConvLayer4DImpl create(int neuronChannel, Function activateRef, int width, int height, int depth, int time, Filter filter) {
		return create(neuronChannel, activateRef, width, height, depth, time, filter, null);
	}
	
	
	/**
	 * Creating convolutional layer with neuron channel, activation function, width, height, depth, and time.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @return 4D convolutional layer.
	 */
	public static ConvLayer4DImpl create(int neuronChannel, Function activateRef, int width, int height, int depth, int time) {
		return create(neuronChannel, activateRef, width, height, depth, time, null, null);
	}


}
