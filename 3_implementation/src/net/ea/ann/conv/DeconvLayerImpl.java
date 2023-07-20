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

/**
 * This class is the default implementation of convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DeconvLayerImpl extends ConvLayerImpl implements DeconvLayer {


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
	public DeconvLayerImpl(int neuronChannel, int width, int height, Filter filter, Id idRef) {
		super(neuronChannel, width, height, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 */
	public DeconvLayerImpl(int neuronChannel, int width, int height, Filter filter) {
		super(neuronChannel, width, height, filter);
	}

	
	/**
	 * Creating deconvolutional layer with neuron channel, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return deconvolutional layer.
	 */
	public static DeconvLayer create(int neuronChannel, int width, int height, Filter filter, Id idRef) {
		if (width <= 0 || height <= 0) return null;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		
		return new DeconvLayerImpl(neuronChannel, width, height, filter, idRef);
	}


	/**
	 * Creating deconvolutional layer with neuron channel, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @return deconvolutional layer.
	 */
	public static DeconvLayer create(int neuronChannel, int width, int height, Filter filter) {
		return create(neuronChannel, width, height, filter, null);
	}
	
	
	/**
	 * Creating deconvolutional layer with neuron channel, width, and height.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @return deconvolutional layer.
	 */
	public static DeconvLayer create(int neuronChannel, int width, int height) {
		return create(neuronChannel, width, height, null, null);
	}


}
