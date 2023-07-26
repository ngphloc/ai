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
 * This class is the default implementation of deconvolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DeconvNetworkImpl extends ConvNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param idRef ID reference.
	 */
	public DeconvNetworkImpl(int neuronChannel, Id idRef) {
		super(neuronChannel, idRef);
	}

	
	/**
	 * Default constructor with neuron channel.
	 * @param neuronChannel neuron channel or depth.
	 */
	public DeconvNetworkImpl(int neuronChannel) {
		super(neuronChannel);
	}

	
	@Override
	public ConvLayer newLayer(int width, int height, Filter filter) {
		return DeconvLayerImpl.create(neuronChannel, width, height, filter);
	}

	
	/**
	 * Creating deconvolutional network with neuron channel and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param idRef ID reference.
	 * @return deconvolutional network.
	 */
	public static ConvNetworkAbstract create(int neuronChannel, Id idRef) {
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new DeconvNetworkImpl(neuronChannel, idRef);
	}

	
	/**
	 * Creating deconvolutional network with neuron channel.
	 * @param neuronChannel neuron channel or depth.
	 * @return deconvolutional network.
	 */
	public static ConvNetworkAbstract create(int neuronChannel) {
		return create(neuronChannel, null);
	}


}
