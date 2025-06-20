/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;

/**
 * This class is the default implementation of convolutional stack network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class StackNetworkImpl extends StackNetworkAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	protected StackNetworkImpl(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		super(neuronChannel, activateRef, contentActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 */
	protected StackNetworkImpl(int neuronChannel, Function activateRef, Function contentActivateRef) {
		this(neuronChannel, activateRef, contentActivateRef, null);
	}

	
	/**
	 * Creating stack network with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return stack network.
	 */
	public static StackNetworkImpl create(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		return new StackNetworkImpl(neuronChannel, activateRef, contentActivateRef, idRef);
	}


	/**
	 * Creating stack network with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @return stack network.
	 */
	public static StackNetworkImpl create(int neuronChannel, Function activateRef, Function contentActivateRef) {
		return create(neuronChannel, activateRef, contentActivateRef, null);
	}


	/**
	 * Creating stack network with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return stack network.
	 */
	public static StackNetworkImpl create(int neuronChannel, Function contentActivateRef, Id idRef) {
		return create(neuronChannel, null, contentActivateRef, idRef);
	}

	
	/**
	 * Creating stack network with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @return stack network.
	 */
	public static StackNetworkImpl create(int neuronChannel, Function contentActivateRef) {
		return create(neuronChannel, null, contentActivateRef, null);
	}

	
	/**
	 * Creating raster stack network with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return raster stack network.
	 */
	public static StackNetworkRaster createRSN(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		return new StackNetworkRaster(neuronChannel, activateRef, contentActivateRef, idRef);
	}


	/**
	 * Creating raster stack network with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @return raster stack network.
	 */
	public static StackNetworkRaster createRSN(int neuronChannel, Function activateRef, Function contentActivateRef) {
		return createRSN(neuronChannel, activateRef, contentActivateRef, null);
	}


	/**
	 * Creating stack network with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return raster stack network.
	 */
	public static StackNetworkRaster createRSN(int neuronChannel, Function contentActivateRef, Id idRef) {
		return createRSN(neuronChannel, null, contentActivateRef, idRef);
	}

	
	/**
	 * Creating stack network with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @return raster stack network.
	 */
	public static StackNetworkRaster createRSN(int neuronChannel, Function contentActivateRef) {
		return createRSN(neuronChannel, null, contentActivateRef, null);
	}

	
}
