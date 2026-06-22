/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;

/**
 * This class implements matrix neural network with dropout technique.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DropoutNetwork extends MatrixNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public DropoutNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(DropoutLayer.DROPOUT_MODE_FIELD, DropoutLayer.DROPOUT_MODE_DEFAULT);
		config.put(DropoutLayer.DROPOUT_RATE_FIELD, DropoutLayer.DROPOUT_RATE_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public DropoutNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public DropoutNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public DropoutNetwork(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	@Override
	protected MatrixLayerAbstract newLayer() {
		DropoutLayer layer = new DropoutLayer(neuronChannel, getActivateRef(), getConvActivateRef(), idRef);
		layer.setNetwork(this);
		return layer;
	}


	/**
	 * Checking dropout mode.
	 * @return dropout mode.
	 */
	boolean paramIsDropoutMode() {
		if (config.containsKey(DropoutLayer.DROPOUT_MODE_FIELD))
			return config.getAsBoolean(DropoutLayer.DROPOUT_MODE_FIELD);
		else
			return DropoutLayer.DROPOUT_MODE_DEFAULT;
	}
	
	
	/**
	 * Setting dropout mode.
	 * @param dropout dropout mode.
	 * @return this network.
	 */
	DropoutNetwork paramSetDropoutMode(boolean dropout) {
		config.put(DropoutLayer.DROPOUT_MODE_FIELD, dropout);
		return this;
	}


	/**
	 * Getting dropout rate.
	 * @return dropout rate.
	 */
	double paramGetDropoutRate() {
		if (config.containsKey(DropoutLayer.DROPOUT_RATE_FIELD))
			return config.getAsReal(DropoutLayer.DROPOUT_RATE_FIELD);
		else
			return DropoutLayer.DROPOUT_RATE_DEFAULT;
	}
	
	
	/**
	 * Setting dropout rate.
	 * @param dropoutRate dropout rate.
	 * @return network.
	 */
	DropoutNetwork paramSetDropoutRate(double dropoutRate) {
		dropoutRate = dropoutRate < 0 ? 0 : dropoutRate;
		dropoutRate = dropoutRate > 1 ? 1 : dropoutRate;
		config.put(DropoutLayer.DROPOUT_RATE_FIELD, dropoutRate);
		return this;
	}


}
