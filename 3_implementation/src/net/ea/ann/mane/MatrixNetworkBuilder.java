/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

import net.ea.ann.core.function.Function;

/**
 * This class provides utility methods to create matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public final class MatrixNetworkBuilder implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixNetworkBuilder(int neuronChannel) {
		this.neuronChannel = neuronChannel;
	}

	
	/**
	 * Setting activation reference.
	 * @param activateRef activation reference.
	 * @return this builder.
	 */
	public MatrixNetworkBuilder setActivateRef(Function activateRef) {
		this.activateRef = activateRef;
		return this;
	}


}
