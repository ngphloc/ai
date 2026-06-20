/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.io.Serializable;

/**
 * This builder class provides methods to build models based on matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ManeBuilder implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Default constructor.
	 */
	public ManeBuilder() {
		super();
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ManeBuilder(int neuronChannel) {
		this();
		this.neuronChannel = neuronChannel;
	}
	
	
}
