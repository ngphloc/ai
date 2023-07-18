/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.Neuron;
import net.ea.ann.core.NeuronValue;

/**
 * This class represents convolutional neuron.
 * 
 * @author Loc Nguyen
 *
 */
public class ConvNeuron implements Neuron {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal value.
	 */
	protected NeuronValue value = null;
	
	
	/**
	 * Constructor with specific layer.
	 * @param layer specific layer.
	 */
	public ConvNeuron(ConvLayer layer) {
		this.value = layer.newNeuronValue();
	}

	
	@Override
	public NeuronValue getValue() {
		return value;
	}

	
	/**
	 * Setting value.
	 * @param value specific value.
	 */
	public void setValue(NeuronValue value) {
		this.value = value;
	}
	
	
}
