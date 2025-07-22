/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.value.NeuronValue;

/**
 * This class is the default implementation of convolutional neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvNeuronImpl implements ConvNeuron {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal value.
	 */
	protected NeuronValue value = null;
	
	
	/**
	 * Input.
	 */
	protected NeuronValue input = null;
	
	
	/**
	 * Input.
	 */
	protected Object tag = null;

	
	/**
	 * Constructor with specific layer.
	 * @param layer specific layer.
	 */
	public ConvNeuronImpl(ConvLayer layer) {
		this.value = layer.newNeuronValue();
	}

	
	@Override
	public NeuronValue getValue() {
		return value;
	}

	
	@Override
	public NeuronValue setValue(NeuronValue value) {
		NeuronValue prevValue = this.value;
		this.value = value;
		return prevValue;
	}


	@Override
	public NeuronValue getInput() {
		return input;
	}


	@Override
	public void setInput(NeuronValue input) {
		this.input = input;
	}


	@Override
	public Object getTag() {
		return tag;
	}


	@Override
	public void setTag(Object tag) {
		this.tag = tag;
	}


	@Override
	public void clear() {
		NeuronValue zero = input.zero();
		setValue(zero);
		setInput(null);
		setTag(null);
	}
	
	
}
