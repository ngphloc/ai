/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import net.ea.ann.core.function.Function;

/**
 * This class is default implementation of standard layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LayerStandardImpl extends LayerStandardAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public LayerStandardImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}


	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public LayerStandardImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 */
	public LayerStandardImpl() {
		this(1, null, null);
	}

	
	@Override
	public NeuronValue newNeuronValue() {
		if (neuronChannel <= 0)
			return null;
		else if (neuronChannel == 1)
			return new NeuronValue1(0.0).zero();
		else
			return new NeuronValueV(neuronChannel).zero();
	}

	
}
