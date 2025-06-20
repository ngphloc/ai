/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

/**
 * This interface represents a utility to create neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NeuronValueCreator extends Serializable, Cloneable {

	
	/**
	 * Creating an empty neuron value.
	 * @return empty neuron value.
	 */
	NeuronValue newNeuronValue();

	
	/**
	 * Creating an empty neuron value from specified channel.
	 * @param neuronChannel specified channel.
	 * @return empty neuron value.
	 */
	static NeuronValue newNeuronValue(int neuronChannel) {
		if (neuronChannel <= 0)
			return null;
		else if (neuronChannel == 1)
			return new NeuronValue1(0.0).zero();
		else
			return new NeuronValueV(neuronChannel).zero();
	}


}
