/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.LayerStandard;

/**
 * This interface represents soft-max function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Softmax extends Probability, FunctionDelay {

	
	/**
	 * Creating soft-max function.
	 * @param neuronChannel neuron channel.
	 * @param layer layer.
	 * @return soft-max function.
	 */
	static Softmax create(int neuronChannel, LayerStandard layer) {
		if (neuronChannel < 1)
			return null;
		else if (neuronChannel == 1)
			return Softmax1.create(layer);
		else
			return SoftmaxV.create(layer);
	}

	
}
