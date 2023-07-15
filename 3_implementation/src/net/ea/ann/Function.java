/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.io.Serializable;

/**
 * This interface represents activation function for each neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Function extends Serializable, Cloneable {

	
	/**
	 * Evaluating specified value.
	 * @param x specified value.
	 * @return evaluated value.
	 */
	NeuronValue eval(NeuronValue x);
	
	
	/**
	 * Calculate gradient (the first order derivative) at specified value.
	 * @param x specified value.
	 * @return gradient (the first order derivative) at specified value.
	 */
	NeuronValue derivative(NeuronValue x);
	
	
}
