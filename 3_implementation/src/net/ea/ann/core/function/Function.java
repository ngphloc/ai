/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents function.
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
	NeuronValue evaluate(NeuronValue x);
	
	
	/**
	 * Calculate gradient (the first order derivative) at specified value.
	 * @param x specified value.
	 * @return gradient (the first order derivative) at specified value.
	 */
	NeuronValue derivative(NeuronValue x);


}
