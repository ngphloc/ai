/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents invertible function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface FunctionInvertible extends Function {

	
	/**
	 * Evaluating specified value by inverse function.
	 * @param y specified value.
	 * @return evaluated value by inverse function.
	 */
	NeuronValue evaluateInverse(NeuronValue y);


	/**
	 * Calculate gradient (the first order derivative) at specified value by inverse function.
	 * @param y specified value.
	 * @return gradient (the first order derivative) at specified value by inverse function.
	 */
	NeuronValue derivativeInverse(NeuronValue y);

	
}
