/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

/**
 * This interface represents a weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface WeightValue extends Value {

	
	/**
	 * Retrieve zero value.
	 * @return zero value.
	 */
	WeightValue zero();
	
	
	/**
	 * Getting identity.
	 * @return identity.
	 */
	WeightValue identity();

	
	/**
	 * Add to other neuron value.
	 * @param value other neuron value.
	 * @return added value.
	 */
	WeightValue add(NeuronValue value);


	/**
	 * Subtract to other value.
	 * @param value other value.
	 * @return subtracted value.
	 */
	WeightValue subtract(NeuronValue value);


}
