/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents standard neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Neuron extends Serializable, Cloneable {

	
	/**
	 * Getting input value.
	 * @return input value.
	 */
	NeuronValue getValue();

	
}
