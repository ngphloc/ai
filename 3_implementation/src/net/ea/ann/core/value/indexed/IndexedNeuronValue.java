/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.indexed;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents an indexed neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface IndexedNeuronValue extends NeuronValue {

	
	/**
	 * Getting current indexed value.
	 * @return current indexed value.
	 */
	NeuronValue v();
	
	
	/**
	 * Getting index.
	 * @return index.
	 */
	int getIndex();
	
	
	/**
	 * Setting index
	 * @param index specified index.
	 */
	void setIndex(int index);
	
	
	/**
	 * Getting size.
	 * @return size.
	 */
	int size();
	
	
	/**
	 * Getting value at specified index.
	 * @param index specified index.
	 * @return value at specified index.
	 */
	NeuronValue get(int index);
	
	
	/**
	 * Setting value at specified index.
	 * @param index specified index.
	 * @param value specified value.
	 * @return previous value.
	 */
	NeuronValue set(int index, NeuronValue value);
	
	
}
