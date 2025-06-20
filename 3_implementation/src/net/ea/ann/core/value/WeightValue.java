/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.util.Arrays;

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
	WeightValue unit();

	
	/**
	 * Converting this weight value to neuron value.
	 * @return converted neuron value.
	 */
	NeuronValue toNeuronValue();
	
	
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


	/**
	 * Concatenating two arrays
	 * @param array1 first array.
	 * @param array2 second array.
	 * @return the array concatenated from the two arrays.
	 */
	static WeightValue[] concatArray(WeightValue[] array1, WeightValue[] array2) {
		if (array1 == null && array2 == null) return null;
		if (array1 != null && array2 == null) return (array1.length != 0 ? array1 : null);
		if (array1 == null && array2 != null) return (array2.length != 0 ? array2 : null);
		if (array1.length == 0 && array2.length == 0) return null;
		if (array1.length > 0 && array2.length == 0) return array1;
		if (array1.length == 0 && array2.length > 0) return array2;
		
		int n = array1.length + array2.length;
		WeightValue[] array = Arrays.copyOfRange(array1, 0, n);
		for (int i = array1.length; i < n; i++) array[i] = array2[i - array1.length];
		
		return array;
	}


}
