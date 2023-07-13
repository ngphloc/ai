/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.io.Serializable;
import java.util.Arrays;

/**
 * This interface represents a value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Value extends Serializable, Cloneable {

	
	/**
	 * Add to other value.
	 * @param value other value.
	 * @return added value.
	 */
	Value add(Value value);
	

	/**
	 * Subtract to other value.
	 * @param value other value.
	 * @return subtracted value.
	 */
	Value subtract(Value value);
	
	
	/**
	 * Multiply with other value.
	 * @param value other value.
	 * @return multiplied value.
	 */
	Value multiply(Value value);
	
	
	/**
	 * Multiply with other value.
	 * @param value other value.
	 * @return multiplied value.
	 */
	Value multiply(double value);


	/**
	 * Divide by other value.
	 * @param value other value.
	 * @return divided value.
	 */
	Value divide(Value value);

	
	/**
	 * Calculate norm.
	 * @return norm value.
	 */
	double norm();
	
	
	/**
	 * Create an array of values.
	 * @param length array length.
	 * @param layer referred layer.
	 * @return array of values.
	 */
	public static Value[] makeArray(int length, Layer layer) {
		Value[] array = new Value[length];
		for (int j = 0; j < length; j++)
			array[j] = layer != null ? layer.newValue() : new ValueScalar(0.0);
		
		return array;
	}
	
	
	/**
	 * Adjusting array by length.
	 * @param array specified array.
	 * @param length specified length.
	 * @return adjusted array.
	 */
	public static Value[] adjustArray(Value[] array, int length, Layer layer) {
		if (length <= 0) return array;
		if (array == null || array.length == 0) {
			array = Value.makeArray(length, layer);
		}
		else if (array.length < length) {
			int originLength = array.length;
			array = Arrays.copyOfRange(array, 0, length);
			for (int j = originLength; j < length; j++) {
				if (array[j] == null)
					array[j] = layer != null ? layer.newValue() : new ValueScalar(0.0);
			}
		}
		
		return array;
	}
	
	
}
