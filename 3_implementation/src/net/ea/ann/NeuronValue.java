/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.util.Arrays;

/**
 * This interface represents a neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NeuronValue extends Value {


	/**
	 * Getting zero.
	 * @return zero.
	 */
	NeuronValue zero();
	
	
	/**
	 * Getting identity.
	 * @return identity.
	 */
	NeuronValue identity();
	
	
	/**
	 * Create weight value.
	 * @return weight value.
	 */
	WeightValue newWeightValue();
	
	
	/**
	 * Getting negative inverse.
	 * @return negative inverse.
	 */
	NeuronValue negative();
	
	
	/**
	 * Add to other value.
	 * @param value other value.
	 * @return added value.
	 */
	NeuronValue add(NeuronValue value);
	

	/**
	 * Subtract to other value.
	 * @param value other value.
	 * @return subtracted value.
	 */
	NeuronValue subtract(NeuronValue value);
	
	
	/**
	 * Multiply with other neuron value.
	 * @param value other neuron value.
	 * @return multiplied value.
	 */
	NeuronValue multiply(NeuronValue value);

	
	/**
	 * Multiply with other weight value.
	 * @param value other weight value.
	 * @return multiplied value.
	 */
	NeuronValue multiply(WeightValue value);
	
	
	/**
	 * Multiply with other value.
	 * @param value other value.
	 * @return multiplied value.
	 */
	NeuronValue multiply(double value);


	/**
	 * Multiply with other derivative.
	 * @param value other derivative.
	 * @return multiplied value.
	 */
	NeuronValue multiplyDerivative(NeuronValue derivative);

	
	/**
	 * Taking squared root of this value.
	 * @return squared root of this value.
	 */
	NeuronValue sqrt();

	
	/**
	 * Calculate norm.
	 * @return norm value.
	 */
	double norm();
	
	
	/**
	 * Calculating inverse of matrix.
	 * @param matrix specific matrix.
	 * @return inverse of matrix.
	 */
	NeuronValue[][] matrixInverse(NeuronValue[][] matrix);
	
	
	/**
	 * Multiplying matrix and vector.
	 * @param matrix specific matrix.
	 * @param vector specific vector.
	 * @return applied vector.
	 */
	static NeuronValue[] matrixMultiply(NeuronValue[][] matrix, NeuronValue[] vector) {
		if (matrix == null || vector == null) return null;
		
		NeuronValue[] result = new NeuronValue[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = vector[0].zero();
			for (int j = 0; j < matrix[i].length; j++) {
				result[i] = result[i].add(matrix[i][j].multiply(vector[j]));
			}
		}
		
		return result;
	}

	
	/**
	 * Making squared root of matrix.
	 * @param matrix specific matrix.
	 * @return squared root matrix.
	 */
	static NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		if (matrix == null) return null;
		
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].sqrt();
			}
		}
		
		return result;
	}

	
	/**
	 * Create an array of values.
	 * @param length array length.
	 * @param layer referred layer.
	 * @return array of values.
	 */
	static NeuronValue[] makeArray(int length, Layer layer) {
		NeuronValue[] array = new NeuronValue[length];
		for (int j = 0; j < length; j++)
			array[j] = layer.newNeuronValue();
		
		return array;
	}


	/**
	 * Adjusting array by length.
	 * @param array specified array.
	 * @param length specified length.
	 * @return adjusted array.
	 */
	static NeuronValue[] adjustArray(NeuronValue[] array, int length, Layer layer) {
		if (length <= 0) return array;
		if (array == null || array.length == 0) {
			array = NeuronValue.makeArray(length, layer);
		}
		else if (array.length < length) {
			int originLength = array.length;
			array = Arrays.copyOfRange(array, 0, length);
			for (int j = originLength; j < length; j++) {
				if (array[j] == null)
					array[j] = layer.newNeuronValue();
			}
		}
		
		return array;
	}


}
