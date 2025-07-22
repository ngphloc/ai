/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.util.Arrays;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.raster.RasterAssoc;

/**
 * This interface represents a neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NeuronValue extends Value {


	/**
	 * Maximum range.
	 */
	static int MAX_RANGE = 5;
	
	
	/**
	 * Getting zero.
	 * @return zero.
	 */
	NeuronValue zero();
	
	
	/**
	 * Getting identity.
	 * @return identity.
	 */
	NeuronValue unit();
	
	
	/**
	 * Getting length.
	 * @return length.
	 */
	int length();
	
	
	/**
	 * Getting dimension.
	 * @return dimension.
	 */
	int dim();
	
	
	/**
	 * Resizing this value.
	 * @param newDim new dimension.
	 * @return resized value.
	 */
	NeuronValue resize(int newDim);
	
	
	/**
	 * Duplicate this neuron value.
	 * @return duplicated neuron value.
	 */
	NeuronValue duplicate();
	
	
	/**
	 * Checking equality to other value.
	 * @param value other value
	 * @return whether to be equal to other value.
	 */
	boolean equals(NeuronValue value);
	
	
	/**
	 * Create weight value.
	 * @return weight value.
	 */
	WeightValue newWeightValue();
	
	
	/**
	 * Converting this value to weight value.
	 * @return converted 
	 */
	WeightValue toWeightValue();
	
	
	/**
	 * Getting negative inverse.
	 * @return negative inverse.
	 */
	NeuronValue negative();
	
	
	/**
	 * Checking whether this value can be inverted.
	 * @return whether this value can be inverted.
	 */
	boolean canInvert();
	
	
	/**
	 * Getting multiplication inverse.
	 * @return multiplication inverse.
	 */
	NeuronValue inverse();

	
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
	 * @param derivative other derivative.
	 * @return multiplied value.
	 */
	NeuronValue multiplyDerivative(NeuronValue derivative);

	
	/**
	 * Divide with other value.
	 * @param value other value.
	 * @return multiplied value.
	 */
	NeuronValue divide(NeuronValue value);
	
	
	/**
	 * Divide with other value.
	 * @param value other value.
	 * @return multiplied value.
	 */
	NeuronValue divide(double value);

	
	/**
	 * Taking power of this value with specified exponent.
	 * @param exponent specified exponent.
	 * @return power of this value with specified exponent.
	 */
	NeuronValue power(double exponent);

	
	/**
	 * Taking squared root of this value.
	 * @return squared root of this value.
	 */
	NeuronValue sqrt();

	
	/**
	 * Natural exponent of this value.
	 * @return natural exponent of this value.
	 */
	NeuronValue exp();

	
	/**
	 * Natural logarithm of this value.
	 * @return natural logarithm of this value.
	 */
	NeuronValue log();

	
	/**
	 * Calculate mean.
	 * @return mean value.
	 */
	double mean();
	
	
	/**
	 * Calculate norm.
	 * @return norm value.
	 */
	double norm();
	
	
	/**
	 * Converting double value to neuron value.
	 * @param value specific double value.
	 * @return neuron value.
	 */
	NeuronValue valueOf(double value);
	
	
	/**
	 * Minimum comparison with other value.
	 * @param value other value.
	 * @return maximum one.
	 */
	NeuronValue min(NeuronValue value);

	
	/**
	 * Maximum comparison with other value.
	 * @param value other value.
	 * @return maximum one.
	 */
	NeuronValue max(NeuronValue value);
	
	
	/**
	 * Checking whether the specified matrix is invertible.
	 * @param matrix specified matrix.
	 * @return whether the specified matrix is invertible.
	 */
	boolean matrixIsInvertible(NeuronValue[][] matrix);
	
	
	/**
	 * Calculating determinant of specified matrix.
	 * @param matrix specified matrix.
	 * @return determinant of specified matrix.
	 */
	NeuronValue matrixDet(NeuronValue[][] matrix);

	
	/**
	 * Calculating inverse of matrix.
	 * @param matrix specific matrix.
	 * @return inverse of matrix.
	 */
	NeuronValue[][] matrixInverse(NeuronValue[][] matrix);
	
	
	/**
	 * Calculating square root of matrix.
	 * @param matrix specific matrix.
	 * @return square root of matrix.
	 */
	NeuronValue[][] matrixSqrt(NeuronValue[][] matrix);

	
	/**
	 * Flattening this value into array of smaller dimension value.
	 * @param smallerDim smaller dimension.
	 * @return array of smaller dimension values.
	 */
	NeuronValue[] flatten(int smallerDim);
	
	
	/**
	 * Flattening array
	 * @param array specified array.
	 * @param smallerDim smaller dimension.
	 * @return flattened array.
	 */
	NeuronValue[] flatten(NeuronValue[] array, int smallerDim);

	
	/**
	 * Aggregating value array into singular value.
	 * @param array value array.
	 * @return singular value.
	 */
	NeuronValue aggregate(NeuronValue[] array);
	
	
	/**
	 * Aggregating array according to larger dimension.
	 * @param array specified array.
	 * @param largerDim larger dimension.
	 * @return aggregated array.
	 */
	NeuronValue[] aggregate(NeuronValue[] array, int largerDim);
	
	
	/**
	 * Evaluating this neuron value given specified function.
	 * @param f specified function.
	 * @return evaluated value.
	 */
	NeuronValue evaluate(Function f);
	
	
	/**
	 * Taking derivative of specified function at this neuron value.
	 * @param f specified function.
	 * @return derivative of specified function at this neuron value.
	 */
	NeuronValue derivative(Function f);
	
	
	/**
	 * Inverse evaluating this neuron value given specified function.
	 * @param f specified function.
	 * @return evaluated value.
	 */
	NeuronValue evaluateInverse(FunctionInvertible f);
	
	
	/**
	 * Taking inverse derivative of specified function at this neuron value.
	 * @param f specified function.
	 * @return derivative of specified function at this neuron value.
	 */
	NeuronValue derivativeInverse(FunctionInvertible f);

	
	/**
	 * Create an array of values.
	 * @param length array length.
	 * @param creator referred creator to create new neuron value.
	 * @return array of values.
	 */
	static NeuronValue[] makeArray(int length, NeuronValueCreator creator) {
		NeuronValue[] array = new NeuronValue[length];
		NeuronValue zero = creator.newNeuronValue().zero();
		for (int j = 0; j < length; j++) array[j] = zero;
		
		return array;
	}


	/**
	 * Adjusting array by length.
	 * @param array specified array.
	 * @param length specified length.
	 * @param creator referred creator to create new neuron value. This parameter can be null if array is not empty.
	 * @return adjusted array.
	 */
	static NeuronValue[] adjustArray(NeuronValue[] array, int length, NeuronValueCreator creator) {
		if (length <= 0) return array;
		
		if (array == null || array.length == 0) {
			array = NeuronValue.makeArray(length, creator);
		}
		else if (array.length < length) {
			int originLength = array.length;
			array = Arrays.copyOfRange(array, 0, length);
			NeuronValue zero = array[0].zero();
			for (int j = originLength; j < length; j++) {
				if (array[j] == null) array[j] = zero;
			}
		}
		return array;
	}


	/**
	 * Concatenating two arrays
	 * @param array1 first array.
	 * @param array2 second array.
	 * @return the array concatenated from the two arrays.
	 */
	static NeuronValue[] concatArray(NeuronValue[] array1, NeuronValue[] array2) {
		if (array1 == null && array2 == null) return null;
		if (array1 != null && array2 == null) return (array1.length != 0 ? array1 : null);
		if (array1 == null && array2 != null) return (array2.length != 0 ? array2 : null);
		if (array1.length == 0 && array2.length == 0) return null;
		if (array1.length > 0 && array2.length == 0) return array1;
		if (array1.length == 0 && array2.length > 0) return array2;
		
		int n = array1.length + array2.length;
		NeuronValue[] array = Arrays.copyOfRange(array1, 0, n);
		for (int i = array1.length; i < n; i++) array[i] = array2[i - array1.length];
		
		return array;
	}
	
	
	/**
	 * Flattening array according to smaller dimension.
	 * @param array specified array.
	 * @param smallerDim smaller dimension.
	 * @return flattened array.
	 */
	static NeuronValue[] flattenByDim(NeuronValue[] array, int smallerDim) {
		if (array == null || array.length == 0)
			return array;
		else
			return array[0].flatten(array, smallerDim);
	}

	
	/**
	 * Flattening array according to smaller channel.
	 * @param array specified array.
	 * @param smallerChannel smaller channel.
	 * @return flattened array.
	 */
	static NeuronValue[] flattenByChannel(NeuronValue[] array, int smallerChannel) {
		if (array == null || array.length == 0)
			return array;
		else if (array[0] instanceof NeuronValueComposite)
			return ((NeuronValueComposite)array[0]).flattenByChannel(array, smallerChannel);
		else
			return array[0].flatten(array, smallerChannel);
	}

	
	/**
	 * Aggregating array according to larger dimension.
	 * @param array specified array.
	 * @param largerDim larger dimension.
	 * @return aggregated array.
	 */
	public static NeuronValue[] aggregateByDim(NeuronValue[] array, int largerDim) {
		if (array == null || array.length == 0 || largerDim <= 0) return array;
		if (largerDim == array[0].dim()) return array;
		if (largerDim < array[0].dim()) {
			NeuronValue[] result = new NeuronValue[array.length];
			for (int i = 0; i < array.length; i++) result[i] = array[i].resize(largerDim);  
			return result;
		}
		
		int step = largerDim / array[0].dim();
		step = step <= array.length ? step : array.length; 
		List<NeuronValue> result = Util.newList(0);
		for (int i = 0; i < array.length; i += step) {
			NeuronValue[] a = RasterAssoc.extractRange1D(NeuronValue.class, array, i, step);
			NeuronValue v = a[0].aggregate(a);
			result.add(v.resize(largerDim));
		}
		return result.toArray(new NeuronValue[] {});
	}

	
	/**
	 * Aggregating array according to larger channel.
	 * @param array specified array.
	 * @param largerChannel larger channel.
	 * @return aggregated array.
	 */
	static NeuronValue[] aggregateByChannel(NeuronValue[] array, int largerChannel) {
		if (array == null || array.length == 0 || largerChannel <= 0) return array;
		if (!(array[0] instanceof NeuronValueComposite)) return aggregateByDim(array, largerChannel);
		NeuronValueComposite firstValue = (NeuronValueComposite)array[0];
		if (largerChannel == firstValue.getNeuronChannel()) return array;
		if (largerChannel < firstValue.getNeuronChannel()) {
			NeuronValue[] result = new NeuronValue[array.length];
			for (int i = 0; i < array.length; i++) result[i] = ((NeuronValueComposite)array[i]).resizeByChannel(largerChannel);  
			return result;
		}

		int step = largerChannel / firstValue.getNeuronChannel();
		step = step <= array.length ? step : array.length; 
		List<NeuronValue> result = Util.newList(0);
		for (int i = 0; i < array.length; i += step) {
			NeuronValueComposite[] a = RasterAssoc.extractRange1D(NeuronValueComposite.class, (NeuronValueComposite[])array, i, step);
			NeuronValue v = a[0].aggregateByChannel(a);
			result.add(v);
		}
		return result.toArray(new NeuronValue[] {});
	}

	
	/**
	 * Transposing matrix.
	 * @param matrix specified matrix.
	 * @return transposed matrix.
	 */
	static NeuronValue[][] transpose(NeuronValue[][] matrix) {
		if (matrix == null) return null;
		int m = matrix.length, n = matrix[0].length;
		if (m == 1 && n == 1) return matrix;
		
		NeuronValue[][] trans = new NeuronValue[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				trans[i][j] = matrix[j][i];
			}
		}
		return trans;
	}
	
	
	/**
	 * Negating matrix.
	 * @param matrix specified matrix.
	 * @return negated matrix.
	 */
	static NeuronValue[][] negative(NeuronValue[][] matrix) {
		if (matrix == null) return null;
		int m = matrix.length, n = matrix[0].length;
		NeuronValue[][] negated = new NeuronValue[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				negated[i][j] = matrix[i][j].negative();
			}
		}
		return negated;
	}
	
	
	/**
	 * Negating vector.
	 * @param vector specified vector.
	 * @return negated vector.
	 */
	static NeuronValue[] negative(NeuronValue[] vector) {
		if (vector == null) return null;
		int n = vector.length;
		NeuronValue[] negated = new NeuronValue[n];
		for (int i = 0; i < n; i++) negated[i] = vector[i].negative();
		return negated;
	}

	
	/**
	 * Checking if specified matrix is invertible.
	 * @param matrix specified square matrix.
	 * @return whether specified matrix is invertible.
	 */
	static boolean isInvertible(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return false;
		else
			return matrix[0][0].matrixIsInvertible(matrix);
	}
	
	
	/**
	 * Calculating determinant of specified matrix.
	 * @param matrix specified square matrix.
	 * @return determinant of specified matrix.
	 */
	static NeuronValue det(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return null;
		else
			return matrix[0][0].matrixDet(matrix);
	}

	
	/**
	 * Calculating inverse of specified matrix.
	 * @param matrix specified square matrix.
	 * @return inverse of specified matrix.
	 */
	static NeuronValue[][] inverse(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return null;
		else
			return matrix[0][0].matrixInverse(matrix);
	}

	
	/**
	 * Calculating square root of specified matrix.
	 * @param matrix specified square matrix.
	 * @return square root of specified matrix.
	 */
	static NeuronValue[][] sqrt(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return null;
		else
			return matrix[0][0].matrixSqrt(matrix);
	}

	
	/**
	 * Adding two matrices.
	 * @param A the first matrix.
	 * @param B the second matrix.
	 * @return sum of the two matrices.
	 */
	static NeuronValue[][] add(NeuronValue[][] A, NeuronValue[][] B) {
		if (A == null || B == null) return null;
		int m = A.length, n = A[0].length;
		NeuronValue[][] C = new NeuronValue[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) C[i][j] = A[i][j].add(B[i][j]);
		}
		return C;
	}

	
	/**
	 * Adding two vectors.
	 * @param v1 the first vector.
	 * @param v2 the second vector.
	 * @return sum of the two matrices.
	 */
	static NeuronValue[] add(NeuronValue[] v1, NeuronValue[] v2) {
		if (v1 == null || v2 == null) return null;
		int n = Math.min(v1.length, v2.length);
		NeuronValue[] v = new NeuronValue[n];
		for (int i = 0; i < n; i++) v[i] = v1[i].add(v2[i]);
		return v;
	}

	
	/**
	 * Subtracting two matrices.
	 * @param A the first matrix.
	 * @param B the second matrix.
	 * @return subtraction of the two matrices.
	 */
	static NeuronValue[][] subtract(NeuronValue[][] A, NeuronValue[][] B) {
		if (A == null || B == null) return null;
		int m = A.length, n = A[0].length;
		NeuronValue[][] C = new NeuronValue[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) C[i][j] = A[i][j].subtract(B[i][j]);
		}
		return C;
	}

	
	/**
	 * Subtracting two vectors.
	 * @param v1 the first vector.
	 * @param v2 the second vector.
	 * @return subtraction of the two matrices.
	 */
	static NeuronValue[] subtract(NeuronValue[] v1, NeuronValue[] v2) {
		if (v1 == null || v2 == null) return null;
		int n = Math.min(v1.length, v2.length);
		NeuronValue[] v = new NeuronValue[n];
		for (int i = 0; i < n; i++) v[i] = v1[i].subtract(v2[i]);
		return v;
	}

	
	/**
	 * Multiplying two matrices.
	 * @param A the first matrix.
	 * @param B the second matrix.
	 * @return product of the two matrices.
	 */
	static NeuronValue[][] multiply(NeuronValue[][] A, NeuronValue[][] B) {
		if (A == null || B == null) return null;
		int m = A.length, n = B.length, o = B[0].length;
		NeuronValue[][] C = new NeuronValue[m][o];
		NeuronValue zero = A[0][0].zero();
		for (int i = 0; i < m; i++) {
			for (int k = 0; k < o; k++) {
				C[i][k] = zero;
				for (int j = 0; j < n; j++) {
					C[i][k] = C[i][k].add(A[i][j].multiply(B[j][k]));
				}
			}
		}
		return C;
	}
	
	
	/**
	 * Multiplying matrix and vector.
	 * @param matrix specific matrix.
	 * @param vector specific vector.
	 * @return applied vector.
	 */
	static NeuronValue[] multiply(NeuronValue[][] matrix, NeuronValue[] vector) {
		if (matrix == null || vector == null) return null;
		NeuronValue[] result = new NeuronValue[matrix.length];
		NeuronValue zero = vector[0].zero();
		for (int i = 0; i < matrix.length; i++) {
			result[i] = zero;
			for (int j = 0; j < matrix[i].length; j++) {
				result[i] = result[i].add(matrix[i][j].multiply(vector[j]));
			}
		}
		return result;
	}

	
	/**
	 * Multiplying vector and matrix.
	 * @param vector specific vector which is row vector.
	 * @param matrix specific matrix.
	 * @return applied vector.
	 */
	static NeuronValue[] multiply(NeuronValue[] vector, NeuronValue[][] matrix) {
		if (vector == null || matrix == null) return null;
		int m = matrix.length, n = matrix[0].length;
		NeuronValue[] result = new NeuronValue[n];
		NeuronValue zero = vector[0].zero();
		for (int j = 0; j < n; j++) {
			result[j] = zero;
			for (int i = 0; i < m; i++) {
				result[j] = result[j].add(matrix[i][j].multiply(vector[i]));
			}
		}
		return result;
	}

	
	/**
	 * Multiplying matrix and value.
	 * @param matrix specific matrix.
	 * @param value specific value.
	 * @return multiplied matrix.
	 */
	static NeuronValue[][] multiply(NeuronValue[][] matrix, NeuronValue value) {
		if (matrix == null || value == null) return null;
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].multiply(value);
			}
		}
		return result;
	}

	
	/**
	 * Multiplying matrix and value.
	 * @param matrix specific matrix.
	 * @param value specific value.
	 * @return multiplied matrix.
	 */
	static NeuronValue[][] multiply(NeuronValue[][] matrix, double value) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix.length == 0) return null;
		return multiply(matrix, matrix[0][0].valueOf(value));
	}
	
	
	/**
	 * Multiplying vector and value.
	 * @param vector specific vector.
	 * @param value specific value.
	 * @return multiplied vector.
	 */
	static NeuronValue[] multiply(NeuronValue[] vector, NeuronValue value) {
		if (vector == null || value == null) return null;
		NeuronValue[] result = new NeuronValue[vector.length];
		for (int i = 0; i < vector.length; i++) result[i] = vector[i].multiply(value);
		return result;
	}

	
	/**
	 * Multiplying vector and value.
	 * @param vector specific vector.
	 * @param value specific value.
	 * @return multiplied vector.
	 */
	static NeuronValue[] multiply(NeuronValue[] vector, double value) {
		if (vector == null || vector.length == 0) return null;
		return multiply(vector, vector[0].valueOf(value));
	}

	
	/**
	 * Dividing matrix and value.
	 * @param matrix specific matrix.
	 * @param value specific value.
	 * @return divided matrix.
	 */
	static NeuronValue[][] divide(NeuronValue[][] matrix, NeuronValue value) {
		if (matrix == null || value == null) return null;
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].divide(value);
			}
		}
		return result;
	}

	
	/**
	 * Dividing matrix and value.
	 * @param matrix specific matrix.
	 * @param value specific value.
	 * @return divided matrix.
	 */
	static NeuronValue[][] divide(NeuronValue[][] matrix, double value) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix.length == 0) return null;
		return divide(matrix, matrix[0][0].valueOf(value));
	}

	
	/**
	 * Product of two vectors.
	 * @param v1 the first vector.
	 * @param v2 the second vector.
	 * @return product of the two vectors.
	 */
	static NeuronValue product(NeuronValue[] v1, NeuronValue[] v2) {
		if (v1 == null || v2 == null || v1.length == 0 || v2.length == 0) return null;
		int n = Math.min(v1.length, v2.length);
		NeuronValue prod = v1[0].zero();
		for (int i = 0; i < n; i++) prod = prod.add(v1[i].multiply(v2[i]));
		return prod;
	}
	
	
	/**
	 * Finding maximum value.
	 * @param vector specified vector.
	 * @return maximum value.
	 */
	static NeuronValue max(NeuronValue[] vector) {
		if (vector == null || vector.length == 0) return null;
		if (vector.length == 1) return vector[0];
		NeuronValue max = vector[0];
		for (int i = 0; i < vector.length; i++) max = max.max(vector[i]);
		return max;
	}
	
	
	/**
	 * Evaluating matrix by function.
	 * @param f specific function.
	 * @param matrix specific matrix.
	 * @return evaluated matrix.
	 */
	static NeuronValue[][] evaluate(Function f, NeuronValue[][] matrix) {
		if (f == null || matrix == null) return null;
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].evaluate(f);
			}
		}
		return result;
	}

	
	/**
	 * Evaluating vector by function.
	 * @param f specific function.
	 * @param vector specific vector.
	 * @return evaluated vector.
	 */
	static NeuronValue[] evaluate(Function f, NeuronValue[] vector) {
		if (f == null || vector == null) return null;
		NeuronValue[] result = new NeuronValue[vector.length];
		for (int i = 0; i < vector.length; i++) result[i] = vector[i].evaluate(f);
		return result;
	}


	/**
	 * Taking derivative of matrix by function.
	 * @param f specific function.
	 * @param matrix specific matrix.
	 * @return derivative matrix.
	 */
	static NeuronValue[][] derivative(Function f, NeuronValue[][] matrix) {
		if (f == null || matrix == null) return null;
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].derivative(f);
			}
		}
		return result;
	}

	
	/**
	 * Taking derivative of vector by function.
	 * @param f specific function.
	 * @param vector specific vector.
	 * @return derivative vector.
	 */
	static NeuronValue[] derivative(Function f, NeuronValue[] vector) {
		if (f == null || vector == null) return null;
		NeuronValue[] result = new NeuronValue[vector.length];
		for (int i = 0; i < vector.length; i++) result[i] = vector[i].derivative(f);
		return result;
	}


	/**
	 * Inverse evaluating matrix by invertible function.
	 * @param f specific invertible function.
	 * @param matrix specific matrix.
	 * @return inverse evaluated matrix.
	 */
	static NeuronValue[][] evaluateInverse(FunctionInvertible f, NeuronValue[][] matrix) {
		if (f == null || matrix == null) return null;
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].evaluateInverse(f);
			}
		}
		return result;
	}

	
	/**
	 * Inverse evaluating vector by invertible function.
	 * @param f specific invertible function.
	 * @param vector specific vector.
	 * @return inverse evaluated vector.
	 */
	static NeuronValue[] evaluateInverse(FunctionInvertible f, NeuronValue[] vector) {
		if (f == null || vector == null) return null;
		NeuronValue[] result = new NeuronValue[vector.length];
		for (int i = 0; i < vector.length; i++) result[i] = vector[i].evaluateInverse(f);
		return result;
	}


	/**
	 * Taking inverse derivative of matrix by invertible function.
	 * @param f specific invertible function.
	 * @param matrix specific matrix.
	 * @return inverse derivative matrix.
	 */
	static NeuronValue[][] derivativeInverse(FunctionInvertible f, NeuronValue[][] matrix) {
		if (f == null || matrix == null) return null;
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = matrix[i][j].derivativeInverse(f);
			}
		}
		return result;
	}

	
	/**
	 * Taking inverse derivative of vector by invertible function.
	 * @param f specific invertible function.
	 * @param vector specific vector.
	 * @return inverse derivative vector.
	 */
	static NeuronValue[] derivativeInverse(FunctionInvertible f, NeuronValue[] vector) {
		if (f == null || vector == null) return null;
		NeuronValue[] result = new NeuronValue[vector.length];
		for (int i = 0; i < vector.length; i++) result[i] = vector[i].derivativeInverse(f);
		return result;
	}


	/**
	 * Calculating norm mean.
	 * @param array array.
	 * @return norm mean.
	 */
	static double normMean(NeuronValue[] array) {
		if (array == null || array.length == 0) return 0;
		double mean = 0;
		for (NeuronValue value : array) mean += value.norm();
		return mean / (double)array.length;
	}
	
	
	/**
	 * Calculating value mean.
	 * @param array array.
	 * @return value mean.
	 */
	static NeuronValue valueMean(NeuronValue[] array) {
		if (array == null || array.length == 0) return null;
		NeuronValue mean = null;
		for (NeuronValue value : array) {
			if (mean == null)
				mean = value;
			else
				mean = mean.add(value);
		}
		return mean.divide((double)array.length);
	}

	
	/**
	 * Calculating norm mean.
	 * @param matrix matrix.
	 * @return norm mean.
	 */
	static double normMean(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0) return 0;
		double mean = 0;
		int count = 0;
		for (NeuronValue[] array : matrix) {
			for (NeuronValue value : array) {
				mean += value.norm();
				count++;
			}
		}
		return mean / (double)count;
	}


	/**
	 * Calculating value mean.
	 * @param matrix matrix.
	 * @return value mean.
	 */
	static NeuronValue valueMean(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0) return null;
		NeuronValue mean = null;
		int count = 0;
		for (NeuronValue[] array : matrix) {
			for (NeuronValue value : array) {
				if (mean == null)
					mean = value;
				else
					mean = mean.add(value);
				count++;
			}
		}
		return mean.divide((double)count);
	}


}
