/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.vector;

import java.util.Collection;
import java.util.List;

import net.ea.ann.conv.ContentValue;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueComposite;
import net.ea.ann.core.value.WeightValue;

/**
 * This class is default implementation of neuron value vector.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValueVectorImpl implements NeuronValueVector {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zero.
	 */
	private static NeuronValueVectorImpl zero = null;
	
	
	/**
	 * Zero.
	 */
	private static NeuronValueVectorImpl unit = null;

	
	/**
	 * Internal vector.
	 */
	protected NeuronValue[] v = null;

	
	/**
	 * Zero value.
	 */
	protected NeuronValue zeroValue = null;
	
	
	/**
	 * Constructor with dimension and initial value.
	 * @param dim vector dimension.
	 * @param initValue initial value.
	 */
	public NeuronValueVectorImpl(int dim, NeuronValue initValue) {
		if (dim <= 0) throw new IllegalArgumentException();
		this.v = new NeuronValue[dim];
		for (int i = 0; i < this.v.length; i++) this.v[i] = initValue;
		if (this.v.length > 0) this.zeroValue = initValue.zero();
	}

	
	/**
	 * Constructor with value array.
	 * @param array double array.
	 */
	public NeuronValueVectorImpl(NeuronValue...array) {
		if (array == null || array.length == 0) throw new IllegalArgumentException();
		this.v = new NeuronValue[array.length];
		for (int i = 0; i < this.v.length; i++) this.v[i] = array[i];
		if (this.v.length > 0) this.zeroValue = this.v[0].zero();
	}

	
	/**
	 * Constructor with values collection.
	 * @param values values collection.
	 */
	public NeuronValueVectorImpl(Collection<NeuronValue> values) {
		if (values == null || values.size() == 0) throw new IllegalArgumentException();
		this.v = new NeuronValue[values.size()];
		int i = 0;
		for (NeuronValue value : values) {
			this.v[i] = value;
			i++;
		}
		if (this.v.length > 0) this.zeroValue = this.v[0].zero();
	}

	
	@Override
	public NeuronValue zero() {
		if (zero == this) return zero;
		if (zero != null && zero.v.length == this.v.length && zero.zeroValue == this.zeroValue) return zero;
		zero = new NeuronValueVectorImpl(this.v.length, this.zeroValue);
		return zero;
	}

	
	@Override
	public NeuronValue unit() {
		if (unit == this) return unit;
		if (unit != null && unit.v.length == this.v.length && unit.zeroValue == this.zeroValue) return unit;
		unit = new NeuronValueVectorImpl(this.v.length, this.zeroValue.unit());
		return unit;
	}

	
	@Override
	public int length() {
		return v.length;
	}

	
	@Override
	public int dim() {
		return length();
	}


	@Override
	public int getNeuronChannel() {
		if (v == null || v.length == 0)
			return 0;
		else if (v[0] instanceof NeuronValueComposite)
			return ((NeuronValueComposite)v[0]).getNeuronChannel();
		else
			return v[0].length();
	}


	@Override
	public NeuronValue resize(int newDim) {
		if (newDim == this.v.length) return this;
		if (newDim <= 1) return this.v[0].duplicate();
		
		NeuronValueVectorImpl newValue = new NeuronValueVectorImpl(newDim, this.zeroValue);
		int n = Math.min(newValue.v.length, this.v.length);
		for (int i = 0; i < n; i++) newValue.v[i] = this.v[i];
		return newValue;
	}

	
	@Override
	public NeuronValue resizeByChannel(int newChannel) {
		if (newChannel <= 0 || newChannel == getNeuronChannel()) return this;
		
		NeuronValue[] newValues = new NeuronValue[this.v.length];
		for (int i = 0; i < this.v.length; i++) {
			NeuronValue[] array = new NeuronValue[] {this.v[i]};
			if (newChannel < getNeuronChannel())
				newValues[i] = NeuronValue.flattenByChannel(array, newChannel)[0];
			else if (this.v[i] instanceof NeuronValueComposite)
				newValues[i] = ((NeuronValueComposite)this.v[i]).aggregateByChannel(array);
			else
				newValues[i] = this.v[i].aggregate(array);
		}
		return new NeuronValueVectorImpl(newValues);
	}

	
	@Override
	public NeuronValue duplicate() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, this.zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].duplicate();
		return result;
	}

	
	@Override
	public boolean equals(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		for (int i = 0; i < this.v.length; i++) {
			if(!this.v[i].equals(other.v[i])) return false;
		}
		return true;
	}

	
	@Override
	public NeuronValue get(int index) {
		return v[index];
	}


	@Override
	public NeuronValue set(int index, NeuronValue element) {
		NeuronValue replaced = v[index];
		v[index] = element;
		return replaced;
	}


	@Override
	public WeightValue newWeightValue() {
		return new WeightValueVectorImpl(v.length, zeroValue.newWeightValue()).zeroW();
	}

	
	@Override
	public WeightValue toWeightValue() {
		WeightValueVectorImpl result = new WeightValueVectorImpl(this.v.length, zeroValue.newWeightValue());
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].toWeightValue();
		return result;
	}

	
	@Override
	public NeuronValue negative() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].negative();
		return result;
	}

	
	@Override
	public NeuronValue abs() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].abs();
		return result;
	}


	@Override
	public boolean canInvert() {
		if (this.v.length == 0) return false;
		for (int i = 0; i < this.v.length; i++) {
			if (!this.v[i].canInvert()) return false;
		}
		return true;
	}

	
	@Override
	public boolean canInvertWise() {
		return canInvert();
	}


	@Override
	public NeuronValue inverse() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) {
			if (!this.v[i].canInvert()) return null;
			result.v[i] = this.v[i].inverse();
		}
		return result;
	}

	
	@Override
	public NeuronValue add(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].add(other.v[i]);
		return result;
	}

	
	@Override
	public NeuronValue subtract(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].subtract(other.v[i]);
		return result;
	}

	
	@Override
	public NeuronValue multiply(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].multiply(other.v[i]);
		return result;
	}

	
	@Override
	public NeuronValue multiply(WeightValue value) {
		WeightValueVectorImpl other = (WeightValueVectorImpl)value;
		int n = other.length();
		if (n == 0) return multiply0(value);
		if (!(other.v[0] instanceof WeightValueVectorImpl)) return multiply0(value);
		NeuronValue zeroValue = ((NeuronValueVectorImpl)other.v[0].toValue()).zeroValue;
		if (!zero.getClass().equals(this.zeroValue.getClass())) return multiply0(value);
		
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(n, zeroValue); 
		for (int i = 0; i < n; i++) {
			NeuronValueVector product = (NeuronValueVector)this.product((NeuronValueVector)other.v[i].toValue());
			result.set(i,product);
		}
		return result;
	}

	
	/**
	 * Multiply with other weight value.
	 * @param value other weight value.
	 * @return multiplied value.
	 */
	private NeuronValue multiply0(WeightValue value) {
		WeightValueVectorImpl other = (WeightValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].multiply(other.v[i]);
		return result;
	}
	
	
	@Override
	public NeuronValueVectorImpl multiply(double value) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].multiply(value);
		return result;
	}

	
	@Override
	public NeuronValue multiplyDerivative(NeuronValue derivative) {
		return multiply(derivative);
	}

	
	/**
	 * Calculate dot product of two vectors.
	 * @param vector specified vector.
	 * @return dot product of two vectors.
	 */
	NeuronValue product(NeuronValueVector vector) {
		int n = Math.min(this.length(), vector.length());
		if (n == 0) return null;
		NeuronValue prod = this.zeroValue;
		for (int i = 0; i < n; i++) prod = prod.add(this.v[i].multiply(vector.get(i)));
		return prod;
	}

	
	@Override
	public NeuronValue divide(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue); 
		for (int i = 0; i < this.v.length; i++) {
			if (other.v[i].canInvert())
				result.v[i] = this.v[i].divide(other.v[i]);
			else
				return null;
		}
		return result;
	}

	
	@Override
	public NeuronValue divide(double value) {
		if (value == 0) return null;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].divide(value);
		return result;
	}

	
	@Override
	public NeuronValue power(double exponent) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].power(exponent);
		return result;
	}

	
	@Override
	public NeuronValue sqrt() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].sqrt();
		return result;
	}

	
	@Override
	public NeuronValue exp() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].exp();
		return result;
	}

	
	@Override
	public NeuronValue log() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].log();
		return result;
	}

	
	@Override
	public double mean() {
		double sum = 0;
		for (int i = 0; i < this.v.length; i++) sum += this.v[i].mean();
		return sum / this.v.length;
	}

	
	@Override
	public double norm() {
		double norm = 0;
		for (int i = 0; i < this.v.length; i++) norm += this.v[i].norm();
		return norm / this.v.length;
	}

	
	@Override
	public NeuronValue valueOf(double value) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i].valueOf(value);
		return result;
	}

	
	@Override
	public NeuronValue min(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].min(other.v[i]);
		return result;
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].max(other.v[i]);
		return result;
	}

	
	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((NeuronValueVector[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return false;
		for (int i = 0; i < matrixList.size(); i++) {
			boolean invertible = NeuronValue.isInvertible(matrixList.get(i));
			if (!invertible) return false;
		}
		return true;
	}

	
	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((NeuronValueVector[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		NeuronValueVector detVector = new NeuronValueVectorImpl(length(), zeroValue);
		for (int i = 0; i < detVector.length(); i++) {
			NeuronValue det = NeuronValue.det(matrixList.get(i));
			detVector.set(i, det);
		}
		return detVector;
	}

	
	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((NeuronValueVector[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<NeuronValue[][]> inverseList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			NeuronValue[][] inverse = NeuronValue.inverse(matrixList.get(i));
			if (inverse == null || inverse.length == 0) return null;
			
			inverseList.add(inverse);
		}
		return fromMatrixList(inverseList);
	}

	
	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((NeuronValueVector[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<NeuronValue[][]> sqrtList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			NeuronValue[][] sqrt = NeuronValue.sqrt(matrixList.get(i));
			if (sqrt == null || sqrt.length == 0) return null;
			
			sqrtList.add(sqrt);
		}
		return fromMatrixList(sqrtList);
	}

	
	@Override
	public NeuronValue[] flatten(int smallerDim) {
		if (smallerDim == this.v.length || smallerDim < 1) return new NeuronValue[] {this};
		if (smallerDim > this.v.length) {
			NeuronValueVectorImpl result = new NeuronValueVectorImpl(smallerDim, zeroValue);
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i];
			return new NeuronValue[] {result};
		}
		
		int ratio = this.v.length / smallerDim;
		NeuronValue[] array = new NeuronValue[ratio];
		for (int r = 0; r < ratio; r++) {
			int rIndex = r*smallerDim;
			if (smallerDim > 1) {
				array[r] = new NeuronValueVectorImpl(smallerDim, zeroValue);
				for (int i = 0; i < smallerDim; i++) ((NeuronValueVectorImpl)array[r]).v[i] = this.v[rIndex+i];
			}
			else
				array[r] = this.v[rIndex];
		}
		return array;
	}

	
	@Override
	public NeuronValue[] flatten(NeuronValue[] array, int smallerDim) {
		if (array == null || array.length == 0 || smallerDim < 1) return array;
		if (smallerDim >= array[0].length()) return array;
		
		int ratio = array[0].length() / smallerDim;
		ratio = ratio < 1 ? 1 : ratio;
		NeuronValue[] result = new NeuronValue[ratio*array.length];
		for (int i = 0; i < array.length; i++) {
			NeuronValue[] flat = array[i].flatten(smallerDim);
			for (int j = 0; j < flat.length; j++) {
				if (smallerDim > 1)
					result[i*ratio + j] = flat[j];
				else if (flat[j] instanceof NeuronValueVector)
					result[i*ratio + j] = ((NeuronValueVectorImpl)flat[j]).v[0];
				else
					result[i*ratio + j] = flat[j];
			}
		}
		return result;
	}

	
	@Override
	public NeuronValue[] flattenByChannel(int smallerChannel) {
		if (this.length() == 0 || smallerChannel <= 0) return new NeuronValue[] {this};
		NeuronValue firstValue = this.v[0];
		boolean composite = firstValue instanceof NeuronValueComposite;
		if (composite) return flatten(smallerChannel);
		if (smallerChannel >= firstValue.length()) return new NeuronValue[] {this};
		
		int k = firstValue.flatten(smallerChannel).length;
		List<NeuronValue[]> dataList = Util.newList(k);
		for (int j = 0; j < k; j++) dataList.add(new NeuronValue[this.v.length]);
		
		for (int i = 0; i < this.v.length; i++) {
			NeuronValue[] flatten = this.v[i].flatten(smallerChannel);
			for (int j = 0; j < flatten.length; j++) dataList.get(j)[i] = flatten[j]; //flatten.length = k
		}

		NeuronValue[] flatten = new NeuronValue[k];
		for (int j = 0; j < k; j++) flatten[j] = new NeuronValueVectorImpl(dataList.get(j));
		return flatten;
	}
	
	
	@Override
	public NeuronValue[] flattenByChannel(NeuronValue[] array, int smallerChannel) {
		if (array == null || array.length == 0 || smallerChannel <= 0) return array;
		NeuronValue firstValue = ((ContentValue)array[0]).get(0).getValue();
		boolean composite = firstValue instanceof NeuronValueComposite;
		if (composite) return flatten(array, smallerChannel);
		if (smallerChannel >= firstValue.length()) return array;
		
		int ratio = firstValue.length() / smallerChannel;
		ratio = ratio < 1 ? 1 : ratio;
		NeuronValue[] result = new NeuronValue[ratio*array.length];
		for (int i = 0; i < array.length; i++) {
			NeuronValueVector value = (NeuronValueVector)array[i];
			NeuronValue[] flat = value.flattenByChannel(smallerChannel);
			for (int j = 0; j < flat.length; j++) result[i*ratio + j] = flat[j];
		}
		return result;
	}

	
	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		if (array == null || array.length == 0) return null;
		List<NeuronValue> aggre = Util.newList(0);
		for (NeuronValue value : array) {
			if (value instanceof NeuronValueVector) {
				for (int i = 0; i < v.length; i++) aggre.add(((NeuronValueVector)value).get(i));
			}
			else
				aggre.add(value);
		}
		
		if (aggre.size() == 0)
			return null;
		else if (aggre.size() > 1)
			return new NeuronValueVectorImpl(aggre);
		else
			return aggre.get(0);
	}

	
	@Override
	public NeuronValue[] aggregate(NeuronValue[] array, int largerDim) {
		return NeuronValue.aggregateByDim(array, largerDim);
	}

	
	@Override
	public NeuronValue aggregateByChannel(NeuronValue[] array) {
		if (array == null || array.length == 0) return null;
		
		NeuronValue[] data = new NeuronValue[length()];
		for (int i = 0; i < data.length; i++) {
			NeuronValue[] values = new NeuronValueVector[array.length];
			for (int j = 0; j < array.length; j++) values[j] = ((NeuronValueVector)array[j]).get(i);
			if (values[0] instanceof NeuronValueComposite)
				data[i] = ((NeuronValueComposite)values[0]).aggregateByChannel(values);
			else
				data[i] = values[0].aggregate(values);
		}
		return new NeuronValueVectorImpl(data);
	}


	@Override
	public NeuronValue[] aggregateByChannel(NeuronValue[] array, int largerChannel) {
		return NeuronValue.aggregateByChannel(array, largerChannel);
	}

	
	@Override
	public NeuronValue evaluate(Function f) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, this.zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].evaluate(f);
		return result;
	}


	@Override
	public NeuronValue derivative(Function f) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, this.zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].derivative(f);
		return result;
	}


	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, this.zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].evaluateInverse(f);
		return result;
	}


	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, this.zeroValue);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].derivativeInverse(f);
		return result;
	}


	/**
	 * Converting list of value matrices to value matrix.
	 * @param matrixList list of value matrices.
	 * @return value matrix.
	 */
	public static NeuronValueVector[][] fromMatrixList(List<NeuronValue[][]> matrixList) {
		if (matrixList == null || matrixList.size() == 0) return null;
		
		int dim = matrixList.size();
		NeuronValue[][] first = matrixList.get(0);
		NeuronValue zero = first[0][0].zero();
		NeuronValueVector[][] matrix = new NeuronValueVector[first.length][];
		for (int i = 0; i < first.length; i++) {
			matrix[i] = new NeuronValueVector[first[i].length];
			
			for (int j = 0; j < first[i].length; j++) {
				matrix[i][j] = new NeuronValueVectorImpl(dim, zero);
				for (int d = 0; d < dim; d++) matrix[i][j].set(d, matrixList.get(d)[i][j]);
			}
		}
		return matrix;
	}

	
	/**
	 * Converting value matrix to list of value matrices.
	 * @param matrix value matrix.
	 * @return list of value matrices.
	 */
	public static List<NeuronValue[][]> toMatrixList(NeuronValueVector[][] matrix) {
		if (matrix == null || matrix.length == 0) return null;
		
		int dim = ((NeuronValueVector)matrix[0][0]).length();
		List<NeuronValue[][]> matrixList = Util.newList(dim);
		for (int d = 0; d < dim; d++) matrixList.add(new NeuronValue[matrix.length][]);
		
		for (int i = 0; i < matrix.length; i++) {
			for (int d = 0; d < dim; d++) matrixList.get(d)[i] = new NeuronValue[matrix[i].length];

			for (int j = 0; j < matrix[i].length; j++) {
				NeuronValueVector value = ((NeuronValueVector)matrix[i][j]);
				for (int d = 0; d < dim; d++) matrixList.get(d)[i][j] = value.get(d);
			}
		}
		return matrixList;
	}


}
