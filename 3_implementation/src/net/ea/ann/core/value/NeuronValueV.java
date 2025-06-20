/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.util.Collection;
import java.util.List;

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.core.value.vector.NeuronValueVector;
import net.ea.ann.core.value.vector.WeightValueVector;

/**
 * This class represents a vector neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValueV implements NeuronValue, TextParsable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Common zero.
	 */
	private static NeuronValueV zero = null;
	
	
	/**
	 * Array of zeros.
	 */
	private static NeuronValueV[] zeros = new NeuronValueV[MAX_RANGE + 1];
	
	
	/**
	 * Common unit.
	 */
	private static NeuronValueV unit = null;

	
	/**
	 * Array of zeros.
	 */
	private static NeuronValueV[] units = new NeuronValueV[MAX_RANGE + 1];

	
	/**
	 * Static initialization block.
	 */
	static {
		try {
			for (int i = 0; i < zeros.length; i++) {
				try {
					zeros[i] = new NeuronValueV(i, 0);
				} catch (Throwable e) {Util.trace(e);}
			}
		} catch (Throwable e) {Util.trace(e);}
		
		try {
			for (int i = 0; i < units.length; i++) {
				try {
					units[i] = new NeuronValueV(i, 1);
				} catch (Throwable e) {Util.trace(e);}
			}
		} catch (Throwable e) {Util.trace(e);}
	}
	
	
	/**
	 * Internal vector.
	 */
	protected double[] v = null;
	
	
	/**
	 * Constructor with dimension and initial value.
	 * @param dim vector dimension.
	 * @param initValue initial value.
	 */
	public NeuronValueV(int dim, double initValue) {
		this.v = new double[dim < 0? 0 : dim];
		for (int i = 0; i < this.v.length; i++) this.v[i] = initValue;
	}

	
	/**
	 * Constructor with double array.
	 * @param array double array.
	 */
	public NeuronValueV(double...array) {
		this.v = array != null ? new double[array.length] : new double[0];
		for (int i = 0; i < this.v.length; i++) this.v[i] = array[i];
	}
	

	/**
	 * Constructor with values collection.
	 * @param values values collection.
	 */
	public NeuronValueV(Collection<Double> values) {
		this.v = new double[values.size()];
		int i = 0;
		for (double value : values) {
			this.v[i] = value;
			i++;
		}
	}
	
	
	/**
	 * Constructor with dimension.
	 * @param dim vector dimension.
	 */
	public NeuronValueV(int dim) {
		this(dim, 0);
	}

	
	@Override
	public NeuronValue zero() {
		if (zero == this) return zero;
		if (zero != null && zero.v.length == this.v.length) return zero;
		if (this.v.length < zeros.length) {
			zero = zeros[this.v.length];
			zero = zero != null ? zero : new NeuronValueV(this.v.length, 0);
		}
		else
			zero = new NeuronValueV(this.v.length, 0);
		return zero;
	}

	
	@Override
	public NeuronValue unit() {
		if (unit == this) return unit;
		if (unit != null && unit.v.length == this.v.length) return unit;
		if (this.v.length < zeros.length) {
			unit = units[this.v.length];
			unit = unit != null ? unit : new NeuronValueV(this.v.length, 1);
		}
		else
			unit = new NeuronValueV(this.v.length, 1);
		return unit;
	}

	
	@Override
	public int length() {
		return this.v.length;
	}
	

	@Override
	public int dim() {
		return length();
	}


	@Override
	public NeuronValue resize(int newDim) {
		if (newDim == this.v.length) return this;
		if (newDim <= 1) return new NeuronValue1(this.v[0]);
		
		NeuronValueV newValue = new NeuronValueV(newDim, 0);
		int n = Math.min(newValue.v.length, this.v.length);
		for (int i = 0; i < n; i++) newValue.v[i] = this.v[i];
		return newValue;
	}
	
	
	@Override
	public NeuronValue duplicate() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i];
		return result;
	}


	@Override
	public boolean equals(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		for (int i = 0; i < this.v.length; i++) {
			if(this.v[i] != other.v[i]) return false;
		}
		return true;
	}


	/**
	 * Getting value at specific index.
	 * @param index specific index.
	 * @return value at specific index.
	 */
	public double get(int index) {
		return this.v[index];
	}
	

	/**
	 * Getting value at specific index.
	 * @param index specific index.
	 * @param value specific value.
	 */
	public void set(int index, double value) {
		this.v[index] = value;
	}
	
	
	@Override
	public WeightValue newWeightValue() {
		return new WeightValueV(v.length).zero();
	}

	
	@Override
	public WeightValue toWeightValue() {
		WeightValueV result = new WeightValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i];
		return result;
	}

	
	@Override
	public NeuronValue negative() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = -this.v[i];
		return result;
	}

	
	@Override
	public boolean canInvert() {
		if (this.v.length == 0) return false;
		for (int i = 0; i < this.v.length; i++) {
			if (this.v[i] == 0) return false;
		}
		return true;
	}


	@Override
	public NeuronValue inverse() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) {
			if (this.v[i] == 0) return null;
			result.v[i] = 1/this.v[i];
		}
		return result;
	}

	
	@Override
	public NeuronValue add(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] + other.v[i];
		return result;
	}

	
	@Override
	public NeuronValue subtract(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] - other.v[i];
		return result;
	}

	
	@Override
	public NeuronValue multiply(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] * other.v[i];
		return result;
	}

	
	@Override
	public NeuronValue multiply(WeightValue value) {
		if (value instanceof WeightValueVector) {
			NeuronValueVector vector = (NeuronValueVector)((WeightValueVector)value).toNeuronValue();
			int n = vector.length();
			if (n == 0) return null;
			NeuronValueV result = new NeuronValueV(n, 0);
			for (int i = 0; i < n; i++) {
				NeuronValue1 product = this.product((NeuronValueV)vector.get(i));
				result.set(i, product.get());
			}
			return result;
		}
		else {
			WeightValueV other = (WeightValueV)value;
			NeuronValueV result = new NeuronValueV(this.v.length); 
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] * other.v[i];
			return result;
		}
	}

	
	@Override
	public NeuronValue multiply(double value) {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] * value;
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
	NeuronValue1 product(NeuronValueV vector) {
		int n = Math.min(this.length(), vector.length());
		if (n == 0) return null;
		double prod = 0;
		for (int i = 0; i < n; i++) prod += this.v[i] * vector.v[i];
		return new NeuronValue1(prod);
	}
	
	
	@Override
	public NeuronValue divide(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) {
			if (other.v[i] != 0)
				result.v[i] = this.v[i] / other.v[i];
			else
				return null;
		}
		return result;
	}

	
	@Override
	public NeuronValue divide(double value) {
		if (value == 0) return null;
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] / value;
		return result;
	}

	
	@Override
	public NeuronValue power(double exponent) {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.pow(this.v[i], exponent);
		return result;
	}


	@Override
	public NeuronValue sqrt() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.sqrt(this.v[i]);
		return result;
	}

	
	@Override
	public NeuronValue exp() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.exp(this.v[i]);
		return result;
	}


	@Override
	public NeuronValue log() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.log(this.v[i]);
		return result;
	}


	@Override
	public double mean() {
		double sum = 0;
		for (int i = 0; i < this.v.length; i++) sum += this.v[i];
		return sum / this.v.length;
	}


	@Override
	public double norm() {
		double ss = 0;
		for (int i = 0; i < this.v.length; i++) ss += this.v[i] * this.v[i];
		return Math.sqrt(ss);
	}

	
	@Override
	public NeuronValue valueOf(double value) {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = value;
		return result;
	}


	@Override
	public NeuronValue min(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.min(this.v[i], other.v[i]);
		return result;
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.max(this.v[i], other.v[i]);
		return result;
	}


	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		List<double[][]> matrixList = toMatrixList(matrix);
		if (matrixList == null || matrixList.size() == 0) return false;
		for (int i = 0; i < matrixList.size(); i++) {
			boolean invertible = Util.isInvertible(matrixList.get(i));
			if (!invertible) return false;
		}
		return true;
	}


	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		List<double[][]> matrixList = toMatrixList(matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<Double> detList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			double det = Util.det(matrixList.get(i));
			detList.add(det);
		}
		return new NeuronValueV(detList);
	}


	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		List<double[][]> matrixList = toMatrixList(matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<double[][]> inverseList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			double[][] inverse = Util.inverse(matrixList.get(i));
			if (inverse == null || inverse.length == 0) return null;
			
			inverseList.add(inverse);
		}
		return fromMatrixList(inverseList);
	}
	

	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		List<double[][]> matrixList = toMatrixList(matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<double[][]> sqrtList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			double[][] sqrt = Util.sqrt(matrixList.get(i));
			if (sqrt == null || sqrt.length == 0) return null;
			
			sqrtList.add(sqrt);
		}
		return fromMatrixList(sqrtList);
	}

	
	/**
	 * Converting list of double matrices to value matrix.
	 * @param matrixList list of double matrices.
	 * @return value matrix.
	 */
	public static NeuronValue[][] fromMatrixList(List<double[][]> matrixList) {
		if (matrixList == null || matrixList.size() == 0) return null;
		
		int dim = matrixList.size();
		double[][] first = matrixList.get(0);
		NeuronValue[][] matrix = new NeuronValue[first.length][];
		for (int i = 0; i < first.length; i++) {
			matrix[i] = new NeuronValue[first[i].length];
			
			for (int j = 0; j < first[i].length; j++) {
				matrix[i][j] = new NeuronValueV(dim);
				for (int d = 0; d < dim; d++)
					((NeuronValueV)matrix[i][j]).v[d] = matrixList.get(d)[i][j];
			}
		}
		return matrix;
	}

	
	/**
	 * Converting value matrix to list of double matrices.
	 * @param matrix value matrix.
	 * @return list of double matrices.
	 */
	public static List<double[][]> toMatrixList(NeuronValue[][] matrix) {
		if (matrix == null || matrix.length == 0) return null;
		
		int dim = ((NeuronValueV)matrix[0][0]).v.length;
		List<double[][]> matrixList = Util.newList(dim);
		for (int d = 0; d < dim; d++) matrixList.add(new double[matrix.length][]);
		
		for (int i = 0; i < matrix.length; i++) {
			for (int d = 0; d < dim; d++) matrixList.get(d)[i] = new double[matrix[i].length];

			for (int j = 0; j < matrix[i].length; j++) {
				NeuronValueV value = ((NeuronValueV)matrix[i][j]);
				for (int d = 0; d < dim; d++) matrixList.get(d)[i][j] = value.v[d];
			}
		}
		return matrixList;
	}


	@Override
	public NeuronValue[] flatten(int smallerDim) {
		if (smallerDim == this.v.length || smallerDim < 1) return new NeuronValue[] {this};
		if (smallerDim > this.v.length) {
			NeuronValueV result = new NeuronValueV(smallerDim, 0);
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i];
			return new NeuronValue[] {result};
		}
		
		int ratio = this.v.length / smallerDim;
		NeuronValue[] array = new NeuronValue[ratio];
		for (int r = 0; r < ratio; r++) {
			int rIndex = r*smallerDim;
			if (smallerDim > 1) {
				array[r] = new NeuronValueV(smallerDim, 0);
				for (int i = 0; i < smallerDim; i++) ((NeuronValueV)array[r]).v[i] = this.v[rIndex+i];
			}
			else
				array[r] = new NeuronValue1(this.v[rIndex]);
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
				else if (flat[j] instanceof NeuronValue1)
					result[i*ratio + j] = flat[j];
				else if (flat[j] instanceof NeuronValueV)
					result[i*ratio + j] = new NeuronValue1(((NeuronValueV)flat[j]).v[0]);
				else
					result[i*ratio + j] = flat[j];
			}
		}
		return result;
	}

	
	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		if (array == null || array.length == 0) return null;
		List<Double> aggre = Util.newList(0);
		for (NeuronValue value : array) {
			if (value instanceof NeuronValueV) {
				double[] v = ((NeuronValueV)value).v;
				for (int i = 0; i < v.length; i++) aggre.add(v[i]);
			}
			else if (value instanceof NeuronValue1)
				aggre.add(((NeuronValue1)value).v);
		}
		
		if (aggre.size() == 0)
			return null;
		else if (aggre.size() > 1)
			return new NeuronValueV(aggre);
		else
			return new NeuronValue1(aggre.get(0));
	}


	@Override
	public NeuronValue[] aggregate(NeuronValue[] array, int largerDim) {
		return NeuronValue.aggregateByDim(array, largerDim);
	}

	
	@Override
	public NeuronValue evaluate(Function f) {
		return f.evaluate(this);
	}


	@Override
	public NeuronValue derivative(Function f) {
		return f.derivative(this);
	}

	
	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		return f.evaluateInverse(this);
	}


	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		return f.derivativeInverse(this);
	}


	@Override
	public String toText() {
		if (v == null || v.length == 0) return "";
		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < v.length; i++) {
			if ( i > 0) buffer.append(", ");
			buffer.append(Util.format(v[i]));
		}
		return buffer.toString();
	}
	

	/**
	 * Checking whether neuron value vector is finite.
	 * @param value neuron value vector.
	 * @return whether neuron value vector is finite.
	 */
	public static boolean isFinite(NeuronValueV value) {
		if (value == null) return false;
		if (value.length() == 0) return false;
		for (int i = 0; i < value.length(); i++) {
			if (!Double.isFinite(value.get(i))) return false;
		}
		return true;
	}
	
	
	/**
	 * Checking whether neuron value vector is infinite.
	 * @param value neuron value vector.
	 * @return whether neuron value vector is infinite.
	 */
	public static boolean isInfinite(NeuronValueV value) {
		if (value == null) return false;
		if (value.length() == 0) return false;
		for (int i = 0; i < value.length(); i++) {
			if (!Double.isInfinite(value.get(i))) return false;
		}
		return true;
	}

	
}
