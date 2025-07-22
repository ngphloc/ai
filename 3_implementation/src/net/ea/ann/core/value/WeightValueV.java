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

/**
 * This class represents a vector weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightValueV implements WeightValue, TextParsable {

	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Common zero.
	 */
	private static WeightValueV zero = null;
	
	
	/**
	 * Array of zeros.
	 */
	private static WeightValueV[] zeros = new WeightValueV[NeuronValue.MAX_RANGE + 1];
	
	
	/**
	 * Common unit.
	 */
	private static WeightValueV unit = null;

	
	/**
	 * Array of zeros.
	 */
	private static WeightValueV[] units = new WeightValueV[NeuronValue.MAX_RANGE + 1];

	
	/**
	 * Static initialization block.
	 */
	static {
		try {
			for (int i = 0; i < zeros.length; i++) {
				try {
					zeros[i] = new WeightValueV(i, 0);
				} catch (Throwable e) {Util.trace(e);}
			}
		} catch (Throwable e) {Util.trace(e);}
		
		try {
			for (int i = 0; i < units.length; i++) {
				try {
					units[i] = new WeightValueV(i, 1);
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
	public WeightValueV(int dim, double initValue) {
		this.v = new double[dim < 0 ? 0 : dim];
		for (int i = 0; i < this.v.length; i++) this.v[i] = initValue;
	}

	
	/**
	 * Constructor with double array.
	 * @param array double array.
	 */
	public WeightValueV(double...array) {
		this.v = array != null ? new double[array.length] : new double[0];
		for (int i = 0; i < this.v.length; i++) this.v[i] = array[i];
	}

	
	/**
	 * Constructor with values collection.
	 * @param values values collection.
	 */
	public WeightValueV(Collection<Double> values) {
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
	public WeightValueV(int dim) {
		this(dim, 0);
	}

	
	@Override
	public WeightValue zeroW() {
		if (zero == this) return zero;
		if (zero != null && zero.v.length == this.v.length) return zero;
		if (this.v.length < zeros.length) {
			zero = zeros[this.v.length];
			zero = zero != null ? zero : new WeightValueV(this.v.length, 0);
		}
		else
			zero = new WeightValueV(this.v.length, 0);
		return zero;
	}

	
	@Override
	public WeightValue unitW() {
		if (unit == this) return unit;
		if (unit != null && unit.v.length == this.v.length) return unit;
		if (this.v.length < zeros.length) {
			unit = units[this.v.length];
			unit = unit != null ? unit : new WeightValueV(this.v.length, 1);
		}
		else
			unit = new WeightValueV(this.v.length, 1);
		return unit;
	}

	
	@Override
	public NeuronValue toValue() {
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i];
		return result;
	}


	@Override
	public WeightValue addValue(NeuronValue value) {
		WeightValueV result = new WeightValueV(this.v.length);
		if (value instanceof NeuronValue1) {
			double other = ((NeuronValue1)value).get();
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] + other;
		}
		else {
			NeuronValueV other = (NeuronValueV)value;
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] + other.v[i];
		}
		return result;
	}

	
	@Override
	public WeightValue subtractValue(NeuronValue value) {
		WeightValueV result = new WeightValueV(this.v.length);
		if (value instanceof NeuronValue1) {
			double other = ((NeuronValue1)value).get();
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] - other;
		}
		else {
			NeuronValueV other = (NeuronValueV)value;
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] - other.v[i];
		}
		return result;
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
	 * Converting list of double matrices to value matrix.
	 * @param matrixList list of double matrices.
	 * @return value matrix.
	 */
	public static WeightValue[][] fromMatrixList(List<double[][]> matrixList) {
		if (matrixList == null || matrixList.size() == 0) return null;
		
		int dim = matrixList.size();
		double[][] first = matrixList.get(0);
		WeightValue[][] matrix = new WeightValue[first.length][];
		for (int i = 0; i < first.length; i++) {
			matrix[i] = new WeightValue[first[i].length];
			
			for (int j = 0; j < first[i].length; j++) {
				matrix[i][j] = new WeightValueV(dim);
				for (int d = 0; d < dim; d++)
					((WeightValueV)matrix[i][j]).v[d] = matrixList.get(d)[i][j];
			}
		}
		return matrix;
	}

	
	/**
	 * Converting value matrix to list of double matrices.
	 * @param matrix value matrix.
	 * @return list of double matrices.
	 */
	public static List<double[][]> toMatrixList(WeightValue[][] matrix) {
		if (matrix == null || matrix.length == 0) return null;
		
		int dim = ((WeightValueV)matrix[0][0]).v.length;
		List<double[][]> matrixList = Util.newList(dim);
		for (int d = 0; d < dim; d++) matrixList.add(new double[matrix.length][]);
		
		for (int i = 0; i < matrix.length; i++) {
			for (int d = 0; d < dim; d++) matrixList.get(d)[i] = new double[matrix[i].length];

			for (int j = 0; j < matrix[i].length; j++) {
				WeightValueV value = ((WeightValueV)matrix[i][j]);
				for (int d = 0; d < dim; d++) matrixList.get(d)[i][j] = value.v[d];
			}
		}
		return matrixList;
	}


}
