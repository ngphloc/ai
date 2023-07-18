/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.util.List;

/**
 * This class represents a vector neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValueV implements NeuronValue {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
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
		dim = dim < 1? 1 : dim;
		this.v = new double[dim];
		for (int i = 0; i < dim; i++) this.v[i] = initValue;
	}

	
	/**
	 * Constructor with double array.
	 * @param array double array.
	 */
	public NeuronValueV(double...array) {
		this.v = new double[array.length];
		for (int i = 0; i < array.length; i++) this.v[i] = array[i];
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
		return new NeuronValueV(v.length, 0);
	}

	
	@Override
	public NeuronValue identity() {
		return new NeuronValueV(v.length, 1);
	}

	
	/**
	 * Getting length of this vector.
	 * @return length of this vector.
	 */
	public int length() {
		return this.v.length;
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
	public NeuronValue negative() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = -this.v[i];
		
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
		WeightValueV other = (WeightValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] * other.v[i];
		
		return result;
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

	
	@Override
	public NeuronValue sqrt() {
		NeuronValueV result = new NeuronValueV(this.v.length);
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.sqrt(this.v[i]);
		
		return result;
	}

	
	@Override
	public double norm() {
		double ss = 0;
		for (int i = 0; i < this.v.length; i++) ss += this.v[i] * this.v[i];
		
		return Math.sqrt(ss);
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		WeightValueV other = (WeightValueV)value;
		NeuronValueV result = new NeuronValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = Math.max(this.v[i], other.v[i]);
		
		return result;
	}


	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		List<double[][]> matrixList = toDouble(matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<double[][]> inverseList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			double[][] inverse = Util.inverse(matrixList.get(i));
			if (inverse == null || inverse.length == 0) return null;
			
			inverseList.add(inverse);
		}

		return fromDouble(inverseList);
	}
	

	/**
	 * Converting list of double matrices to value matrix.
	 * @param matrixList list of double matrices.
	 * @return value matrix.
	 */
	private static  NeuronValue[][] fromDouble(List<double[][]> matrixList) {
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
	private static List<double[][]> toDouble(NeuronValue[][] matrix) {
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
	

}
