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

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;

/**
 * This class represents a scalar neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValue1 implements NeuronValue, TextParsable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zero.
	 */
	private final static NeuronValue1 zero = new NeuronValue1(0.0);
	
	
	/**
	 * Zero.
	 */
	private final static NeuronValue1 unit = new NeuronValue1(1.0);

	
	/**
	 * Internal value
	 */
	protected double v = 0.0;
	
	
	/**
	 * Constructor with double value.
	 * @param v double value.
	 */
	public NeuronValue1(double v) {
		this.v = v;
	}

	
	/**
	 * Getting double value.
	 * @return double value.
	 */
	public double get() {
		return v;
	}
	

	@Override
	public NeuronValue zero() {
		return zero;
	}
	
	
	@Override
	public NeuronValue unit() {
		return unit;
	}


	@Override
	public int length() {
		return 1;
	}


	@Override
	public int dim() {
		return length();
	}


	@Override
	public NeuronValue resize(int newDim) {
		if (newDim <= 1) return this;
		NeuronValueV newValue = new NeuronValueV(newDim, 0);
		newValue.v[0] = this.v;
		return newValue;
	}


	@Override
	public NeuronValue duplicate() {
		return new NeuronValue1(this.v);
	}


	@Override
	public boolean equals(NeuronValue value) {
		return this.v == ((NeuronValue1)value).v;
	}


	@Override
	public WeightValue newWeightValue() {
		return new WeightValue1(0.0).zeroW();
	}


	@Override
	public WeightValue toWeightValue() {
		return new WeightValue1(v);
	}

	
	@Override
	public NeuronValue negative() {
		return new NeuronValue1(-this.v);
	}

	
	@Override
	public boolean canInvert() {
		return (this.v != 0);
	}


	@Override
	public NeuronValue inverse() {
		if (this.v != 0)
			return new NeuronValue1(1/this.v);
		else
			return null;
	}

	
	@Override
	public NeuronValue add(NeuronValue value) {
		return new NeuronValue1(this.v + ((NeuronValue1)value).v);
	}

	
	@Override
	public NeuronValue subtract(NeuronValue value) {
		return new NeuronValue1(this.v - ((NeuronValue1)value).v);
	}

	
	@Override
	public NeuronValue multiply(NeuronValue value) {
		return new NeuronValue1(this.v * ((NeuronValue1)value).v);
	}


	@Override
	public NeuronValue multiply(WeightValue value) {
		if (value instanceof WeightValueV) {
			NeuronValueV vector = (NeuronValueV)((WeightValueV)value).toValue();
			int n = vector.length();
			if (n == 0) return null;
			NeuronValueV result = new NeuronValueV(n, 0);
			for (int i = 0; i < n; i++) result.set(i, this.v*vector.get(i));
			return n > 1 ? result : new NeuronValue1(result.get(0));
		}
		else
			return new NeuronValue1(this.v * ((WeightValue1)value).get());
	}

	
	@Override
	public NeuronValue multiply(double value) {
		return new NeuronValue1(this.v * value);
	}

	
	@Override
	public NeuronValue multiplyDerivative(NeuronValue derivative) {
		return multiply(derivative);
	}

	
	@Override
	public NeuronValue divide(NeuronValue value) {
		double v0 = ((NeuronValue1)value).get();
		if (v0 != 0)
			return new NeuronValue1(this.v / v0);
		else
			return null;
	}

	
	@Override
	public NeuronValue divide(double value) {
		if (value != 0)
			return new NeuronValue1(this.v / value);
		else
			return null;
	}

	
	@Override
	public NeuronValue power(double exponent) {
		return new NeuronValue1(Math.pow(this.v, exponent));
	}


	@Override
	public NeuronValue sqrt() {
		return new NeuronValue1(Math.sqrt(this.v));
	}


	@Override
	public NeuronValue exp() {
		return new NeuronValue1(Math.exp(this.v));
	}


	@Override
	public NeuronValue log() {
		return new NeuronValue1(Math.log(this.v));
	}


	@Override
	public double mean() {
		return this.v;
	}


	@Override
	public double norm() {
		return Math.abs(this.v);
	}

	
	@Override
	public NeuronValue valueOf(double value) {
		return new NeuronValue1(value);
	}


	@Override
	public NeuronValue min(NeuronValue value) {
		return new NeuronValue1(Math.min(this.v, ((NeuronValue1)value).get()));
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		return new NeuronValue1(Math.max(this.v, ((NeuronValue1)value).get()));
	}


	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		double[][] dmatrix = toMatrix(matrix);
		if (dmatrix == null) return false;
		return NeuronValueM.isInvertible(dmatrix);
	}


	
	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		double[][] values = toMatrix(matrix);
		if (values == null) return null;
		return valueOf(NeuronValueM.det(values));
	}


	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		double[][] result = toMatrix(matrix);
		if (result == null) return null;
		result = NeuronValueM.inverse(result);
		return fromMatrix(result);
	}


	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		double[][] result = toMatrix(matrix);
		if (result == null) return null;
		result = NeuronValueM.sqrt(result);
		return fromMatrix(result);
	}

	
	@Override
	public NeuronValue[] flatten(int smallerDim) {
		if (smallerDim <= 1) return new NeuronValue[] {this};
		
		NeuronValue[] array = new NeuronValue[smallerDim];
		array[0] = this;
		for (int r = 1; r < smallerDim; r++) array[r] = new NeuronValue1(0);
		return array;
	}


	@Override
	public NeuronValue[] flatten(NeuronValue[] array, int smallerDim) {
		if (array == null || array.length == 0 || smallerDim <= 1) return array;
		
		List<NeuronValue> result = Util.newList(0);
		for (int i = 0; i < array.length; i++) result.addAll(Arrays.asList(array[i].flatten(smallerDim)));
		return result.toArray(new NeuronValue[] {});
	}


	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		if (array == null || array.length == 0) return null;
		List<Double> aggre = Util.newList(0);
		for (NeuronValue value : array) aggre.add(((NeuronValue1)value).v);
		
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
		return Util.format(v);
	}


	/**
	 * Converting double matrix to value matrix.
	 * @param matrix double matrix.
	 * @return value matrix.
	 */
	public static NeuronValue[][] fromMatrix(double[][] matrix) {
		if (matrix == null) return null;
		
		NeuronValue[][] result = new NeuronValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new NeuronValue1[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = new NeuronValue1(matrix[i][j]);
			}
		}
		return result;
	}


	/**
	 * Converting value matrix to double matrix.
	 * @param matrix value matrix.
	 * @return double matrix.
	 */
	public static double[][] toMatrix(NeuronValue[][] matrix) {
		if (matrix == null) return null;
		
		double[][] result = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new double[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = ((NeuronValue1)(matrix[i][j])).get();
			}
		}
		return result;
	}
	

}
