/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.raster.Size;

/**
 * This class represents a matrix neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValueM extends MatrixReal implements NeuronValue, WeightValue {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with data.
	 * @param data data.
	 */
	private NeuronValueM(double[][] data) {
		super(data);
	}
	
	
	/**
	 * Constructor with size and specified value.
	 * @param size size.
	 * @param value specified value.
	 */
	protected NeuronValueM(Size size, double value) {
		super(size, value);
	}

	
	@Override
	public NeuronValue zero() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue zeroW() {
		return (WeightValue)zero();
	}


	@Override
	public NeuronValue unit() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue unitW() {
		return (WeightValue)unit();
	}


	@Override
	public int length() {
		return rows()*columns();
	}

	
	@Override
	public int dim() {
		return rows();
	}

	
	@Override
	public NeuronValue resize(int newDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue duplicate() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public boolean equals(NeuronValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue newWeightValue() {
		return (WeightValue)zero();
	}

	
	@Override
	public WeightValue toWeightValue() {
		return this;
	}

	
	@Override
	public NeuronValue toValue() {
		return this;
	}


	@Override
	public NeuronValue negative() {
		return (NeuronValue)negative0();
	}

	
	@Override
	public NeuronValue abs() {
		int m = data.length, n = data[0].length;
		double[][] abs = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) abs[i][j] = Math.abs(data[i][j]);
		}
		return (NeuronValue)wrap(abs);
	}


	@Override
	public boolean canInvert() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public boolean canInvertWise() {
		throw new RuntimeException("Not implemented yet");
	}


	@Override
	public NeuronValue inverse() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue add(NeuronValue value) {
		return (NeuronValue)add((Matrix)value);
	}

	
	@Override
	public WeightValue addValue(NeuronValue value) {
		return (WeightValue)add(value);
	}


	@Override
	public NeuronValue subtract(NeuronValue value) {
		return (NeuronValue)subtract((Matrix)value);
	}

	
	@Override
	public WeightValue subtractValue(NeuronValue value) {
		return (WeightValue)subtract(value);
	}


	@Override
	public NeuronValue multiply(NeuronValue value) {
		return (NeuronValue)multiply0(value);
	}

	
	@Override
	public NeuronValue multiplyWise(NeuronValue value) {
		return (NeuronValue)multiplyWise((Matrix)value);
	}

	
	@Override
	public NeuronValue multiply(WeightValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue multiply(double value) {
		return (NeuronValue)multiply0(value);
	}

	
	@Override
	public NeuronValue multiplyDerivative(NeuronValue derivative) {
		return derivative.multiplyWise(this);
	}

	
	@Override
	public NeuronValue divide(NeuronValue value) {
		return (NeuronValue)divide0(value);
	}

	
	@Override
	public NeuronValue divide(double value) {
		return (NeuronValue)divide0(value);
	}

	
	@Override
	public NeuronValue power(double exponent) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue sqrt() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue exp() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue log() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public double mean() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public double norm() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue valueOf(double value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue min(NeuronValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[] flatten(int smallerDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[] flatten(NeuronValue[] array, int smallerDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[] aggregate(NeuronValue[] array, int largerDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue evaluate(Function f) {
		return (NeuronValue)evaluate0(f);
	}

	
	@Override
	public NeuronValue derivative(Function f) {
		return (NeuronValueM)derivativeWise(f);
	}

	
	@Override
	public NeuronValue derivativeWiseBy(Function f) {
		return (NeuronValueM)derivativeWise(f);
	}


	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		throw new RuntimeException("Not implemented yet");
	}


	@Override
	protected Matrix wrap(double[][] data) {
		if (data == null || data.length == 0) return null;
		int n = data[0].length;
		if (n == 0) return null;
		for (int i = 1; i < data.length; i++) {
			if (data[i] == null || data[i].length != n) return null;
		}
		return new NeuronValueM(data);
	}


	@Override
	public Matrix create(Size size) {
		if (size.height <= 0 || size.width <= 0)
			return null;
		else
			return new NeuronValueM(size, 0);
	}


	/**
	 * Creating matrix from data array.
	 * @param data data array.
	 * @return matrix.
	 */
	public static Matrix create(double[][] data) {
		return new NeuronValueM(null).wrap(data);
	}

	
}



