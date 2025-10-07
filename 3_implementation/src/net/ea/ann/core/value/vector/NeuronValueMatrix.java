/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.vector;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixImpl;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.WeightValue;

/**
 * This class is default implementation of neuron value matrix.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValueMatrix extends MatrixImpl implements NeuronValue {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with numbers of rows and columns along with specified value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified value.
	 */
	public NeuronValueMatrix(int rows, int columns, NeuronValue value) {
		super(rows, columns, value);
	}

	
	/**
	 * Constructor with numbers of rows and columns along with specified double value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified double value.
	 */
	public NeuronValueMatrix(int rows, int columns, double value) {
		this(rows, columns, new NeuronValue1(value));
	}

	
	/**
	 * Constructor with numbers of rows and columns along.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 */
	public NeuronValueMatrix(int rows, int columns) {
		this(rows, columns, 0);
	}

	
	@Override
	public NeuronValue zero() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue unit() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public int length() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public int dim() {
		throw new RuntimeException("Not implemented yet");
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
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue toWeightValue() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue negative() {
		return (NeuronValue)negative0();
	}

	
	@Override
	public NeuronValue abs() {
		return (NeuronValue)abs0();
	}


	@Override
	public boolean canInvert() {
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
	public NeuronValue subtract(NeuronValue value) {
		return (NeuronValue)subtract((Matrix)value);
	}

	
	@Override
	public NeuronValue multiply(NeuronValue value) {
		return (NeuronValue)multiply0(value);
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
		throw new RuntimeException("Not implemented yet");
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
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue derivative(Function f) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		throw new RuntimeException("Not implemented yet");
	}


}
