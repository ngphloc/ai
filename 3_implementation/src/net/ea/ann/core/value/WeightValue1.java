/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.Util;

/**
 * This class represents a scalar weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightValue1 implements WeightValue, TextParsable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zero.
	 */
	private final static WeightValue1 zero = new WeightValue1(0.0);
	
	
	/**
	 * Zero.
	 */
	private final static WeightValue1 unit = new WeightValue1(1.0);

	
	/**
	 * Internal value
	 */
	protected double v = 0.0;

	
	/**
	 * Constructor with double value.
	 * @param v double value.
	 */
	public WeightValue1(double v) {
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
	public WeightValue zero() {
		return zero;
	}
	
	
	@Override
	public WeightValue unit() {
		return unit;
	}

	
	@Override
	public NeuronValue toNeuronValue() {
		return new NeuronValue1(v);
	}


	@Override
	public WeightValue add(NeuronValue value) {
		return new WeightValue1(this.v + ((NeuronValue1)value).get());
	}


	@Override
	public WeightValue subtract(NeuronValue value) {
		return new WeightValue1(this.v - ((NeuronValue1)value).get());
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
	public static WeightValue[][] fromMatrix(double[][] matrix) {
		if (matrix == null) return null;
		
		WeightValue[][] result = new WeightValue[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new WeightValue1[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = new WeightValue1(matrix[i][j]);
			}
		}
		return result;
	}


	/**
	 * Converting value matrix to double matrix.
	 * @param matrix value matrix.
	 * @return double matrix.
	 */
	public static double[][] toMatrix(WeightValue[][] matrix) {
		if (matrix == null) return null;
		
		double[][] result = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = new double[matrix[i].length];
			for (int j = 0; j < matrix[i].length; j++) {
				result[i][j] = ((WeightValue1)(matrix[i][j])).get();
			}
		}
		return result;
	}


}
