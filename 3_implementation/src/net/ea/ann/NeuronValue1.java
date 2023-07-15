/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

/**
 * This class represents a scalar neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValue1 implements NeuronValue {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
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
		return new NeuronValue1(0.0);
	}
	
	
	@Override
	public NeuronValue identity() {
		return new NeuronValue1(1.0);
	}


	@Override
	public NeuronValue negative() {
		return new NeuronValue1(-this.v);
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
	public double norm() {
		return Math.abs(this.v);
	}

	
	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}


	@Override
	public NeuronValue sqrt() {
		return new NeuronValue1(Math.sqrt(this.v));
	}


	@Override
	public String toString() {
		return Util.format(v);
	}


}
