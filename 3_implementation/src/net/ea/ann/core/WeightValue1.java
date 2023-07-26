/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

/**
 * This class represents a scalar weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightValue1 implements WeightValue {


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
		return new WeightValue1(0.0);
	}
	
	
	@Override
	public WeightValue identity() {
		return new WeightValue1(1.0);
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
	public String toString() {
		return Util.format(v);
	}

	
}
