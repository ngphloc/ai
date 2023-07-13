/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

/**
 * This class represents a scalar value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ValueScalar implements Value {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal value
	 */
	protected double v = 0;
	
	
	/**
	 * Constructor with double value.
	 * @param v double value.
	 */
	public ValueScalar(double v) {
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
	public Value add(Value value) {
		return new ValueScalar(this.v + ((ValueScalar)value).v);
	}

	
	@Override
	public Value subtract(Value value) {
		return new ValueScalar(this.v - ((ValueScalar)value).v);
	}

	
	@Override
	public Value multiply(Value value) {
		return new ValueScalar(this.v * ((ValueScalar)value).v);
	}

	
	@Override
	public Value multiply(double value) {
		return new ValueScalar(this.v * value);
	}

	
	@Override
	public Value divide(Value value) {
		return new ValueScalar(this.v / ((ValueScalar)value).v);
	}


	@Override
	public double norm() {
		return Math.abs(this.v);
	}

	
}
