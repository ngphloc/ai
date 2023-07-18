/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

/**
 * This class represents a vector weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightValueV implements WeightValue {

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
	public WeightValueV(int dim, double initValue) {
		dim = dim < 1? 1 : dim;
		this.v = new double[dim];
		for (int i = 0; i < dim; i++) this.v[i] = initValue;
	}

	
	/**
	 * Constructor with dimension.
	 * @param dim vector dimension.
	 */
	public WeightValueV(int dim) {
		this(dim, 0);
	}

	
	@Override
	public WeightValue zero() {
		return new WeightValueV(v.length, 0);
	}

	
	@Override
	public WeightValue add(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		WeightValueV result = new WeightValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] + other.v[i];
		
		return result;
	}

	
	@Override
	public WeightValue subtract(NeuronValue value) {
		NeuronValueV other = (NeuronValueV)value;
		WeightValueV result = new WeightValueV(this.v.length); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i] - other.v[i];
		
		return result;
	}

	
}
