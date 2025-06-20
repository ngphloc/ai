/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

/**
 * This class represents accumulative mean.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Mean implements Serializable, Cloneable {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Mean value.
	 */
	protected NeuronValue mean = null;
	
	
	/**
	 * Count.
	 */
	protected int count = 1;
	
	
	/**
	 * Private constructor with mean and count.
	 * @param mean mean.
	 * @param count count.
	 */
	private Mean(NeuronValue mean, int count) {
		this.mean = mean;
		this.count = count;
	}
	
	
	/**
	 * Constructor with value.
	 * @param value value.
	 */
	public Mean(NeuronValue value) {
		reset(value);
	}
	
	
	/**
	 * Resetting mean by value.
	 * @param value specified value.
	 * @return this mean.
	 */
	public Mean reset(NeuronValue value) {
		this.mean = value;
		this.count = 1;
		return this;
	}
	
	
	/**
	 * Accumulating this mean.
	 * @param value specified value.
	 * @return accumulated mean.
	 */
	public Mean accum(NeuronValue value) {
		this.mean = this.mean.multiply(count).add(value).divide(count+1);
		this.count++;
		return this;
	}
	
	
	/**
	 * Duplicating mean.
	 * @return duplicated mean.
	 */
	public Mean duplicate() {
		return new Mean(mean.duplicate(), count);
	}

	
	/**
	 * Shallow duplicating.
	 * @return shallowly duplicated mean.
	 */
	public Mean duplicateShallow() {
		return new Mean(mean, count);
	}

	
	/**
	 * Getting mean value.
	 * @return mean value.
	 */
	public NeuronValue getMean() {
		return mean;
	}
	
	
	/**
	 * Getting sum.
	 * @return sum.
	 */
	public NeuronValue getSum() {
		return mean.multiply(count);
	}
	
	
}

