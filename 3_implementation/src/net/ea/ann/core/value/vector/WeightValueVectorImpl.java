/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.vector;

import java.util.Collection;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.core.value.WeightValue;

/**
 * This class is default implementation of weight value vector.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightValueVectorImpl implements WeightValueVector {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zero.
	 */
	private static WeightValueVectorImpl zero = null;
	
	
	/**
	 * Zero.
	 */
	private static WeightValueVectorImpl unit = null;

	
	/**
	 * Internal vector.
	 */
	protected WeightValue[] v = null;

	
	/**
	 * Zero value.
	 */
	protected WeightValue zeroValue = null;
	
	
	/**
	 * Constructor with dimension and initial value.
	 * @param dim vector dimension.
	 * @param initValue initial value.
	 */
	public WeightValueVectorImpl(int dim, WeightValue initValue) {
		this.v = new WeightValue[dim < 0 ? 0 : dim];
		for (int i = 0; i < this.v.length; i++) this.v[i] = initValue;
		if (this.v.length > 0) this.zeroValue = initValue.zeroW();
	}

	
	/**
	 * Constructor with value array.
	 * @param array double array.
	 */
	public WeightValueVectorImpl(WeightValue...array) {
		this.v = array != null ? new WeightValue[array.length] : new WeightValue[0];
		for (int i = 0; i < this.v.length; i++) this.v[i] = array[i];
		if (this.v.length > 0) this.zeroValue = this.v[0].zeroW();
	}

	
	/**
	 * Constructor with values collection.
	 * @param values values collection.
	 */
	public WeightValueVectorImpl(Collection<WeightValue> values) {
		this.v = new WeightValue[values.size()];
		int i = 0;
		for (WeightValue value : values) {
			this.v[i] = value;
			i++;
		}
		if (this.v.length > 0) this.zeroValue = this.v[0].zeroW();
	}

	
	@Override
	public WeightValue zeroW() {
		if (zero == this) return zero;
		if (zero != null && zero.v.length == this.v.length && zero.zeroValue == this.zeroValue) return zero;
		zero = new WeightValueVectorImpl(this.v.length, this.zeroValue);
		return zero;
	}

	
	@Override
	public WeightValue unitW() {
		if (unit == this) return unit;
		if (unit != null && unit.v.length == this.v.length && unit.zeroValue == this.zeroValue) return unit;
		unit = new WeightValueVectorImpl(this.v.length, this.zeroValue.unitW());
		return unit;
	}

	
	@Override
	public int length() {
		return v.length;
	}

	
	@Override
	public NeuronValue toValue() {
		NeuronValueVectorImpl result = new NeuronValueVectorImpl(this.v.length, zeroValue.toValue()); 
		for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].toValue();
		return result;
	}

	
	@Override
	public WeightValue addValue(NeuronValue value) {
		WeightValueVectorImpl result = new WeightValueVectorImpl(this.v.length, zeroValue);
		if ((value instanceof NeuronValue1) || (value instanceof NeuronValueV)) {
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].addValue(value);
		}
		else {
			NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].addValue(other.v[i]);
		}
		return result;
	}

	
	@Override
	public WeightValue subtractValue(NeuronValue value) {
		WeightValueVectorImpl result = new WeightValueVectorImpl(this.v.length, zeroValue);
		if ((value instanceof NeuronValue1) || (value instanceof NeuronValueV)) {
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].subtractValue(value);
		}
		else {
			NeuronValueVectorImpl other = (NeuronValueVectorImpl)value;
			for (int i = 0; i < this.v.length; i++) result.v[i] = this.v[i].subtractValue(other.v[i]);
		}
		return result;
	}

	
}
