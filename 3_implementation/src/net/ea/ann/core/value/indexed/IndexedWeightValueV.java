/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.indexed;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.WeightValue;
import net.ea.ann.core.value.WeightValueV;

/**
 * This class represents a vector indexed weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IndexedWeightValueV implements IndexedWeightValue {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Array of values.
	 */
	private WeightValueV[] values = null;

	
	/**
	 * Internal index.
	 */
	private int index = 0;
	
	
	/**
	 * Constructor with size, dimension, and initial value.
	 * @param size specified size.
	 * @param dim specified dimension.
	 * @param initialValue initial value.
	 */
	public IndexedWeightValueV(int size, int dim, double initialValue) {
		size = size < 0 ? 0 : size;
		dim = dim < 0 ? 0 : size;
		values = new WeightValueV[size];
		for (int i = 0; i < values.length; i++) values[i] = new WeightValueV(dim, initialValue);
	}

	
	/**
	 * Constructor with size, dimension, and initial value.
	 * @param size specified size.
	 * @param dim specified dimension.
	 */
	public IndexedWeightValueV(int size, int dim) {
		this(size, dim, 0);
	}
	
	
	@Override
	public WeightValue zero() {
		return re(v().zero());
	}

	
	@Override
	public WeightValue unit() {
		return re(v().unit());
	}

	
	@Override
	public NeuronValue toNeuronValue() {
		int size = this.size();
		IndexedNeuronValueV neuronValue = new IndexedNeuronValueV(size, 1);
		for (int i = 0; i < size; i++) neuronValue.set(i, this.get(i).toNeuronValue());
		neuronValue.setIndex(this.getIndex());
		return neuronValue;
	}

	
	@Override
	public WeightValue add(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().add(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public WeightValue subtract(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().subtract(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public WeightValue v() {
		return values[getIndex()];
	}

	
	/**
	 * Re-indexing and renewing the specified value. 
	 * @param value specified value.
	 * @return re-indexed value.
	 */
	private IndexedWeightValue renew(WeightValue value) {
		if (value == null || !(value instanceof WeightValueV)) return null;
		IndexedWeightValueV newValue = null;
		try {newValue = (IndexedWeightValueV)this.clone();}
		catch (Throwable e) {Util.trace(e);}
		
		if (newValue != null) newValue.values[getIndex()] = (WeightValueV)value;
		return newValue;
	}
	
	
	/**
	 * Re-indexing the specified value. 
	 * @param value specified value.
	 * @return re-indexed value.
	 */
	private IndexedWeightValue re(WeightValue value) {
		return renew(value);
	}


	@Override
	public int getIndex() {
		return index;
	}


	@Override
	public void setIndex(int index) {
		this.index = index;
	}


	@Override
	public int size() {
		return values.length;
	}


	@Override
	public WeightValue get(int index) {
		return values[getIndex()];
	}
	
	
	@Override
	public WeightValue set(int index, WeightValue value) {
		if (value == null || !(value instanceof WeightValueV)) return null;
		WeightValue old = get(index);
		values[index] = (WeightValueV)value;
		return old;
	}


}
