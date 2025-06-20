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
import net.ea.ann.core.value.WeightValue1;

/**
 * This class represents a scalar indexed weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IndexedWeightValue1 implements IndexedWeightValue {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Array of values.
	 */
	private WeightValue1[] values = null;
	
	
	/**
	 * Internal index.
	 */
	private int index = 0;
	
	
	/**
	 * Constructor with size and initial value.
	 * @param size specified size.
	 * @param initialValue initial value.
	 */
	public IndexedWeightValue1(int size, double initialValue) {
		size = size < 1 ? 1 : size;
		values = new WeightValue1[size];
		for (int i = 0; i < values.length; i++) values[i] = new WeightValue1(initialValue);
	}

	
	/**
	 * Constructor with size.
	 * @param size specified size.
	 */
	public IndexedWeightValue1(int size) {
		this(size, 0);
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
		IndexedNeuronValue1 neuronValue = new IndexedNeuronValue1(size, 0);
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
		if (value == null || !(value instanceof WeightValue1)) return null;
		IndexedWeightValue1 newValue = null;
		try {newValue = (IndexedWeightValue1)this.clone();}
		catch (Throwable e) {Util.trace(e);}
		
		if (newValue != null) newValue.values[getIndex()] = (WeightValue1)value;
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
		return values[index];
	}
	
	
	@Override
	public WeightValue set(int index, WeightValue value) {
		if (value == null || !(value instanceof WeightValue1)) return null;
		WeightValue old = get(index);
		values[index] = (WeightValue1)value;
		return old;
	}


}
