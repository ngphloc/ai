/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents soft-max function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SoftmaxV implements Softmax {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * All values
	 */
	private NeuronValue[] allValues = null;
	
	
	/**
	 * Standard layer.
	 */
	private LayerStandard layer = null;
	
	
	/**
	 * Constructor with all values.
	 * @param allValues all values.
	 */
	private SoftmaxV(NeuronValue[] allValues) {
		this.allValues = allValues;
	}

	
	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	private SoftmaxV(LayerStandard layer) {
		this.layer = layer;
	}
	
	
	/**
	 * Getting all values.
	 * @return all values.
	 */
	private NeuronValue[] getAllValues() {
		if (allValues != null && allValues.length > 0)
			return allValues;
		else if (layer != null)
			return layer.getOutput();
		else
			return null;
	}
	
	
	@Override
	public boolean isNorm() {
		return true;
	}


	@Override
	public NeuronValue evaluate(NeuronValue x) {
		return Softmax.softmax(getAllValues(), x);
	}

	
	/*
	 * Derivative of one variable on the diagonal.
	 */
	@Override
	public NeuronValue derivative(NeuronValue x) {
		NeuronValue v = evaluate(x);
		if (v == null) return null;
		NeuronValue unit = v.unit();
		return v.multiply(unit.subtract(v));
	}

	
	/**
	 * Creating soft-max function with all values.
	 * @param allValues all values.
	 * @return soft-max function.
	 */
	public static SoftmaxV create(NeuronValue[] allValues) {
		return allValues != null && allValues.length > 0 ? new SoftmaxV(allValues) : null;
	}
	

	/**
	 * Creating soft-max function with standard layer.
	 * @param layer standard layer.
	 * @return soft-max function.
	 */
	public static SoftmaxV create(LayerStandard layer) {
		return layer != null && layer.size() > 0 ? new SoftmaxV(layer) : null;
	}


	/**
	 * Creating soft-max function with standard neuron.
	 * @param neuron standard neuron.
	 * @return soft-max function.
	 */
	public static SoftmaxV create(NeuronStandard neuron) {
		return neuron != null ? new SoftmaxV(neuron.getLayer()) : null;
	}


}
