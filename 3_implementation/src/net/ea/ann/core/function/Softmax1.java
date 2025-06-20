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
import net.ea.ann.core.value.NeuronValue1;

/**
 * This class represents soft-max function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Softmax1 implements Softmax {


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
	private Softmax1(NeuronValue[] allValues) {
		this.allValues = allValues;
	}

	
	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	private Softmax1(LayerStandard layer) {
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
		if (x == null) return null;
		NeuronValue[] all = getAllValues();
		if (all == null || all.length == 0) return null;
		
		NeuronValue max = NeuronValue.max(all);
		max = max != null ? max : all[0].zero();
		NeuronValue[] array = new NeuronValue[all.length];
		boolean finite = true;
		for (int i = 0; i < array.length; i++) {
			array[i] = all[i].subtract(max).exp();
			double v = ((NeuronValue1)array[i]).get();
			if (Double.isInfinite(v) || v > Float.MAX_VALUE)
				array[i] = new NeuronValue1(Float.MAX_VALUE);
			finite = finite && Double.isFinite(v);
		}
		if (!finite)
			return x.valueOf(1.0/(double)all.length);
		
		NeuronValue zero = x.zero();
		NeuronValue sum = zero;
		for (NeuronValue v : array) sum = sum.add(v);
		if (sum.equals(zero))
			return x.valueOf(1.0/(double)all.length);

		NeuronValue xexp = x.subtract(max).exp();
		double v = ((NeuronValue1)xexp).get();
		if (Double.isInfinite(v) || v > Float.MAX_VALUE)
			xexp = x.valueOf(Float.MAX_VALUE);
		
		NeuronValue1 value = (NeuronValue1)xexp.divide(sum);
		return value.get() > 1 ? value.unit() : value;
	}

	
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
	public static Softmax1 create(NeuronValue[] allValues) {
		return allValues != null && allValues.length > 0 ? new Softmax1(allValues) : null;
	}
	

	/**
	 * Creating soft-max function with standard layer.
	 * @param layer standard layer.
	 * @return soft-max function.
	 */
	public static Softmax1 create(LayerStandard layer) {
		return layer != null && layer.size() > 0 ? new Softmax1(layer) : null;
	}


	/**
	 * Creating soft-max function with standard neuron.
	 * @param neuron standard neuron.
	 * @return soft-max function.
	 */
	public static Softmax1 create(NeuronStandard neuron) {
		return neuron != null ? new Softmax1(neuron.getLayer()) : null;
	}


}
