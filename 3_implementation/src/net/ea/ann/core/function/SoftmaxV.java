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
		return softmaxOne(getAllValues(), x);
	}

	
//	@Override
//	public NeuronValue evaluate(NeuronValue x) {
//		if (x == null) return null;
//		NeuronValue[] all = getAllValues();
//		if (all == null || all.length == 0) return null;
//		
//		NeuronValue max = NeuronValue.max(all);
//		max = max != null ? max : all[0].zero();
//		NeuronValue[] array = new NeuronValue[all.length];
//		boolean finite = true;
//		for (int i = 0; i < array.length; i++) {
//			array[i] = all[i].subtract(max).exp();
//			NeuronValueV v = (NeuronValueV)array[i];
//			if (NeuronValueV.isInfinite(v) || isLargerThanFloatMax(v))
//				array[i] = new NeuronValue1(Float.MAX_VALUE);
//			finite = finite && NeuronValueV.isFinite(v);
//		}
//		if (!finite)
//			return x.valueOf(1.0/(double)all.length);
//		
//		NeuronValue zero = x.zero();
//		NeuronValue sum = zero;
//		for (NeuronValue v : array) sum = sum.add(v);
//		if (sum.equals(zero))
//			return x.valueOf(1.0/(double)all.length);
//
//		NeuronValue xexp = x.subtract(max).exp();
//		NeuronValueV v = (NeuronValueV)xexp;
//		if (NeuronValueV.isInfinite(v) || isLargerThanFloatMax(v))
//			xexp = x.valueOf(Float.MAX_VALUE);
//		
//		NeuronValue value = xexp.divide(sum);
//		return value;
//	}


	/*
	 * Gradient on diagonal.
	 */
	@Override
	public NeuronValue derivative(NeuronValue x) {
		NeuronValue v = evaluate(x);
		if (v == null) return null;
		NeuronValue unit = v.unit();
		return v.multiply(unit.subtract(v));
	}

	
//	/**
//	 * Checking whether neuron value vector is larger than float maximum.
//	 * @param value neuron value vector.
//	 * @return whether neuron value vector is larger than float maximum.
//	 */
//	private static boolean isLargerThanFloatMax(NeuronValueV value) {
//		if (value == null) return false;
//		if (value.length() == 0) return false;
//		for (int i = 0; i < value.length(); i++) {
//			if (value.get(i) <= Float.MAX_VALUE) return false;
//		}
//		return true;
//	}

	
	/**
	 * Calculating softmax of specified value and array. 
	 * @param all array.
	 * @param x value.
	 * @return soft-max of specified value and array.
	 */
	private static NeuronValue softmaxOne(NeuronValue[] all, NeuronValue x) {
		if (x == null || all == null || all.length == 0) return null;
		
		NeuronValue max = NeuronValue.max(all);
		max = max != null ? max : all[0].zero();
		NeuronValue[] exps = new NeuronValue[all.length];
		NeuronValue sum = x.zero();
		for (int i = 0; i < all.length; i++) {
			exps[i] = all[i].subtract(max).exp();
			sum = sum.add(exps[i]);
		}
		
		NeuronValue xexp = x.subtract(max).exp();
		return xexp.divide(sum);
	}

	
	/**
	 * Calculating soft-max of specified array.
	 * @param all array.
	 * @return soft-max of specified array.
	 */
	public static NeuronValue[] softmax(NeuronValue[] all) {
		if (all == null || all.length == 0) return null;
		
		NeuronValue max = NeuronValue.max(all);
		max = max != null ? max : all[0].zero();
		NeuronValue[] exps = new NeuronValue[all.length];
		NeuronValue sum = all[0].zero();
		for (int i = 0; i < all.length; i++) {
			exps[i] = all[i].subtract(max).exp();
			sum = sum.add(exps[i]);
		}
		
		NeuronValue[] softmax = new NeuronValue[all.length];
		for (int i = 0; i < all.length; i++) softmax[i] = exps[i].divide(sum);
		return softmax;
	}
	
	
	/**
	 * Calculating soft-max derivative on diagonal of specified array.
	 * @param all specified array.
	 * @return soft-max derivative of specified array.
	 */
	public static NeuronValue[] softmaxDerivativeDiagonal(NeuronValue[] all) {
		NeuronValue[] softmax = softmax(all);
		if (softmax == null || softmax.length == 0) return null;
		
		NeuronValue unit = all[0].unit();
		NeuronValue[] gradient = new NeuronValue[softmax.length];
		for (int i = 0; i < softmax.length; i++) {
			gradient[i] = softmax[i].multiply(unit.subtract(softmax[i]));
		}
		return gradient;
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
