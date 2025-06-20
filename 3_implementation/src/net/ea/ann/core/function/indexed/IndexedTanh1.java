/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function.indexed;

import net.ea.ann.core.function.Tanh1;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.indexed.IndexedNeuronValue1;

/**
 * Indexed Tanh function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IndexedTanh1 extends Tanh1 {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with minimum, maximum, and slope.
	 * @param min minimum value.
	 * @param max maximum value.
	 * @param slope slope value.
	 */
	public IndexedTanh1(double min, double max, double slope) {
		super(min, max, slope);
	}

	
	/**
	 * Constructor with minimum and maximum.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public IndexedTanh1(double min, double max) {
		super(min, max);
	}

	
	/**
	 * Default constructor.
	 */
	public IndexedTanh1() {
		
	}


	@Override
	public NeuronValue evaluate(NeuronValue x) {
		if (x == null || !(x instanceof IndexedNeuronValue1)) return null;
		return ((IndexedNeuronValue1)x).re(super.evaluate(((IndexedNeuronValue1)x).v()));
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		if (x == null || !(x instanceof IndexedNeuronValue1)) return null;
		return ((IndexedNeuronValue1)x).re(super.derivative(((IndexedNeuronValue1)x).v()));
	}

	
	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		if (y == null || !(y instanceof IndexedNeuronValue1)) return null;
		return ((IndexedNeuronValue1)y).re(super.evaluateInverse(((IndexedNeuronValue1)y).v()));
	}


	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		if (y == null || !(y instanceof IndexedNeuronValue1)) return null;
		return ((IndexedNeuronValue1)y).re(super.derivativeInverse(((IndexedNeuronValue1)y).v()));
	}


}
