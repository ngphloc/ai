/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function.indexed;

import net.ea.ann.core.function.ReLUV;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.indexed.IndexedNeuronValueV;

/**
 * This class represents indexed rectified linear unit (ReLU) function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IndexedReLUV extends ReLUV {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with minimum array and maximum array.
	 * @param min minimum array.
	 * @param max maximum array.
	 */
	public IndexedReLUV(double[] min, double[] max) {
		super(min, max);
	}


	/**
	 * Constructor with dim, minimum and maximum.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public IndexedReLUV(int dim, double min, double max) {
		super(dim, min, max);
	}


	/**
	 * Constructor with dimension.
	 * @param dim specific dimension.
	 */
	public IndexedReLUV(int dim) {
		super(dim);
	}


	@Override
	public NeuronValue evaluate(NeuronValue x) {
		if (x == null || !(x instanceof IndexedNeuronValueV)) return null;
		return ((IndexedNeuronValueV)x).re(super.evaluate(((IndexedNeuronValueV)x).v()));
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		if (x == null || !(x instanceof IndexedNeuronValueV)) return null;
		return ((IndexedNeuronValueV)x).re(super.derivative(((IndexedNeuronValueV)x).v()));
	}

	
	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		if (y == null || !(y instanceof IndexedNeuronValueV)) return null;
		return ((IndexedNeuronValueV)y).re(super.evaluateInverse(((IndexedNeuronValueV)y).v()));
	}


	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		if (y == null || !(y instanceof IndexedNeuronValueV)) return null;
		return ((IndexedNeuronValueV)y).re(super.derivativeInverse(((IndexedNeuronValueV)y).v()));
	}


}
