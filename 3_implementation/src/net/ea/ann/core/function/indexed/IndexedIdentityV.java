/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function.indexed;

import net.ea.ann.core.function.IdentityV;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.indexed.IndexedNeuronValueV;

/**
 * This class represents indexed identity function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IndexedIdentityV extends IdentityV {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public IndexedIdentityV() {

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
