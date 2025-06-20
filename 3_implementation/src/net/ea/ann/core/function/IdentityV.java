/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueV;

/**
 * This class represents identity function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IdentityV implements Identity {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public IdentityV() {

	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		return x;
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		NeuronValueV v = (NeuronValueV)x;
		return v.unit();
	}


	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		return evaluate(y);
	}


	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		return derivative(y);
	}
	

}
