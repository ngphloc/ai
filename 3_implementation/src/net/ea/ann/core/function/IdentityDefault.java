/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents default identity function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IdentityDefault implements Identity {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Getting identity function.
	 * @return identity function.
	 */
	public static IdentityDefault identity() {return new IdentityDefault();}
	
	
	/**
	 * Default constructor.
	 */
	public IdentityDefault() {
		super();
	}


	@Override
	public NeuronValue evaluate(NeuronValue x) {
		return x;
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		return x.unit();
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
