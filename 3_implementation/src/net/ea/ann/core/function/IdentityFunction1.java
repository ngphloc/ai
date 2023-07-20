/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.NeuronValue1;

/**
 * This class represents identity function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IdentityFunction1 implements Function {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public IdentityFunction1() {
		super();
	}


	@Override
	public NeuronValue eval(NeuronValue x) {
		return x;
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		return new NeuronValue1(1);
	}

	
}
