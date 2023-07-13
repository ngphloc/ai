/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

/**
 * Logistic function
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LogisticFunctionScalar implements Function {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public LogisticFunctionScalar() {

	}

	
	@Override
	public Value eval(Value x) {
		double v = ((ValueScalar)x).get();
		return new ValueScalar(1.0 / (1.0 + Math.exp(-v)));
	}


	@Override
	public Value derivative(Value x) {
		double v = ((ValueScalar)x).get();
		return new ValueScalar(v * (1-v));
	}


}
