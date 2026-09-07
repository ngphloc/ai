/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;

/**
 * This class represents leaky rectified linear unit (leaky ReLU) function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ReLULeaky1 implements ReLU {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Decreasing factor.
	 */
	final static double ALPHA = 0 ;//0.01;
	
	
	/**
	 * Decreasing factor.
	 */
	protected double alpha = ALPHA;
	
	
	/**
	 * Constructor with decreasing factor.
	 * @param alpha decreasing factor.
	 */
	public ReLULeaky1(double alpha) {
		this.alpha = alpha;
	}

	
	/**
	 * Default constructor.
	 */
	public ReLULeaky1() {
		this(ALPHA);
	}

	
	@Override
	public boolean isNorm() {return true;}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1(v > 0 ? v : (alpha == 0 ? 0 : alpha*v));
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1(v > 0 ? 1 : alpha);
	}

	
	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		double v = ((NeuronValue1)y).get();
		double factor = alpha != 0 ? 1.0/alpha : 0;
		return new NeuronValue1(v > 0 ? v : (factor == 0 ? 0 : factor*v));
	}

	
	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		double v = ((NeuronValue1)y).get();
		double factor = alpha != 0 ? 1.0/alpha : 0;
		return new NeuronValue1(v > 0 ? 1 : factor);
	}


}
