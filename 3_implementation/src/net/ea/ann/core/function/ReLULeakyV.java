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
 * This class represents leaky rectified linear unit (leaky ReLU) function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ReLULeakyV implements ReLU {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Decreasing factor.
	 */
	protected double alpha = ReLULeaky1.ALPHA;
	
	
	/**
	 * Constructor with decreasing factor.
	 * @param alpha decreasing factor.
	 */
	public ReLULeakyV(double alpha) {
		this.alpha = alpha;
	}

	
	/**
	 * Default constructor.
	 */
	public ReLULeakyV() {
		this(ReLULeaky1.ALPHA);
	}


	@Override
	public boolean isNorm() {return true;}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		NeuronValueV value = (NeuronValueV)x;
		int n = value.length();
		NeuronValueV result = new NeuronValueV(n, 0.0);
		for (int i = 0; i < n; i++) {
			double v = value.get(i);
			result.set(i, v > 0 ? v : (alpha == 0 ? 0 : alpha*v));
		}
		
		return result;
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		NeuronValueV value = (NeuronValueV)x;
		int n = value.length();
		NeuronValueV result = new NeuronValueV(n, 0.0);
		for (int i = 0; i < n; i++) {
			double v = value.get(i);
			result.set(i, v > 0 ? 1 : alpha);
		}
		
		return result;
	}

	
	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		NeuronValueV value = (NeuronValueV)y;
		int n = value.length();
		NeuronValueV result = new NeuronValueV(n, 0.0);
		double factor = alpha != 0 ? 1.0/alpha : 0;
		for (int i = 0; i < n; i++) {
			double v = value.get(i);
			result.set(i, v > 0 ? v : (factor == 0 ? 0 : factor*v));
		}
		
		return result;
	}

	
	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		NeuronValueV value = (NeuronValueV)y;
		int n = value.length();
		NeuronValueV result = new NeuronValueV(n, 0.0);
		double factor = alpha != 0 ? 1.0/alpha : 0;
		for (int i = 0; i < n; i++) {
			double v = value.get(i);
			result.set(i, v > 0 ? 1 : factor);
		}
		
		return result;
	}


}
