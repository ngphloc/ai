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
 * This class represents rectified linear unit (ReLU) function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ReLU1 implements ReLU {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Minimum value.
	 */
	private double min = 0;

	
	/**
	 * Maximum value.
	 */
	private double max = 1;
	
	
	/**
	 * Default constructor.
	 */
	public ReLU1() {

	}

	
	/**
	 * Constructor with minimum and maximum.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public ReLU1(double min, double max) {
		this.min = min;
		this.max = max;
	}

	
	/**
	 * Constructor with minimum.
	 * @param min minimum value.
	 */
	public ReLU1(double min) {
		this.min = min;
		this.max = Double.NaN;
	}

	
	/**
	 * Checking whether to concern maximum.
	 * @return whether to concern maximum.
	 */
	private boolean isConcernMax() {
		return !Double.isNaN(max) && CONCERN_MAX;
	}
	
	
	@Override
	public boolean isNorm() {
		return min == 0 && max == 1;
	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		if (isConcernMax())
			return new NeuronValue1( Math.max(min, Math.min(max, v)) );
		else
			return new NeuronValue1( Math.max(min, v) );
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		if ((v < min) || (isConcernMax() && v > max))
			return new NeuronValue1(0);
		else
			return new NeuronValue1(1);
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
