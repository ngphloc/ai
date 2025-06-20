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
 * Tanh function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Tanh1 extends Logistic1 {


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
	public Tanh1(double min, double max, double slope) {
		super(min, max, slope);
	}

	
	/**
	 * Constructor with minimum and maximum.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public Tanh1(double min, double max) {
		this(min, max, 1);
	}

	
	/**
	 * Default constructor.
	 */
	public Tanh1() {
		this(0, 1, 1);
	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1((max-min) / (1.0 + Math.exp(2*slope*(mid-v))) + min);
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		double v = ((NeuronValue1)evaluate(x)).get();
		return new NeuronValue1(2*slope * (v-min) * (max-v) / (max-min));
	}


	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		double v = ((NeuronValue1)y).get();
		if (v <= min || max <= v) return null;
		return new NeuronValue1(mid - Math.log((max-v)/(v-min)) / (2*slope));
	}
	
	
	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		double v = ((NeuronValue1)y).get();
		if (v <= min || max <= v) return null;
		return new NeuronValue1((1/(max-v) + 1/(v-min)) / (2*slope));
	}


}
