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
 * Logistic function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Logistic1 implements Logistic {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Minimum value.
	 */
	protected double min = 0;

	
	/**
	 * Maximum value.
	 */
	protected double max = 1;
	
	
	/**
	 * Midpoint.
	 */
	protected double mid = CONCERN_MID ? 0.5 : 0;
	
	
	/**
	 * Slope parameter.
	 */
	protected double slope = DEFAULT_SLOPE;
	
	
	/**
	 * Constructor with minimum, maximum, and slope.
	 * @param min minimum value.
	 * @param max maximum value.
	 * @param slope slope value.
	 */
	public Logistic1(double min, double max, double slope) {
		this.min = min;
		this.max = max;
		this.mid = calcMid(min, max);
		this.slope = slope;
	}

	
	/**
	 * Constructor with minimum and maximum.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public Logistic1(double min, double max) {
		this(min, max, DEFAULT_SLOPE);
	}

	
	/**
	 * Default constructor.
	 */
	public Logistic1() {
		this(0, 1, DEFAULT_SLOPE);
	}

	
	@Override
	public boolean isNorm() {
		return min == 0 && max == 1;
	}


	/**
	 * Calculating middle value from minimum and maximum.
	 * @param min minimum.
	 * @param max maximum.
	 * @return middle value from minimum and maximum.
	 */
	static double calcMid(double min, double max) {
		if (!CONCERN_MID) return 0;
		double bias = Math.abs(max-min) / 2;
		if (bias <= 1)
			return (min+max) / 2;
		else
			return 0;
	}
	
	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1((max-min) / (1.0 + Math.exp(slope*(mid-v))) + min);
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		double v = ((NeuronValue1)evaluate(x)).get();
		return new NeuronValue1(slope * (v-min) * (max-v) / (max-min));
	}


	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		double v = ((NeuronValue1)y).get();
		if (v <= min || max <= v) return null;
		return new NeuronValue1(mid - Math.log((max-v)/(v-min)) / slope);
	}


	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		double v = ((NeuronValue1)y).get();
		if (v <= min || max <= v) return null;
		return new NeuronValue1((1/(max-v) + 1/(v-min)) / slope);
	}


}
