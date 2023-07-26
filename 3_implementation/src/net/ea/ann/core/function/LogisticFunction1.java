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
 * Logistic function with scalar variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LogisticFunction1 implements Function {


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
	 * Midpoint.
	 */
	private double mid = 0.5;
	
	
	/**
	 * Default constructor.
	 */
	public LogisticFunction1() {

	}

	
	/**
	 * Constructor with minimum and maximum.
	 */
	public LogisticFunction1(double min, double max) {
		this.min = min;
		this.max = max;
		this.mid = (min + max) / 2;
	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1((max-min) / (1.0 + Math.exp(mid-v)) + min);
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1((max-min) * (v-min) * (1-v-min));
	}


}
