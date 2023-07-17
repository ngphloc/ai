/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.function;

import net.ea.ann.NeuronValue;
import net.ea.ann.NeuronValue1;

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
	 * Constructor with maximum and midpoint.
	 */
	public LogisticFunction1(double min, double max) {
		this.max = max;
		this.mid = (min + max) / 2;
	}

	
	@Override
	public NeuronValue eval(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1(max / (1.0 + Math.exp(mid-v)));
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		double v = ((NeuronValue1)x).get();
		return new NeuronValue1(max * v * (1-v));
	}


}
