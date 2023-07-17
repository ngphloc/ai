/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.function;

import net.ea.ann.NeuronValue;
import net.ea.ann.NeuronValueV;

/**
 * Logistic function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LogisticFunctionV implements Function {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Maximum value.
	 */
	private double[] max;
	
	
	/**
	 * Midpoint.
	 */
	private double[] mid;
	
	
	/**
	 * Constructor with minimum array and maximum array.
	 * @param min minimum array.
	 * @param max maximum array.
	 */
	public LogisticFunctionV(double[] min, double[] max) {
		this.max = max;
		
		int n = max.length;
		this.mid = new double[n];
		for (int i = 0; i < n; i++) this.mid[i] = (min[i] + max[i]) / 2;
	}

	
	/**
	 * Constructor with dim, minimum and maximum.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public LogisticFunctionV(int dim, double min, double max) {
		this.max = new double[dim];
		this.mid = new double[dim];
		double avg = (min + max) / 2;
		for (int i = 0; i < dim; i++) {
			this.max[i] = max;
			this.mid[i] = avg;
		}
	}

	
	/**
	 * Constructor with dimension.
	 * @param dim specific dimension.
	 */
	public LogisticFunctionV(int dim) {
		this.max = new double[dim];
		this.mid = new double[dim];
		
		for (int i = 0; i < dim; i++) {
			this.max[i] = 1;
			this.mid[i] = 0.5;
		}
	}
	
	
	@Override
	public NeuronValue eval(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)x;
		for (int i = 0; i < n; i++) {
			result.set( i, max[i] / (1.0 + Math.exp(mid[i]-v.get(i))) );
		}
		
		return result;
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)x;
		for (int i = 0; i < n; i++) {
			result.set( i, max[i] * v.get(i) * (1-v.get(i)) );
		}
		
		return result;
	}
	

}
