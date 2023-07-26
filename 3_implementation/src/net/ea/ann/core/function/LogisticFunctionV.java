/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.NeuronValueV;

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
	 * Minimum value.
	 */
	private double[] min;

	
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
		this.min = min;
		this.max = max;
		
		int n = min.length;
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
		this.min = new double[dim];
		this.max = new double[dim];
		this.mid = new double[dim];
		double avg = (min + max) / 2;
		for (int i = 0; i < dim; i++) {
			this.min[i] = min;
			this.max[i] = max;
			this.mid[i] = avg;
		}
	}

	
	/**
	 * Constructor with dimension.
	 * @param dim specific dimension.
	 */
	public LogisticFunctionV(int dim) {
		this.min = new double[dim];
		this.max = new double[dim];
		this.mid = new double[dim];
		
		for (int i = 0; i < dim; i++) {
			this.min[i] = 0;
			this.max[i] = 1;
			this.mid[i] = 0.5;
		}
	}
	
	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)x;
		for (int i = 0; i < n; i++) {
			result.set(i, (max[i]-min[i]) / (1.0 + Math.exp(mid[i]-v.get(i))) + min[i]);
		}
		
		return result;
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)x;
		for (int i = 0; i < n; i++) {
			result.set( i, (max[i]-min[i]) * (v.get(i)-min[i]) * (1-v.get(i)-min[i]) );
		}
		
		return result;
	}
	

}
