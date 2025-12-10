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
 * This class represents logistic function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LogisticV implements Logistic {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Minimum value.
	 */
	protected double[] min;

	
	/**
	 * Maximum value.
	 */
	protected double[] max;
	
	
	/**
	 * Midpoint.
	 */
	protected double[] mid;
	
	
	/**
	 * Slope parameter.
	 */
	protected double[] slope;
	
	
	/**
	 * Constructor with minimum array, maximum array, and slope array.
	 * @param min minimum array.
	 * @param max maximum array.
	 * @param slope slope parameter.
	 */
	public LogisticV(double[] min, double[] max, double[] slope) {
		if (min == null || max == null || slope == null ||
			min.length == 0 || max.length == 0 || slope.length == 0 ||
			max.length != min.length || slope.length != min.length) throw new IllegalArgumentException();
		this.min = min;
		this.max = max;
		this.slope = slope;
		
		int n = min.length;
		this.mid = new double[n];
		for (int i = 0; i < n; i++) this.mid[i] = Logistic1.calcMid(min[i], max[i]);
	}

	
	/**
	 * Constructor with minimum array, maximum array, and slope.
	 * @param min minimum array.
	 * @param max maximum array.
	 * @param slope slope parameter.
	 */
	public LogisticV(double[] min, double[] max, double slope) {
		this(min, max, NeuronValueV.values(slope, min.length));
	}

	
	/**
	 * Constructor with minimum array and maximum array.
	 * @param min minimum array.
	 * @param max maximum array.
	 */
	public LogisticV(double[] min, double[] max) {
		this(min, max, Logistic1.DEFAULT_SLOPE);
	}

	
	/**
	 * Constructor with dimension, minimum, maximum, and slope.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 * @param slope slope parameter.
	 */
	public LogisticV(int dim, double min, double max, double slope) {
		if (dim <= 0) throw new IllegalArgumentException();
		this.min = new double[dim];
		this.max = new double[dim];
		this.mid = new double[dim];
		this.slope = new double[dim];
		double mid = Logistic1.calcMid(min, max);
		for (int i = 0; i < dim; i++) {
			this.min[i] = min;
			this.max[i] = max;
			this.mid[i] = mid;
			this.slope[i] = slope;
		}
	}

	
	/**
	 * Constructor with dimension, minimum and maximum.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public LogisticV(int dim, double min, double max) {
		this(dim, min, max, Logistic1.DEFAULT_SLOPE);
	}

	
	/**
	 * Constructor with dimension.
	 * @param dim specific dimension.
	 */
	public LogisticV(int dim) {
		this(dim, 0, 1, Logistic1.DEFAULT_SLOPE);
	}
	
	
	@Override
	public boolean isNorm() {
		if (min == null || min.length == 0 || max == null || max.length == 0) return false;
		for (int i = 0; i < min.length; i++) {
			if (min[i] != 0 || max[i] != 1) return false;
		}
		return true;
	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		int n = min.length;
		NeuronValueV v = (NeuronValueV)x;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		for (int i = 0; i < n; i++) {
			result.set( i, (max[i]-min[i]) / (1.0 + Math.exp(slope[i]*(mid[i]-v.get(i)))) + min[i] );
		}
		return result;
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		int n = min.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)evaluate(x);
		for (int i = 0; i < n; i++) {
			result.set( i, slope[i] * (v.get(i)-min[i]) * (max[i]-v.get(i)) / (max[i]-min[i]) );
		}
		return result;
	}
	

	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		int n = min.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)y;
		for (int i = 0; i < n; i++) {
			if (v.get(i) <= min[i] || max[i] <= v.get(i)) return null;
			result.set( i, mid[i] - Math.log((max[i]-v.get(i))/(v.get(i)-min[i])) / slope[i] );
		}
		
		return result;
	}

	
	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		int n = min.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)y;
		for (int i = 0; i < n; i++) {
			if (v.get(i) <= min[i] || max[i] <= v.get(i)) return null;
			result.set( i, (1/(max[i]-v.get(i)) + 1/(v.get(i)-min[i])) / slope[i] );
		}
		return result;
	}


}
