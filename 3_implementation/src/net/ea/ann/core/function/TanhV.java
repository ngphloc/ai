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
 * This class represents Tanh function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TanhV extends LogisticV {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with minimum array, maximum array, and slope array.
	 * @param min minimum array.
	 * @param max maximum array.
	 * @param slope slope parameter.
	 */
	public TanhV(double[] min, double[] max, double[] slope) {
		super(min, max, slope);
	}

	
	/**
	 * Constructor with minimum array, maximum array, and slope.
	 * @param min minimum array.
	 * @param max maximum array.
	 * @param slope slope parameter.
	 */
	public TanhV(double[] min, double[] max, double slope) {
		super(min, max, slope);
	}

	
	/**
	 * Constructor with minimum array and maximum array.
	 * @param min minimum array.
	 * @param max maximum array.
	 */
	public TanhV(double[] min, double[] max) {
		this(min, max, 1);
	}

	
	/**
	 * Constructor with dimension, minimum, maximum, and slope.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 * @param slope slope parameter.
	 */
	public TanhV(int dim, double min, double max, double slope) {
		super(dim, min, max, slope);
	}

	
	/**
	 * Constructor with dimension, minimum and maximum.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public TanhV(int dim, double min, double max) {
		this(dim, min, max, 1);
	}

	
	/**
	 * Constructor with dimension.
	 * @param dim specific dimension.
	 */
	public TanhV(int dim) {
		this(dim, 0, 1, 1);
	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		int n = max.length;
		NeuronValueV v = (NeuronValueV)x;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		for (int i = 0; i < n; i++) {
			result.set( i, (max[i]-min[i]) / (1.0 + Math.exp(2*slope[i]*(mid[i]-v.get(i)))) + min[i] );
		}
		
		return result;
	}


	@Override
	public NeuronValue derivative(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)evaluate(x);
		for (int i = 0; i < n; i++) {
			result.set( i, 2*slope[i] * (v.get(i)-min[i]) * (max[i]-v.get(i)) / (max[i]-min[i]) );
		}
		
		return result;
	}


	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)y;
		for (int i = 0; i < n; i++) {
			if (v.get(i) <= min[i] || max[i] <= v.get(i)) return null;
			result.set( i, mid[i] - Math.log((max[i]-v.get(i))/(v.get(i)-min[i])) / (2*slope[i]) );
		}
		
		return result;
	}


	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)y;
		for (int i = 0; i < n; i++) {
			if (v.get(i) <= min[i] || max[i] <= v.get(i)) return null;
			result.set( i, (1/(max[i]-v.get(i)) + 1/(v.get(i)-min[i])) / (2*slope[i]) );
		}
		
		return result;
	}


}
