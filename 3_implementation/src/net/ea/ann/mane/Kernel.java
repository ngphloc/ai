/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

import net.ea.ann.mane.train.AdamOptimizer;
import net.ea.ann.mane.train.Optimizer;

/**
 * This class represent kernel.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Kernel extends Cloneable, Serializable {
	
	
	/**
	 * Optimizer flag.
	 */
	final static boolean OPTIMIZER = true;
	
	
	/**
	 * Always summing flag.
	 */
	final static boolean BILINEAR = true;
	
	
	/**
	 * This class represents null kernel.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class NullKernel implements Kernel {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;
		
		@Override
		public Kernel add(Kernel kernel) {return this;}

		@Override
		public Kernel multiply(double value) {return this;}

		@Override
		public Kernel divide(double value) {return this;}

		@Override
		public Optimizer getOptimizer() {return null;}

		@Override
		public void setOptimizer(Optimizer optimizer) {}
		
	}


	/**
	 * Adding other kernel.
	 * @param kernel other kernel.
	 * @return sum kernel.
	 */
	Kernel add(Kernel kernel);
	
	
	/**
	 * Dividing kernel by value.
	 * @param value value.
	 * @return divided kernel.
	 */
	Kernel multiply(double value);

	
	/**
	 * Dividing kernel by value.
	 * @param value value.
	 * @return divided kernel.
	 */
	Kernel divide(double value);
	
	
	/**
	 * Optimizing kernel itself.
	 * @return kernel itself.
	 */
	default Kernel optimize() {return this;}
	
	
	/**
	 * Getting optimizer.
	 * @return optimizer.
	 */
	Optimizer getOptimizer();
	
	
	/**
	 * Setting optimizer.
	 * @param optimizer optimizer.
	 */
	void setOptimizer(Optimizer optimizer);
	
	
	/**
	 * Create default optimizer.
	 * @return default optimizer.
	 */
	default Optimizer createOptimizer() {return new AdamOptimizer();}
	
	
	/**
	 * Calculating sum.
	 * @param kernels kernels.
	 * @return sum.
	 */
	static Kernel sum(Kernel[] kernels) {
		Kernel sum = kernels[0];
		for (int i = 1; i < kernels.length; i++) sum = sum.add(kernels[i]);
		return sum;
	}
	
	
	/**
	 * Calculating mean.
	 * @param kernels kernels.
	 * @return mean.
	 */
	static Kernel mean(Kernel[] kernels) {
		Kernel sum = sum(kernels);
		return sum.divide(kernels.length);
	}
	
	
}

