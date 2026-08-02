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
	 * L2 regularization flag.
	 */
	final boolean L2 = true;

	
	/**
	 * L2 regularization strength.
	 */
	final double L2_STRENGTH = 1e-4;
	
			
	/**
	 * Optimization flag.
	 */
	final boolean OPTIMIZER = true;
	
	
	/**
	 * Bilinear layers flag.
	 * If this flag is true, the accuracy is higher. If this flag is flag, sum is always, which make the accuracy stabler but lower.
	 * The true flag is effective when the number of filters is large enough.
	 */
	final boolean BILINEAR = true; //false;
	
	
	/**
	 * Large depth is defined for GAP and normalization.
	 */
	final int LARGE_DEPTH = 64; //64;


	/**
	 * Large image size is defined for GAP and normalization. If image size is larger than this large size, layer is not normalized.
	 * The number 224 is industrial standard for high resolution threshold image.
	 */
	final int LARGE_SIZE = 224;
	
	
	/**
	 * Global bias.
	 */
	final boolean GLOBAL_BIAS = false;
	
	
	/**
	 * Maximum gradient norm for gradient clipping which is a useful technique to improve training neural network.
	 * The value ranges from 1.0 to 5.0. The value 0 indicates no gradient clipping.
	 */
	final double GRAD_NORM_MAX_DEFAULT = 1.0;


//	/**
//	 * Matrix normalization flag.
//	 */
//	final boolean MATRIX_NORM = true;


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
	default Optimizer getOptimizer() {return null;}
	
	
	/**
	 * Setting optimizer.
	 * @param optimizer optimizer.
	 */
	default void setOptimizer(Optimizer optimizer) {}
	
	
	/**
	 * Create default optimizer.
	 * @return default optimizer.
	 */
	default Optimizer createOptimizer() {return new AdamOptimizer();}
	
	
	/**
	 * Copying from source kernel.
	 * @param source source kernel.
	 */
	default void copyParameters(Kernel source) {}

	
	/**
	 * Calculating decay factor for L2 regularization.
	 * @param learningRate learning rate.
	 * @param recordCount record count.
	 * @return decay factor for L2 regularization.
	 */
	static double decayL2(double learningRate, int recordCount) {
		assert (learningRate > 0 && learningRate <= 1 && recordCount > 0);
		double lambda = L2_STRENGTH; //Regularization strength.
//		recordCount = recordCount < 1 ? 1 : recordCount;
//		return 1.0 - (learningRate * (lambda/recordCount));
		return 1.0 - learningRate*lambda;
	}


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

