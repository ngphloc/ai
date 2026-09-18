/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.MatrixUtilExt;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.mane.filter.KernelFilter;
import net.ea.ann.mane.train.AdamOptimizer;
import net.ea.ann.mane.train.Optimizer;
import net.ea.ann.mane.weight.NormWeight;
import net.ea.ann.mane.weight.NormWeightMacro;
import net.ea.ann.mane.weight.WeightImpl;

/**
 * This class represent kernel.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Kernel extends Cloneable, Serializable {
	
	
	/**
	 * L2 regularization flag which should be true.
	 */
	final boolean L2 = true;

	
	/**
	 * L2 regularization strength.
	 */
	final double L2_STRENGTH = 1e-4;
	
			
	/**
	 * Optimization flag which should be true.
	 * Improving code related {@link KernelFilter}, {@link WeightImpl}, {@link NormWeight}, {@link NormWeightMacro}.
	 */
	final boolean OPTIMIZER = true; //false
	
	
	/**
	 * Bilinear layers flag which should be true.
	 * If this flag is true, the accuracy may be higher. If this flag is false, sum is always, which make the accuracy stabler but (maybe) lower.
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
	 * Global bias which should be false.
	 */
	final boolean GLOBAL_BIAS = false;
	
	
	/**
	 * Maximum gradient norm for gradient clipping which is a useful technique to improve training neural network.
	 * The value ranges from 0.1 to 1.0 to 5.0 until 10.0. The value 0 indicates no gradient clipping. The value 0.1 is for indivudual gradients. The value 1 is for AdamW optimization. The value 5 for large scale or safe bound.
	 * It is should be 5.0 in this framework.
	 */
	final double GRAD_NORM_MAX_DEFAULT = 0; //0.1, 1.0, 2.0, 5.0, 10.0;


	/**
	 * Speed mode flag which should be true.
	 */
	final boolean SPEED_MODE = true; //false;
	
	
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
	 * Clipping kernel.
	 * Gradient clipping is a useful technique to improve training neural network, which prevents gradient explosion.
	 * @param maxNorm maximum norm.
	 * @return this clipped kernel.
	 */
	default Kernel clip(double maxNorm) {return this;}
	
	
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
	default Kernel copy(Kernel source) {return this;}

	
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
	
	
	/**
	 * Clipping matrices.
	 * @param maxNorm maximum norm.
	 * @param matrices matrices will be clipped.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private static void clipSingular(double maxNorm, Matrix[] matrices) {
		assert (matrices != null && matrices.length > 0 && maxNorm > 0);
		if (maxNorm <= 0) return;
		
		NeuronValue zero = matrices[0].get(0, 0).zero();
		for (int d = 0; d < matrices.length; d++) {
			Matrix matrix = matrices[d];
			if (Kernel.speedMode(zero)) {
				double norm = 0;
				for (int row = 0; row < matrix.rows(); row++) {
					for (int column = 0; column < matrix.columns(); column++) {
						double v = matrix.getv(row, column);
						norm += v*v;
					}
				}
				norm = Math.sqrt(norm);
				if (norm <= maxNorm) continue;

				double scale = maxNorm / norm; //Not necessary to add epsilon because norm is larger than maximum norm and maximum norm is often larger than or equal to 1.
				for (int row = 0; row < matrix.rows(); row++) {
					for (int column = 0; column < matrix.columns(); column++) {
						double value = matrix.getv(row, column);
						matrix.setv(row, column, value*scale);
					}
				}
			}
			else {
				NeuronValue norm = zero;
				for (int row = 0; row < matrix.rows(); row++) {
					for (int column = 0; column < matrix.columns(); column++) {
						NeuronValue v = matrix.get(row, column);
						norm = norm.add(v.multiply(v));
					}
				}
				norm = norm.sqrt();
				if (norm.mean() <= maxNorm || !norm.canInvertWise()) continue;
				
				NeuronValue epsilon = norm.valueOf(1E-12);
				NeuronValue scale = norm.valueOf(maxNorm).divide(norm.add(epsilon));
				for (int row = 0; row < matrix.rows(); row++) {
					for (int column = 0; column < matrix.columns(); column++) {
						NeuronValue value = matrix.get(row, column);
						matrix.set(row, column, value.multiply(scale));
					}
				}
			}
		}
	}

	
	/**
	 * Clipping stacks.
	 * @param maxNorm maximum norm.
	 * @param stacks stacks will be clipped.
	 */
	public static void clip(double maxNorm, MatrixStack[] stacks) {
		assert (stacks != null && stacks.length > 0 && maxNorm > 0);
		if (maxNorm <= 0) return;
		if (maxNorm < 1) {
			int capacity = 0;
			for (MatrixStack stack : stacks) capacity += MatrixUtil.capacity(stack);
			maxNorm = Math.sqrt(maxNorm*maxNorm*capacity);
		}
		
		if (Kernel.speedMode(stacks[0].get().get(0, 0).zero())) {
			double norm = MatrixUtilExt.norm(stacks);
			if (norm <= maxNorm) return;
			
			double scale = maxNorm / norm; //Not necessary to add epsilon because norm is larger than maximum norm and maximum norm is often larger than or equal to 1.
			for (MatrixStack stack : stacks) {
				Matrix[] matrices = stack.matrices();
				for (int d = 0; d < matrices.length; d++) {
					Matrix matrix = matrices[d];
					for (int row = 0; row < matrix.rows(); row++) {
						for (int column = 0; column < matrix.columns(); column++) {
							double value = matrix.getv(row, column);
							matrix.setv(row, column, value*scale);
						}
					}
				}
			}
		}
		else {
			NeuronValue norm = MatrixUtilExt.normV(stacks);
			if (norm.mean() <= maxNorm || !norm.canInvertWise()) return;
			
			NeuronValue epsilon = norm.valueOf(1E-12);
			NeuronValue scale = norm.valueOf(maxNorm).divide(norm.add(epsilon));
			for (MatrixStack stack : stacks) {
				Matrix[] matrices = stack.matrices();
				for (int d = 0; d < matrices.length; d++) {
					Matrix matrix = matrices[d];
					for (int row = 0; row < matrix.rows(); row++) {
						for (int column = 0; column < matrix.columns(); column++) {
							NeuronValue value = matrix.get(row, column);
							matrix.set(row, column, value.multiply(scale));
						}
					}
				}
			}
		}
	}

	
	/**
	 * Clipping matrices.
	 * @param maxNorm maximum norm.
	 * @param matrices matrices will be clipped.
	 */
	public static void clip(double maxNorm, Matrix[] matrices) {
		assert (matrices != null && matrices.length > 0 && maxNorm > 0);
		if (maxNorm <= 0) return;
		if (maxNorm < 1) {
			int capacity = 0;
			for (Matrix matrix : matrices) capacity += MatrixUtil.capacity(matrix);
			maxNorm = Math.sqrt(maxNorm*maxNorm*capacity);
		}
		
		maxNorm *= matrices.length;
		if (Kernel.speedMode(matrices[0].get(0, 0).zero())) {
			double norm = MatrixUtilExt.norm(matrices);
			if (norm <= maxNorm) return;
			
			double scale = maxNorm / norm; //Not necessary to add epsilon because norm is larger than maximum norm and maximum norm is often larger than or equal to 1.
			for (int d = 0; d < matrices.length; d++) {
				Matrix matrix = matrices[d];
				for (int row = 0; row < matrix.rows(); row++) {
					for (int column = 0; column < matrix.columns(); column++) {
						double value = matrix.getv(row, column);
						matrix.setv(row, column, value*scale);
					}
				}
			}
		}
		else {
			NeuronValue norm = MatrixUtilExt.normV(matrices);
			if (norm.mean() <= maxNorm || !norm.canInvertWise()) return;
			
			NeuronValue epsilon = norm.valueOf(1E-12);
			NeuronValue scale = norm.valueOf(maxNorm).divide(norm.add(epsilon));
			for (int d = 0; d < matrices.length; d++) {
				Matrix matrix = matrices[d];
				for (int row = 0; row < matrix.rows(); row++) {
					for (int column = 0; column < matrix.columns(); column++) {
						NeuronValue value = matrix.get(row, column);
						matrix.set(row, column, value.multiply(scale));
					}
				}
			}
		}
	}

	
	/**
	 * Clipping array.
	 * @param maxNorm maximum norm.
	 * @param values values will be clipped.
	 */
	public static void clip(double maxNorm, NeuronValue[] values) {
		assert (values != null && values.length > 0 && maxNorm > 0);
		if (maxNorm <= 0) return;
		if (maxNorm < 1) maxNorm = Math.sqrt(maxNorm*maxNorm*values.length);
		
		if (Kernel.speedMode(values[0])) {
			double norm = MatrixUtilExt.norm(values);
			if (norm <= maxNorm) return;
			
			double scale = maxNorm / norm; //Not necessary to add epsilon because norm is larger than maximum norm and maximum norm is often larger than or equal to 1.
			for (int d = 0; d < values.length; d++) {
				double value = ((NeuronValue1)values[d]).get();
				((NeuronValue1)values[d]).set(value*scale);
			}
		}
		else {
			NeuronValue norm = MatrixUtilExt.normV(values);
			if (norm.mean() <= maxNorm || !norm.canInvertWise()) return;
			
			NeuronValue epsilon = norm.valueOf(1E-12);
			NeuronValue scale = norm.valueOf(maxNorm).divide(norm.add(epsilon));
			for (int d = 0; d < values.length; d++) {
				values[d] = values[d].multiply(scale);
			}
		}
	}

	
	/**
	 * Checking speed mode.
	 * @param hint hinting value.
	 * @return speed mode.
	 */
	static boolean speedMode(NeuronValue hint) {return hint instanceof NeuronValue1 && Kernel.SPEED_MODE;}
	
	
}

