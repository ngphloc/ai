/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.mane.Kernel;

/**
 * This class represents a kernel filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class KernelFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This class represents filter kernel.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class FKernel implements Kernel {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * The weight.
		 */
		protected MatrixStack[] W = null;
		
		/**
		 * Constructor with weight.
		 * @param W weight.
		 */
		public FKernel(MatrixStack[] W) {
			if (!checkValid(W)) throw new IllegalArgumentException();
			this.W = W;
		}

		/**
		 * Checking the weight.
		 * @param W the weight.
		 * @return true if the weight is valid.
		 */
		private static boolean checkValid(MatrixStack[] W) {
			return W != null && W.length > 0;
		}

		/**
		 * Getting width.
		 * @return kernel width.
		 */
		public int width() {
			return W[0].columns();
		}

		/**
		 * Getting kernel height.
		 * @return kernel height.
		 */
		public int height() {
			return W[0].rows();
		}


		/**
		 * Getting kernel depth.
		 * @return kernel depth.
		 */
		public int depth() {
			return W[0].depth();
		}


		/**
		 * Getting kernel time.
		 * @return kernel time.
		 */
		public int time() {
			return W.length;
		}

		@Override
		public FKernel add(Kernel kernel) {
			MatrixStack[] sum = this.W != null ? MatrixStack.sum2(this.W, ((FKernel)kernel).W) : null;
			return new FKernel(sum);
		}

		@Override
		public FKernel multiply(double value) {
			MatrixStack[] d = this.W != null ? MatrixStack.multiply(this.W, value) : null;
			return new FKernel(d);
		}

		@Override
		public FKernel divide(double value) {
			MatrixStack[] d = this.W != null ? MatrixStack.divide(this.W, value) : null;
			return new FKernel(d);
		}

		/**
		 * Calculating sum.
		 * @param kernels kernels.
		 * @return sum.
		 */
		@Deprecated
		private static FKernel sum(FKernel[] kernels) {
			FKernel sum = kernels[0];
			for (int i = 1; i < kernels.length; i++) sum = sum.add(kernels[i]);
			return sum;
		}
		
		/**
		 * Calculating mean.
		 * @param kernels kernels.
		 * @return mean.
		 */
		@SuppressWarnings("unused")
		@Deprecated
		private static FKernel mean(FKernel[] kernels) {
			FKernel sum = sum(kernels);
			return sum.divide(kernels.length);
		}
		
	}

	
	/**
	 * Default constructor.
	 */
	protected KernelFilter() {
		super();
	}

	
}
