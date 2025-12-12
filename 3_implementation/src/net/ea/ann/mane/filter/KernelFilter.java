/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.io.Serializable;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.NeuronValue;

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
	public static class Kernel implements Cloneable, Serializable {
		
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
		public Kernel(MatrixStack[] W) {
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

		
		/**
		 * Adding other kernel.
		 * @param kernel other kernel.
		 * @return sum value.
		 */
		public Kernel add(Kernel kernel) {
			MatrixStack[] sum = this.W != null ? MatrixStack.sum(this.W, kernel.W) : null;
			return new Kernel(sum);
		}
		
		/**
		 * Dividing kernel by value.
		 * @param value value.
		 * @return divided kernel.
		 */
		public Kernel multiply(double value) {
			MatrixStack[] d = this.W != null ? MatrixStack.multiply(this.W, value) : null;
			return new Kernel(d);
		}

		/**
		 * Dividing kernel by value.
		 * @param value value.
		 * @return divided kernel.
		 */
		public Kernel divide(double value) {
			MatrixStack[] d = this.W != null ? MatrixStack.divide(this.W, value) : null;
			return new Kernel(d);
		}

		
		/**
		 * Calculating sum.
		 * @param kernels kernels.
		 * @return sum.
		 */
		public static Kernel sum(Kernel[] kernels) {
			Kernel sum = kernels[0];
			for (int i = 1; i < kernels.length; i++) sum = sum.add(kernels[i]);
			return sum;
		}
		
		/**
		 * Calculating mean.
		 * @param kernels kernels.
		 * @return mean.
		 */
		public static Kernel mean(Kernel[] kernels) {
			Kernel sum = sum(kernels);
			return sum.divide(kernels.length);
		}
		
	}

	
	/**
	 * Default constructor.
	 */
	protected KernelFilter() {
		super();
	}

	
	/**
	 * Accumulating kernel.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 * @return this filter.
	 */
	public abstract ProductFilter accumKernel(Kernel dKernel, double factor);

	
	/**
	 * Forwarding evaluation from previous layer to current layer.
	 * @param time time.
	 * @param prevLayer current layer.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 * @param bias bias.
	 * @param thisActivateRef activation function.
	 */
	public abstract void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef);

	
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	public abstract Kernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef);

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	public abstract Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef);

	
}
