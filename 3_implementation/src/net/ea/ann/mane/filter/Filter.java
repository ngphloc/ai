/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.core.value.MatrixStack;

/**
 * This interface represents a filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter extends Serializable, Cloneable {

	
	/**
	 * Flag to calculate error mean.
	 */
	static boolean CALC_ERROR_MEAN = net.ea.ann.conv.filter.Filter.CALC_ERROR_MEAN;
	
	
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
		 * The first weight.
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
	 * Getting filter width.
	 * @return filter width.
	 */
	default int width() {return 1;}

	
	/**
	 * Getting stride width.
	 * @return stride width.
	 */
	default int getStrideWidth() {
		return isMoveStride() ? width() : 1;
	}
	
	
	/**
	 * Getting filter height.
	 * @return filter height.
	 */
	default int height() {return 1;}
	
	
	/**
	 * Getting stride height.
	 * @return stride height.
	 */
	default int getStrideHeight() {
		return isMoveStride() ? height() : 1;
	}
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @return whether to move according to stride when filtering.
	 */
	boolean isMoveStride();
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @param moveStride flag to whether to move according to stride when filtering.
	 */
	void setMoveStride(boolean moveStride);
	
	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	default boolean isPadZero() {return true;}

	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	default void initialize(double v) {}
	
	
	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	default void initialize(Random rnd) {}
	
	
	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	default int sizeOfParams() {return 0;}

	
}



/**
 * This class is an abstract implementation of filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class FilterAbstract implements Filter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to move according to stride when filtering.
	 */
	protected boolean moveStride = true;

	
	/**
	 * Default constructor.
	 */
	protected FilterAbstract() {
		super();
	}

	
	@Override
	public boolean isMoveStride() {
		return moveStride;
	}


	@Override
	public void setMoveStride(boolean moveStride) {
		this.moveStride = moveStride;
	}


}

