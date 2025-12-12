/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;

/**
 * This class represents parametric weight.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Weight extends Cloneable, Serializable {

	
	/**
	 * This class represent kernel.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	 interface Kernel extends Cloneable, Serializable {
		
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

	
	/**
	 * Accumulating kernel.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 * @return this weight.
	 */
	Weight accumKernel(Kernel dKernel, double factor);

		
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param input inputs.
	 * @param bias biases.
	 * @return evaluated value.
	 */
	Matrix evaluate(Matrix input, Matrix bias);

	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevInput previous inputs.
	 * @param prevOutput previous outputs.
	 * @param thisError current errors.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layers.
	 */
	Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef);

	
	/**
	 * Calculating gradient of the current first weight.
	 * @param time time.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of the current first weight.
	 */
	Kernel dKernel(Matrix prevOutput, Matrix thisError);

	
	/**
	 * Filling weight with specified value.
	 * @param v specified value.
	 */
	default void fill(double v) {}
	
	
	/**
	 * Filling weight with randomizer.
	 * @param rnd randomizer.
	 */
	default void fill(Random rnd) {}


	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	default int sizeOfParams() {return 0;}


}


/**
 * This class represents filter specification.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class WeightSpec implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This enum specifies weight type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum Type {
		
		/**
		 * Normal weight.
		 */
		normal,
		
		/**
		 * Transformed-based weight.
		 */
		transformer,
		
	}

	
	/**
	 * Weight type.
	 */
	public Type type = Type.normal;

	
	/**
	 * Default constructor.
	 */
	public WeightSpec() {

	}

	
}
