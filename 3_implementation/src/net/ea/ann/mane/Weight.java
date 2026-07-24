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
import net.ea.ann.mane.weight.TransformerWeight;
import net.ea.ann.mane.weight.WeightImpl;
import net.ea.ann.mane.weight.WeightNetworkImpl;

/**
 * This class represents parametric weight.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Weight extends Cloneable, Serializable {

	
	/**
	 * Getting back-warding error mode.
	 * If back-warding error mode is true, the weight will be applied into calculating the error of previous layer.
	 * Weight can have false back-warding error mode but filter has always true back-warding error.
	 * The default weight which is {@link WeightImpl} has true back-warding error mode and so true back-warding error mode is usual but
	 * some weights like {@link WeightNetworkImpl} and {@link TransformerWeight} has false back-warding error mode because they focus on their layers.
	 * Anyhow true back-warding error mode is important to connect two successive layers together when it is true.
	 * @return back-warding error mode.
	 */
	default boolean backwardErrorMode( ) {return true;}
	 
	 
	/**
	 * Getting internal weight.
	 * @return internal weight.
	 */
	Kernel kernel();

	
	/**
	 * Accumulating kernel.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 * @return this weight.
	 */
	Weight accumKernel(Kernel dKernel, double factor);

		
	/**
	 * Accumulating kernel for L2 regularization.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 * @param decay decay which is factor of L2 regularization.
	 * @return this weight.
	 */
	default Weight accumKernel(Kernel dKernel, double factor, double decay) {
		return accumKernel(dKernel, factor);
	}
	
	
	/**
	 * Evaluating inputs.
	 * @param input inputs.
	 * @param bias biases.
	 * @return evaluated value.
	 */
	Matrix evaluate(Matrix input, Matrix bias);

	
	/**
	 * Evaluating inputs.
	 * @param input inputs.
	 * @param bias biases.
	 * @param activateRef current activation function.
	 * @return evaluated value.
	 */
	default Matrix evaluate(Matrix input, Matrix bias, Function activateRef) {
		Matrix output = evaluate(input, bias);
		return output != null && activateRef != null ? output.evaluate0(activateRef) : output;
	}
	
	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevOutput previous outputs.
	 * @param thisError current errors.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layers (at previous outputs).
	 */
	Matrix dValue(Matrix prevOutput, Matrix thisError);

	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevInput previous inputs.
	 * @param prevOutput previous outputs.
	 * @param thisError current errors.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layers (at previous outputs).
	 */
	default Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		Matrix dValue = dValue(prevOutput, thisError);
		Matrix derivative = prevInput != null && prevActivateRef != null ? prevInput.derivativeWise(prevActivateRef) : null;
		return derivative != null ? derivative.multiplyWise(dValue) : dValue;
	}
	
	
	/**
	 * Calculating gradient of the current weight.
	 * @param prevOutput previous output. Previous output and this error are in the same current layer.
	 * @param thisError current error.
	 * @return gradient of the current weight (at this error).
	 */
	Kernel dKernel(Matrix prevOutput, Matrix thisError);
	
	
//	/**
//	 * Initializing weight with specified value.
//	 * @param v specified value.
//	 */
//	default void initParams(double v) {}

	
	/**
	 * Filling weight with randomizer.
	 * @param rnd randomizer.
	 */
	default void initParams(Random rnd) {}


	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	default int sizeOfParams() {return 0;}


}



