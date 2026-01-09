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
	 * Getting back-warding error mode.
	 * If back-warding error mode is true, the weight will be applied into calculating the error of previous layer.
	 * @return back-warding error mode.
	 */
	default boolean backwardErrorMode( ) {return true;}
	 
	 
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
	 * Calculating gradient of the current weight.
	 * @param time time.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of the current weight.
	 */
	Kernel dKernel(Matrix prevOutput, Matrix thisError);

	
	/**
	 * Initializing weight with specified value.
	 * @param v specified value.
	 */
	default void initParams(double v) {}
	
	
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



