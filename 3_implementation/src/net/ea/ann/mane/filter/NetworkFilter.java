/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.Network;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;

/**
 * This class represent network weight.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NetworkFilter extends Filter {

	
	@Override
	default boolean doesApplyActivate() {return false;}


	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef, boolean learning, double learningRate);

	
	@Override
	default Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		return dValue(prevInputLayer, prevOutputLayer, thisErrorLayer, thisActivateRef, true, Network.LEARN_RATE_DEFAULT);
	}


	@Override
	default Kernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		throw new RuntimeException("Network-based filter does not calculate gradient of kernel");
	}


	/**
	 * Updating parameters from backward information.
	 * @param recordCount count of records in sample.
	 * @param learningRate learning rate.
	 */
	void updateParametersFromBackwardInfo(int recordCount, double learningRate);
	
	
	/**
	 * Resetting backward information.
	 */
	void resetBackwardInfo();

		
}
