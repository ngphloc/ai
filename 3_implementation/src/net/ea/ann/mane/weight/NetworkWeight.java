/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Weight;
import net.ea.ann.mane.WeightImpl.WKernel;

/**
 * This class represent network weight.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NetworkWeight extends Weight {

	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevInput previous inputs.
	 * @param prevOutput previous outputs.
	 * @param thisError current errors.
	 * @param prevActivateRef previous activation function.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return gradient of previous layers.
	 */
	Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate);


	@Override
	default WKernel dKernel(Matrix prevOutput, Matrix thisError) {
		throw new RuntimeException("Network-based weight does not calculate gradient of kernel");
	}
	
	
}
