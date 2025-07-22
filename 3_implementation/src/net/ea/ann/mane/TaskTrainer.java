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

/**
 * This interface specifies trainer applying into learning matrix neural network for specific task such as classification and reinforcement learning.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@FunctionalInterface
public interface TaskTrainer extends Serializable, Cloneable {


	/**
	 * Learning layer as matrix neural network.
	 * @param layer layer as matrix neural network.
	 * @param inouts sample as collection of input and output whose each element is an 2-component array of input (the first) and output (the second).
	 * @param propagate propagation flag.
	 * @param learningRate learning rate.
	 * @return learning biases.
	 */
	Matrix[] train(MatrixLayer layer, Iterable<Matrix[]> inouts, boolean propagate, double learningRate);

	
}
