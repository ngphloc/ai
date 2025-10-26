/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

/**
 * This interface specifies trainer applying into learning matrix neural network for specific task such as classification and reinforcement learning.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@FunctionalInterface
public interface TaskTrainer {

	
	/**
	 * Learning layer as matrix neural network.
	 * @param layer layer as matrix neural network.
	 * @param inouts sample as collection of input and output.
	 * @param propagate propagation flag.
	 * @param learningRate learning rate.
	 * @return learning biases.
	 */
	Error[] train(MatrixLayer layer, Iterable<Record> inouts, boolean propagate, double learningRate);

	
	/**
	 * Learning layer as matrix neural network.
	 * @param layer layer as matrix neural network.
	 * @param inouts sample as collection of input and output whose each element is an 2-component array of input (the first) and output (the second).
	 * @param learningRate learning rate.
	 * @return learning biases.
	 */
	default Error[] train(MatrixLayer layer, Iterable<Record> inouts, double learningRate) {
		return train(layer, inouts, true, learningRate);
	}
	
	
}
