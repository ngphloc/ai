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
	 * @param sample sample.
	 * @param propagate propagation flag.
	 * @param learningRate learning rate.
	 * @return learning biases.
	 */
	Error[] train(MatrixLayer layer, Iterable<Record> sample, boolean propagate, double learningRate);

	
	/**
	 * Learning layer as matrix neural network.
	 * @param layer layer as matrix neural network.
	 * @param sample sample.
	 * @param learningRate learning rate.
	 * @return learning biases.
	 */
	default Error[] train(MatrixLayer layer, Iterable<Record> sample, double learningRate) {
		return train(layer, sample, true, learningRate);
	}
	
	
}
