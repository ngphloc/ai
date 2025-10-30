/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.Layer;
import net.ea.ann.core.value.Matrix;

/**
 * This interface represents layer in matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface MatrixLayer extends Layer {

	
	/**
	 * Getting previous layer.
	 * @return previous layer.
	 */
	MatrixLayer getPrevLayer();

	
	/**
	 * Getting next layer.
	 * @return next layer.
	 */
	MatrixLayer getNextLayer();

	
	/**
	 * Getting input value.
	 * @return input value.
	 */
	Matrix getInput();
	
	
	/**
	 * Getting output value.
	 * @return output value.
	 */
	Matrix getOutput();
	
	
	/**
	 * Evaluating layer.
	 * @return evaluated matrix as output.
	 */
	Matrix evaluate();

	
	/**
	 * Evaluating and forwarding layer.
	 * @param inputs specified inputs.
	 * @return evaluated matrix as output.
	 */
	Matrix forward(Record...inputs);
	
	
	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param focus focused layer to stop forwarding.
	 * @param learning learning flag.
	 * @param learningRate learning rate.
	 * @return backward error.
	 */
	Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate);

	
	/**
	 * Backward learning.
	 * @param outputErrors output errors.
	 * @param learningRate learning rate.
	 * @return learning errors.
	 */
	default Error[] backward(Error[] outputErrors, double learningRate) {
		return backward(outputErrors, null, true, learningRate);
	}
	
	
}
