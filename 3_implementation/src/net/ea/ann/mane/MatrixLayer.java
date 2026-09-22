/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.util.Arrays;

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
	 * @param params additional parameters.
	 * @return evaluated matrix as output.
	 */
	Matrix evaluate(Object...params);

	
	/**
	 * Evaluating and forwarding layer.
	 * @param inputs specified inputs.
	 * @return evaluated matrix as output.
	 */
	Matrix forward(Record...inputs);
	
	
	/**
	 * Back-warding layer as learning matrix neural network.
	 * This method is the core of matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param focus focused layer to stop back-warding.
	 * @param learning learning flag. If it is false, parameters are not updated (learned).
	 * @param learningRate learning rate.
	 * @param params additional parameters.
	 * @return backward error.
	 */
	Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate, Object...params);

	
	/**
	 * Backward learning.
	 * @param outputErrors output errors.
	 * @param learningRate learning rate.
	 * @param params additional parameters.
	 * @return learning errors.
	 */
	default Error[] backward(Error[] outputErrors, double learningRate, Object...params) {
		return backward(outputErrors, null, true, learningRate, params);
	}
	
	
	/**
	 * Extracting training flag.
	 * @param params parameters.
	 * @return training flag.
	 */
	static boolean extractFastMode(Object[] params) {
		if (params == null || params.length == 0) return false;
		for (Object param : params) {
			if (param != null && param instanceof Boolean) return (Boolean)param;
		}
		return false;
	}

	
	/**
	 * Adding fast mode.
	 * @param params array of parameters.
	 * @param fastMode fast mode.
	 * @return new array of parameters.
	 */
	static Object[] addFastMode(Object[] params, boolean fastMode) {
		return addOtherParam(params, fastMode);
	}
	
	
	/**
	 * Adding other parameter.
	 * @param params array of parameters.
	 * @param otherParam other parameter.
	 * @return new array of parameters.
	 */
	static Object[] addOtherParam(Object[] params, Object otherParam) {
		if (otherParam == null) return params;
		if (params == null || params.length == 0) return new Object[] {otherParam};
		params = Arrays.copyOf(params, params.length + 1);
		params[params.length-1] = otherParam;
		return params;
	}
	
	
}
