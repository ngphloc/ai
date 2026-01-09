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
import net.ea.ann.core.value.MatrixStack;

/**
 * This class implements partially parametric weight based on transformer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class NetworkWeightAbstract implements NetworkWeight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	protected NetworkWeightAbstract() {

	}

	
	/**
	 * Getting width.
	 * @return width.
	 */
	abstract int width();
	
	
	/**
	 * Getting height.
	 * @return height.
	 */
	abstract int height();

	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	abstract int depth();

	
	/**
	 * Getting time.
	 * @return time.
	 */
	abstract int time();

	
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param inputs inputs.
	 * @param bias bias.
	 * @return evaluated layer.
	 */
	abstract Matrix evaluate(int time, MatrixStack inputs, Matrix bias);

		
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param inputs inputs.
	 * @param biases biases.
	 * @return evaluated layer.
	 */
	private MatrixStack evaluate(MatrixStack inputs, MatrixStack biases) {
		if (inputs.depth() != depth() || biases.depth() != time()) throw new IllegalArgumentException();
		int time = time();
		Matrix[] values = new Matrix[time];
		for (int t = 0; t < time; t++) {
			values[t] = evaluate(t, inputs, biases!=null?biases.get(t):null);
		}
		return new MatrixStack(values);
	}
	

	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		MatrixStack inputs = input instanceof MatrixStack ? (MatrixStack)input : new MatrixStack(input);
		MatrixStack biases = bias != null ? (bias instanceof MatrixStack ? (MatrixStack)bias : new MatrixStack(bias)) : null;
		MatrixStack values = evaluate(inputs, biases);
		return values.depth() == 1 ? values.get() : values;
	}

	
	/**
	 * Calculate gradient of previous layers.
	 * @param time time.
	 * @param prevInputs previous inputs.
	 * @param prevOutput previous output.
	 * @param thisError current error.
	 * @param prevActivateRef previous activation function.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return gradient of previous layers.
	 */
	abstract Matrix dValue(int time, MatrixStack prevInputs, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate);

		
	/**
	 * Calculate gradient of previous layers.
	 * @param prevInputs previous inputs.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @param prevActivateRef previous activation function.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return gradient of previous layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputs, MatrixStack prevOutputs, MatrixStack thisErrors, Function prevActivateRef, boolean learning, double learningRate) {
		if (prevInputs.depth() != depth() || prevOutputs.depth() != time() || thisErrors.depth() != time()) throw new IllegalArgumentException();
		int time = time();
		Matrix[] dValues = new Matrix[time];
		for (int t = 0; t < time; t++) {
			dValues[t] = dValue(t, prevInputs, prevOutputs.get(t), thisErrors.get(t), prevActivateRef, learning, learningRate);
		}
		return new MatrixStack(dValues);
	}

	
	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate) {
		MatrixStack prevInputs = prevInput instanceof MatrixStack ? (MatrixStack)prevInput : new MatrixStack(prevInput);
		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dValue = dValue(prevInputs, prevOutputs, thisErrors, prevActivateRef, learning, learningRate);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}
	
	
}
