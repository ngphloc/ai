/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents residual network (residual connection).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ResidualNetwork extends MatrixNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ResidualNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ResidualNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ResidualNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}


	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ResidualNetwork(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	@Override
	Matrix evaluate(Matrix input, Object... params) {
		Matrix output = super.evaluate(input, params);
		output = output.add(getInput());
		getOutputLayer().setOutput(output);
		return output;
	}


	@Override
	protected Matrix calcOutputError(Matrix output, Matrix realOutput, MatrixLayerAbstract outputLayer) {
		LikelihoodGradient grad = this.likelihoodGradient;
		if (grad == null) grad = LikelihoodGradient::error;
		Matrix error = grad.gradient(output, realOutput);
		
		if (outputLayer == null) return error;
		Matrix input = outputLayer.getInput();
		Matrix derivative = input != null ? input.derivativeWise(outputLayer.getActivateRef()) : null;
		if (derivative == null) return error;
		
		NeuronValue unit = derivative.get(0, 0).unit();
		for (int row = 0; row < derivative.rows(); row++) {
			for (int column = 0; column < derivative.columns(); column++) {
				NeuronValue value = derivative.get(row, column);
				derivative.set(row, column, value.add(unit));
			}
		}
		return derivative != null ? derivative.multiplyWise(error) : error;
	}

	
}
