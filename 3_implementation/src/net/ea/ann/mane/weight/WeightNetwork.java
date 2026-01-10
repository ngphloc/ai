/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.MatrixNetworkAssoc;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.Weight;

/**
 * This class is an abstract implementation of network weight based on matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class WeightNetwork extends MatrixNetworkImpl implements NetworkWeight {


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
	public WeightNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public WeightNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public WeightNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public WeightNetwork(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {return this;}


	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		return super.evaluate0(input, new Object[] {});
	}


	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate) {
		return backward(new Error[] {new Error(thisError)}, null, learning, learningRate)[0].error();
	}


	@Override
	public void initParams(double v) {new MatrixNetworkAssoc(this).initParams(v);}


	@Override
	public void initParams(Random rnd) {new MatrixNetworkAssoc(this).initParams(rnd);}


	@Override
	public int sizeOfParams() {return new MatrixNetworkAssoc(this).sizeOfParams();}


}
