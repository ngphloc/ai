/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.MatrixNetworkAssoc;
import net.ea.ann.mane.MatrixNetworkImpl;

/**
 * This class is an abstract implementation of network filter based on matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class FilterNetwork extends MatrixNetworkImpl implements NetworkFilter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to move according to stride when filtering.
	 */
	protected boolean moveStride = MOVE_STRIDE;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public FilterNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public FilterNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public FilterNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public FilterNetwork(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	@Override
	public boolean isMoveStride() {return moveStride;}


	@Override
	public void setMoveStride(boolean moveStride) {this.moveStride = moveStride;}

	
	@Override
	public Filter accumKernel(Kernel dKernel, double factor) {return this;}


	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		Matrix output = evaluate0(prevLayer, new Object[] {});
		MatrixUtil.copy(output, thisInputLayer);
		MatrixUtil.copy(output, thisOutputLayer);
	}


	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef, boolean learning, double learningRate) {
		return backward(new Error[] {new Error(thisErrorLayer)}, null, learning, learningRate)[0].error();
	}


	@Override
	public void initParams(double v) {new MatrixNetworkAssoc(this).initParams(v);}


	@Override
	public void initParams(Random rnd) {new MatrixNetworkAssoc(this).initParams(rnd);}


	@Override
	public int sizeOfParams() {return new MatrixNetworkAssoc(this).sizeOfParams();}
	
	
}
