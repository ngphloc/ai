/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.layers;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.ParameterLayer;

/**
 * This class implements normalization layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NormLayer extends ParameterLayer /*MatrixLayerImpl*/ {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Norm information.
	 */
	protected Object normInfo = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public NormLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public NormLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public NormLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public NormLayer(int neuronChannel) {this(neuronChannel, null, null, null);}


	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (outputErrors.length > 1) throw new IllegalArgumentException();
		return super.backward(outputErrors, focus, learning, learningRate);
	}


	/**
	 * Getting tag.
	 * @return tag.
	 */
	public Object getNormInfo() {return this.normInfo;}
	
	
	/**
	 * Setting norm information.
	 * @param normInfo norm information.
	 */
	public void setNormInfo(Object normInfo) {this.normInfo = normInfo;}
	
	
}


