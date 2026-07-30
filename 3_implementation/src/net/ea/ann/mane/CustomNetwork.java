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

/**
 * This class represents custom matrix neural network.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class CustomNetwork extends MatrixNetworkImpl {


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
	public CustomNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public CustomNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public CustomNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public CustomNetwork(int neuronChannel) {this(neuronChannel, null, null, null);}


	/**
	 * Copying parameters from source network.
	 * @param source source network.
	 */
	protected void copyParameters(MatrixNetworkImpl source) {
		if (this.layers == null) return;
		assert (source.layers != null && source.layers.length == this.layers.length);
		for (int i = 0; i < this.layers.length; i++) {
			if (!(this.layers[i] instanceof CustomLayer)) continue;
			
			assert (source.layers[i] instanceof CustomLayer);
			CustomLayer thisLayer = (CustomLayer)this.layers[i];
			CustomLayer sourceLayer = (CustomLayer)source.layers[i];
			thisLayer.copyParameters(sourceLayer);
		}
	}
	
	
	@Override
	public Matrix evaluate0(Matrix input, Object... params) {
		return super.evaluate0(input, params);
	}


}
