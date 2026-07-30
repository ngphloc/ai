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
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.raster.Size;

/**
 * This class implements custom layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class CustomLayer extends MatrixLayerImpl {


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
	public CustomLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public CustomLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public CustomLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public CustomLayer(int neuronChannel) {this(neuronChannel, null, null, null);}


	/**
	 * Copying parameters from source layer.
	 * @param source source layer.
	 */
	protected void copyParameters(MatrixLayerImpl source) {
		assert (source != null);
		if (this.weight != null && source.weight != null) this.weight.copyParameters(source.weight);
		if (this.bias != null && source.bias != null) MatrixUtil.copy(source.bias, this.bias);
		if (this.filter != null && source.filter != null) this.filter.copyParameters(source.filter);
		if (this.filterBias != null && source.filterBias != null) this.filterBias = source.filterBias/*.duplicate()*/;
	}
	
	
	@Override
	public boolean initialize(Size size, Size prevSize, LayerSpec layerSpec) {
		return super.initialize(size, prevSize, layerSpec);
	}


	@Override
	public Matrix evaluate(Object... params) {
		return super.evaluate(params);
	}


	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		return super.backward(outputErrors, focus, learning, learningRate);
	}


}

