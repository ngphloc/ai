package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.Error.LayerInput;
import net.ea.ann.raster.Size;

/**
 * This class implements null layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NullLayer extends CustomLayer {


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
	public NullLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public NullLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public NullLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public NullLayer(int neuronChannel) {this(neuronChannel, null, null, null);}


	@Override
	public boolean initialize(Size size, Size prevSize, LayerSpec layerSpec) {
		if (size == null || prevSize == null) return false;
		if (size.width != prevSize.width || size.height != prevSize.height || size.depth != prevSize.depth) throw new IllegalArgumentException();
		this.prevInput = this.prevOutput = null;
		this.input = this.output = null;
		this.weight = null;
		this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);
		resetBackwardInfo();

		this.output = this.input = newMatrix(size);
		return true;
	}


	@Override
	public Matrix evaluate(Object... params) {
		if (this.output != this.input || this.prevLayer == null) throw new IllegalArgumentException(); //Null layer cannot be input layer.
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (this.output.rows() != prevLayerOutput.rows() || this.output.columns() != prevLayerOutput.columns() || MatrixUtil.depth(this.output) != MatrixUtil.depth(prevLayerOutput)) throw new IllegalArgumentException();
		
		Error.addLayerOInput(this, params);
		return (this.output = this.input = prevLayerOutput);
	}


	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (outputErrors == null || outputErrors.length == 0) return null;
		if (this.output != this.input || this.prevLayer == null) throw new IllegalArgumentException(); //Null layer cannot be input layer.
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (this.output.rows() != prevLayerOutput.rows() || this.output.columns() != prevLayerOutput.columns() || MatrixUtil.depth(this.output) != MatrixUtil.depth(prevLayerOutput)) throw new IllegalArgumentException();

		for (int i = 0; i < outputErrors.length; i++) {
			Matrix error = outputErrors[i].error(); 

			//Adding residual backward errors.
			if (getEndLayer() != null) {
				LayerInput layerInput = outputErrors[i].layerOInput(getEndLayer());
				Matrix endError = layerInput != null && layerInput.backwardError != null ? layerInput.backwardError.error() : null;
				if (endError != null) error = error.add(endError);
			}
			//Adjusting errors.
			error = adjustError(error, outputErrors[i]);
			assert (error.rows() == this.output.rows() && error.columns() == this.output.columns() && MatrixUtil.depth(error) == MatrixUtil.depth(this.output));

			//Setting backward.
			outputErrors[i].errorSet(error);
		}

		return outputErrors;
	}


	@Override
	protected void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		//Do nothing.
	}


}

