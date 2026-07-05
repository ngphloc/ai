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
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error.LayerInput;
import net.ea.ann.raster.Size;

/**
 * This class implements custom layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FlattenLayer extends CustomLayer {


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
	public FlattenLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public FlattenLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public FlattenLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public FlattenLayer(int neuronChannel) {this(neuronChannel, null, null, null);}


	@Override
	public boolean initialize(Size size, Size prevSize, LayerSpec layerSpec) {
		if (size == null || prevSize == null) return false;
		int length = 0;
		if (size.depth <= 0) {
			if (prevSize.depth > 0) throw new IllegalArgumentException();
			length = prevSize.width*prevSize.height;
		}
		else if (size.depth == 1) {
			if (prevSize.depth <= 0) throw new IllegalArgumentException();
			length = prevSize.width*prevSize.height*prevSize.depth;
		}
		else {
			throw new IllegalArgumentException();
		}
		if (length <= 0 || size.width != 1 || size.height != length) throw new IllegalArgumentException();
		this.prevInput = this.prevOutput = null;
		this.input = this.output = null;
		this.weight = null;
		this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);
		resetBackwardInfo();

		this.output = this.input = newMatrix(new Size(1, length, 1));
		return true;
	}


	@Override
	public Matrix evaluate(Object... params) {
		if (this.output != this.input || this.output.columns() != 1 || MatrixUtil.depth(this.output) != 1) throw new IllegalArgumentException();
		if (this.prevLayer == null) throw new IllegalArgumentException(); //Flatten layer cannot be input layer.
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (MatrixUtil.capacity(prevLayerOutput) != MatrixUtil.capacity(this.output)) throw new IllegalArgumentException();
		
		Matrix[] prevLayerOutputs = MatrixUtil.split(prevLayerOutput);
		int depth = prevLayerOutputs.length, rows = prevLayerOutput.rows(), columns = prevLayerOutput.columns();
		int index = 0;
		for (int d = 0; d < depth; d++) {
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue value = prevLayerOutputs[d].get(row, column);
					this.input.set(index, 0, value);
					index++;
				}
			}
		}
		
		if (this.output != this.input) throw new IllegalArgumentException();
		if (index != MatrixUtil.capacity(this.output)) throw new IllegalArgumentException();
		Error.addLayerOInput(this, params);
		return (this.output = this.input);
	}


	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (outputErrors == null || outputErrors.length == 0) return null;
		if (this.output != this.input || this.prevLayer == null) throw new IllegalArgumentException(); //Flatten layer cannot be input layer.
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (MatrixUtil.capacity(prevLayerOutput) != MatrixUtil.capacity(this.output)) throw new IllegalArgumentException();
		
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

			Matrix backwardError = prevLayerOutput.create(new Size(prevLayerOutput.columns(), prevLayerOutput.rows()));
			int rows = backwardError.rows(), columns = backwardError.columns();
			if (backwardError instanceof MatrixStack) {
				MatrixStack backwardErrors = (MatrixStack)backwardError;
				int depth = backwardErrors.depth();
				for (int d = 0; d < depth; d++) {
					int depthIndex = d*rows*columns;
					for (int row = 0; row < rows; row++) {
						int rowIndex = depthIndex + row*columns;
						for (int column = 0; column < columns; column++) {
							int index = rowIndex + column;
							backwardErrors.get(d).set(row, column, error.get(index, 0));
						}
					}
				}
			}
			else {
				for (int row = 0; row < rows; row++) {
					int rowIndex = row*columns;
					for (int column = 0; column < columns; column++) {
						int index = rowIndex + column;
						backwardError.set(row, column, error.get(index, 0));
					}
				}
			}
			
			//Setting backward.
			outputErrors[i].errorSet(backwardError);
		}

		return outputErrors;
	}


	@Override
	protected void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		//Do nothing.
	}


}
