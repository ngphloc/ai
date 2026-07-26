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
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.MatrixLayerImpl;
import net.ea.ann.raster.Size;

/**
 * This class implements extensive flattening layer with matrix.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FlattenLayer2 extends MatrixLayerImpl {


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
	public FlattenLayer2(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public FlattenLayer2(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public FlattenLayer2(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public FlattenLayer2(int neuronChannel) {this(neuronChannel, null, null, null);}


	@Override
	public boolean initialize(Size size, Size prevSize, LayerSpec layerSpec) {
		if (size == null || prevSize == null || size.depth != 1 || prevSize.depth != 1) return false;
		if (size.depth != 1 || size.width != prevSize.width || size.height != prevSize.height*prevSize.depth) throw new IllegalArgumentException();

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
		this.bias = newMatrix(size);
		return true;
	}


	@Override
	public Matrix evaluate(Object... params) {
		if (this.output != this.input || MatrixUtil.depth(this.output) != 1) throw new IllegalArgumentException();
		if (this.prevLayer == null) throw new IllegalArgumentException(); //Flatten layer cannot be input layer.
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (MatrixUtil.capacity(prevLayerOutput) != MatrixUtil.capacity(this.output)) throw new IllegalArgumentException();
		
		Matrix[] prevLayerOutputs = MatrixUtil.split(prevLayerOutput);
		int depth = prevLayerOutputs.length, rows = prevLayerOutput.rows(), columns = prevLayerOutput.columns();
		int index = 0;
		for (int d = 0; d < depth; d++) {
			int drows = d*rows;
			for (int row = 0; row < rows; row++) {
				int thisRow = drows + row; //Please pay attention to this code line.
				for (int column = 0; column < columns; column++) {
					NeuronValue value = prevLayerOutputs[d].get(row, column);
					this.input.set(thisRow, column, value);
					index++;
				}
			}
		}
		
		if (this.output != this.input) throw new IllegalArgumentException();
		if (index != MatrixUtil.capacity(this.output)) throw new IllegalArgumentException();
		Error.addLayerOInput2(this, params);
		return (this.output = this.input);
	}


	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		outputErrors = super.backward(outputErrors, focus, learning, learningRate);
		
		if (outputErrors == null || outputErrors.length == 0) return null;
		if (this.output != this.input || this.prevLayer == null || this.weight != null || this.filter != null) throw new IllegalArgumentException(); //Flatten layer cannot be input layer.
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (MatrixUtil.capacity(prevLayerOutput) != MatrixUtil.capacity(this.output)) throw new IllegalArgumentException();
		
		int count = 0;
		for (int i = 0; i < outputErrors.length; i++) {
			Matrix error = outputErrors[i].error();
			
			Matrix backwardError = prevLayerOutput.create(new Size(prevLayerOutput.columns(), prevLayerOutput.rows()));
			int rows = backwardError.rows(), columns = backwardError.columns();
			if (backwardError instanceof MatrixStack) {
				MatrixStack backwardErrors = (MatrixStack)backwardError;
				int depth = backwardErrors.depth();
				for (int d = 0; d < depth; d++) {
					int drows = d*rows;
					for (int row = 0; row < rows; row++) {
						int thisRow = drows + row; //Please pay attention to this code line.
						for (int column = 0; column < columns; column++) {
							backwardErrors.get(d).set(row, column, error.get(thisRow, column));
							count++;
						}
					}
				}
			}
			else {
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						backwardError.set(row, column, error.get(row, column));
						count++;
					}
				}
			}
			
			if (count != MatrixUtil.capacity(this.output) || count != MatrixUtil.capacity(backwardError)) throw new IllegalArgumentException();
			
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
