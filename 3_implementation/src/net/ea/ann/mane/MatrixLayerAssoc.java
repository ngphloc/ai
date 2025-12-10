/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.conv.ConvLayer2DImpl;
import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.conv.ConvNeuron;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This utility class provides utility methods for matrix network layer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixLayerAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix network layer.
	 */
	protected MatrixLayerImpl layer = null;
	

	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	public MatrixLayerAssoc(MatrixLayerImpl layer) {
		this.layer = layer;
	}

	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		if (layer.weight != null) layer.weight.fill(v);
		if (layer.bias != null) Matrix.fill(layer.bias, v);
		
		if (layer.filter != null) layer.filter.initialize(v);
		if (layer.filterBias != null) layer.filterBias = layer.filterBias.valueOf(v);
	}


	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	public void initParams(Random rnd) {
		if (layer.weight != null) layer.weight.fill(rnd);
		if (layer.bias != null) Matrix.fill(layer.bias, rnd);
		
		if (layer.filter != null) layer.filter.initialize(rnd);
		if (layer.filterBias != null) layer.filterBias = layer.filterBias.valueOf(NeuronValue.r(rnd));
	}

	
	/**
	 * Converting convolutional layer to matrix.
	 * @param layer convolutional layer.
	 * @return matrix.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private Matrix convLayerToMatrix(ConvLayerSingle2D convLayer) {
		if (convLayer == null) return null;
		int rows = convLayer.getHeight();
		int columns = convLayer.getWidth();
		int depth = convLayer.getDepth();
		Matrix matrix = layer.newMatrix(new Size(columns, rows, depth <= 0 ? 1 : depth));
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				matrix.set(i, j, convLayer.get(j, i).getValue());
			}
		}
		return layer.isVectorized() ? matrix.vec() : matrix;
	}

	
	/**
	 * Converting array to matrix.
	 * @param array array.
	 * @param size size.
	 * @return matrix.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private Matrix arrayToMatrix(NeuronValue[] array, Size size) {
		Matrix matrix = layer.newMatrix(size);
		for (int i = 0; i < size.height; i++) {
			int rowLength = i*size.width;
			for (int j = 0; j < size.width; j++) {
				int index = rowLength + j;
				matrix.set(i, j, array[index]);
			}
		}
		return layer.isVectorized() ? matrix.vec() : matrix;
	}
	
	
	/**
	 * Converting matrix to convolutional layer.
	 * @param matrix matrix.
	 * @return convolutional layer.
	 */
	ConvLayerSingle2D matrixToConvLayer(Matrix matrix) {
		if (matrix == null) return null;
		matrix = layer.isVectorized() ? matrix.vecInverse(layer.vecRows) : matrix;
		int rows = matrix.rows();
		int columns = matrix.columns();
		ConvLayerSingle2D convLayer = newConvLayer(columns, rows);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				ConvNeuron neuron = convLayer.get(j, i);
				neuron.setValue(matrix.get(i, j));
			}
		}
		return convLayer;
	}

	
	/**
	 * Creating convolutional layer.
	 * @param width specified width.
	 * @param height specified height.
	 * @return convolutional layer.
	 */
	ConvLayerSingle2D newConvLayer(int width, int height) {
		return ConvLayer2DImpl.create(layer.neuronChannel, layer.convActivateRef, width, height, null, layer.getIdRef());
	}

	
	/**
	 * Getting previous input as convolutional layer.
	 * @return previous input as convolutional layer.
	 */
	ConvLayerSingle2D getPrevInputConvLayer() {
		return layer.prevInput != null ? matrixToConvLayer(layer.prevInput) : null;
	}
	
	
}
