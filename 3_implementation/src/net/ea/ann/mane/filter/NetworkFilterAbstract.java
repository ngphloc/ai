/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class is an abstract implementation of network filter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class NetworkFilterAbstract extends KernelFilter implements NetworkFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	protected NetworkFilterAbstract() {
		super();
	}
	
	
	@Override
	public boolean doesApplyActivate() {return false;}

	
	/**
	 * Applying this filter to specific layers.
	 * @param time time.
	 * @param depth depth.
	 * @param y y coordinator.
	 * @param x x coordinator.
	 * @param layer layer.
	 * @return filtered value.
	 */
	abstract NeuronValue apply(int time, int depth, int y, int x, Matrix layer);
	
	
	@Override
	NeuronValue apply(int time, int y, int x, MatrixStack layers) {
		int depth = depth();
		NeuronValue sum = null;
		for (int d = 0; d < depth; d++) {
			NeuronValue value = apply(time, d, y, x, layers.get(d));
			sum = sum != null ? sum.add(value) : value;
		}
		return sum;
	}


	/**
	 * Calculating derivative of previous layer given current layer as bias layer.
	 * @param time time.
	 * @param depth depth.
	 * @param thisY current Y coordinator.
	 * @param thisX current X coordinator.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return derivative of previous layers given current layer as bias layer.
	 */
	abstract Matrix dValue(int time, int depth, int thisY, int thisX, Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef, boolean learning, double learningRate);
	
	
	/**
	 * Calculating derivative of previous layer given current layer as bias layer.
	 * @param time time.
	 * @param depth depth.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return derivative of previous layers given current layer as bias layer.
	 */
	private Matrix dValue(int time, int depth, Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef, boolean learning, double learningRate) {
		NeuronValue zero = prevInputLayer.get(0, 0).zero();
		int rows = prevInputLayer.rows(), columns = prevInputLayer.columns();
		Matrix dPrevValues = prevInputLayer.create(new Size(columns, rows));
		MatrixUtil.fill(dPrevValues, zero);
		int[][] dPrevValuesCount = new int[rows][columns];
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < columns; k++) dPrevValuesCount[j][k] = 0;
		}

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayer.columns(), prevHeight = prevInputLayer.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayer.columns(), thisHeight = thisErrorLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				Matrix dPrevValue = this.dValue(time, depth, thisY, thisX, prevInputLayer, prevOutputLayer, thisErrorLayer, thisActivateRef, learning, learningRate);
				if (dPrevValue == null) continue;
				
				for (int j = 0; j < dPrevValue.rows(); j++) {
					int prevRow = prevY + j;
					for (int k = 0; k < dPrevValue.columns(); k++) {
						int prevColumn = prevX + k;
						NeuronValue dv = dPrevValues.get(prevRow, prevColumn).add(dPrevValue.get(j, k));
						dPrevValues.set(prevRow, prevColumn, dv);
						dPrevValuesCount[prevRow][prevColumn] = dPrevValuesCount[prevRow][prevColumn] + 1; 
					}
				} //End dValues.
			}
		}
		
		//Calculating mean of values.
		if (CALC_ERROR_MEAN) {
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					int count = dPrevValuesCount[row][column];
					if (count <= 0) continue;
					NeuronValue mean = dPrevValues.get(row, column).divide(count);
					dPrevValues.set(row, column, mean);
				}
			}
		}
		return dPrevValues;
	}
	
	
	/**
	 * Calculating derivative of previous layers given current layer as bias layer.
	 * @param time time.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return derivative of previous layers given current layer as bias layer.
	 */
	private Matrix dValue(int time, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef, boolean learning, double learningRate) {
		int depth = depth();
		Matrix sum = null;
		for (int d = 0; d < depth; d++) {
			Matrix dValue = dValue(time, d, prevInputLayers.get(d), prevOutputLayer, thisErrorLayer, thisActivateRef, learning, learningRate);
			sum = sum != null ? sum.add(dValue) : dValue;
		}
		return sum;

	}
	

	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @param thisActivateRef activation function of current layers.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevOutputLayers, MatrixStack thisErrorLayers, Function thisActivateRef, boolean learning, double learningRate) {
		if (prevInputLayers.depth() != depth() || prevOutputLayers.depth() != time() || thisErrorLayers.depth() != time()) throw new IllegalArgumentException();
		if (prevOutputLayers.rows() != thisErrorLayers.rows() || prevOutputLayers.columns() != thisErrorLayers.columns()) throw new IllegalArgumentException();

		int time = time();
		Matrix[] dValues = new Matrix[time];
		for (int t = 0; t < time; t++) {
			dValues[t] = dValue(t, prevInputLayers, prevOutputLayers.get(t), thisErrorLayers.get(t), thisActivateRef, learning, learningRate);
		}
		return new MatrixStack(dValues);
	}

	
	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef, boolean learning, double learningRate) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		MatrixStack dValue = dValue(prevInputLayers, prevOutputLayers, thisErrorLayers, thisActivateRef, learning, learningRate);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}


	@Override
	MatrixStack dValue(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		throw new RuntimeException("Not support this method");
	}

	
	@Override
	MatrixStack dKernel(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		throw new RuntimeException("Network-based filter does not calculate gradient of kernel");
	}


}
