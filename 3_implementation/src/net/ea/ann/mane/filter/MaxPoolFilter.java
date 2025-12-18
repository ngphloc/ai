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
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Point;
import net.ea.ann.raster.Size;

/**
 * This class represents max pooling filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MaxPoolFilter extends PoolFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Depth.
	 */
	protected int depth = 1;
	
	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	protected MaxPoolFilter(Size size) {
		super(size);
		this.depth = size.depth < 1 ? 1 : size.depth; 
	}

	
	/**
	 * Getting filter depth.
	 * @return filter depth.
	 */
	public int depth() {return depth;}


	@Override
	public Filter accumKernel(Kernel dKernel, double factor) {
		return this;
	}


	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param y y coordinator.
	 * @param x x coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	private Point apply(int y, int x, Matrix layer) {
		int width = layer.columns();
		int height = layer.rows();
		if (x + width() > width) {
			if (isPadZero())
				return x >= width ? null : null;
			else
				x = width - width();
		}
		if (y + height() > height) {
			if (isPadZero())
				return y >= height ? null : null;
			else
				y = height - height();
		}

		int maxRow = 0, maxColumn = 0;
		for (int i = 0; i < height(); i++) {
			for (int j = 0; j < width(); j++) {
				if (i == 0 && j == 0) continue;
				double value = layer.get(y+i, x+j).mean();
				double maxValue = layer.get(y+maxRow, x+maxColumn).mean();
				if (value > maxValue) {
					maxRow = i;
					maxColumn = j;
				}
			}
		}
		return new Point(x+maxColumn, y+maxRow);
	}

	
	/**
	 * Forwarding evaluation from previous layer to current layer.
	 * @param prevLayer previous layer.
	 * @param thisIndexInputLayer current index input layer.
	 * @param thisOutputLayer current output layer.
	 */
	private void forward0(Matrix prevLayer, Matrix thisIndexInputLayer, Matrix thisOutputLayer) {
		NeuronValueV zeroV = new NeuronValueV(2, 0);
		MatrixUtil.fill(thisIndexInputLayer, zeroV);
		NeuronValue zero = thisOutputLayer != null ? thisOutputLayer.get(0, 0).zero() : prevLayer.get(0, 0).zero();
		MatrixUtil.fill(thisOutputLayer, zero);

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevLayer.columns(), prevHeight = prevLayer.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisOutputLayer.columns(), thisHeight = thisOutputLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Filtering
				Point filteredIndex = this.apply(prevY, prevX, prevLayer);
				if (filteredIndex == null) continue;
				NeuronValue filteredValue = prevLayer.get(filteredIndex.y, filteredIndex.x);
				if (thisIndexInputLayer != null) {
					NeuronValueV prevIndex = new NeuronValueV((double)filteredIndex.y, (double)filteredIndex.x);
					thisIndexInputLayer.set(thisY, thisX, prevIndex);
				}
				if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValue);
			}
		}
	}

	/**
	 * Forwarding evaluation from previous layers to this layers.
	 * @param time time.
	 * @param prevLayers previous layers.
	 * @param thisIndexInputLayers current index input layers.
	 * @param thisOutputLayers current output layers.
	 */
	private void forward(MatrixStack prevLayers, MatrixStack thisIndexInputLayers, MatrixStack thisOutputLayers) {
		if (prevLayers.depth() != depth() || thisIndexInputLayers.depth() != depth() || thisOutputLayers.depth() != depth()) throw new IllegalArgumentException();
		if (thisIndexInputLayers.rows() != thisOutputLayers.rows() || thisIndexInputLayers.columns() != thisOutputLayers.columns()) throw new IllegalArgumentException();
		
		for (int d = 0; d < depth(); d++) {
			forward0(prevLayers.get(d), thisIndexInputLayers.get(d), thisOutputLayers.get(d));
		}
	}
	

	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		MatrixStack prevLayers = prevLayer instanceof MatrixStack ? (MatrixStack)prevLayer : new MatrixStack(prevLayer);
		MatrixStack thisIndexInputLayers = thisInputLayer instanceof MatrixStack ? (MatrixStack)thisInputLayer : new MatrixStack(thisInputLayer);
		MatrixStack thisOutputLayers = thisOutputLayer instanceof MatrixStack ? (MatrixStack)thisOutputLayer : new MatrixStack(thisOutputLayer);
		forward(prevLayers, thisIndexInputLayers, thisOutputLayers);
	}

	
	/**
	 * Calculating derivative of previous layer given current layer as bias layer at specified coordinator.
	 * @param time time.
	 * @param thisX current X coordinator.
	 * @param thisY current Y coordinator.
	 * @param prevInputLayer previous input layers.
	 * @param prevIndexOutputLayer previous input output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layer given current layer as bias layers.
	 */
	private Matrix dValue(int thisX, int thisY, Matrix prevInputLayer, Matrix prevIndexOutputLayer, Matrix thisErrorLayer) {
		int kernelWidth = width(), kernelHeight = height();
		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayer.columns(), prevHeight = prevInputLayer.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
		int prevX = xBlock*strideWidth;
		if (prevX + kernelWidth > prevWidth) {
			if (isPadZero())
				return prevX >= prevWidth ? null : null;
			else {
				prevX = prevWidth - kernelWidth;
				thisX = prevX/strideWidth;
			}
		}
		int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
		int prevY = yBlock*strideHeight;
		if (prevY + kernelHeight > prevHeight) {
			if (isPadZero())
				return prevY >= prevHeight ? null : null;
			else {
				prevY = prevHeight - kernelHeight;
				thisY = prevY/strideHeight;
			}
		}

		NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
		NeuronValueV thisErrorIndex = (NeuronValueV)prevIndexOutputLayer.get(thisY, thisX);
		NeuronValue zero = thisError.zero();
		Matrix dPrevValue = prevInputLayer.create(new Size(kernelWidth, kernelHeight));
		for (int j = 0; j < kernelHeight; j++) {
			int prevRow = prevY + j;
			for (int k = 0; k < kernelWidth; k++) {
				int prevColumn = prevX + k;
				if (thisErrorIndex.get(0) == prevRow && thisErrorIndex.get(1) == prevColumn)
					dPrevValue.set(j, k, thisError);
				else
					dPrevValue.set(j, k, zero);
			}
		}
		return dPrevValue;
	}

	
//	/**
//	 * Calculating derivative of previous layers given current layers as bias layers.
//	 * @param time time.
//	 * @param nextX next X coordinator.
//	 * @param nextY next Y coordinator.
//	 * @param prevInputLayers previous input layers.
//	 * @param prevIndexOutputLayers previous index output layers.
//	 * @param thisErrorLayers current layers as bias layers.
//	 * @param thisActivateRef activation function of current layer.
//	 * @return derivative of previous layers given current layers as bias layers.
//	 */
//	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevIndexOutputLayers, MatrixStack thisErrorLayers) {
//		if (prevInputLayers.depth() != depth() || prevIndexOutputLayers.depth() != depth() || thisErrorLayers.depth() != depth()) throw new IllegalArgumentException();
//		if (prevIndexOutputLayers.rows() != thisErrorLayers.rows() || prevIndexOutputLayers.columns() != thisErrorLayers.columns()) throw new IllegalArgumentException();
//
//		NeuronValue zero = prevInputLayers.get().get(0, 0).zero();
//		Matrix[] dPrevValues = new Matrix[this.depth()];
//		int[][][] dPrevValuesCount = new int[this.depth()][][];
//		for (int i = 0; i < dPrevValues.length; i++) {
//			int rows = prevInputLayers.rows(), columns = prevInputLayers.columns();
//			dPrevValues[i] = prevInputLayers.get().create(new Size(columns, rows));
//			MatrixUtil.fill(dPrevValues[i], zero);
//			dPrevValuesCount[i] = new int[rows][columns];
//			for (int j = 0; j < rows; j++) {
//				for (int k = 0; k < columns; k++) dPrevValuesCount[i][j][k] = 0;
//			}
//		}
//
//		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
//		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
//		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
//		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
//		int thisWidth = thisErrorLayers.columns(), thisHeight = thisErrorLayers.rows();
//		for (int thisY = 0; thisY < thisHeight; thisY++) {
//			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
//			int prevY = yBlock*strideHeight;
//			if (prevY >= prevHeight) continue;
//			
//			for (int thisX = 0; thisX < thisWidth; thisX++) {
//				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
//				int prevX = xBlock*strideWidth;
//				if (prevX >= prevWidth) continue;
//				
//				//Calculating gradient.
//				for (int i = 0; i < this.depth(); i++) {
//					Matrix dPrevValue = this.dValue(thisX, thisY, prevInputLayers.get(i), prevIndexOutputLayers.get(i), thisErrorLayers.get(i));
//					if (dPrevValue == null) continue;
//					for (int j = 0; j < dPrevValue.rows(); j++) {
//						int prevRow = prevY + j;
//						for (int k = 0; k < dPrevValue.columns(); k++) {
//							int prevColumn = prevX + k;
//							NeuronValue dv = dPrevValues[i].get(prevRow, prevColumn).add(dPrevValue.get(j, k));
//							dPrevValues[i].set(prevRow, prevColumn, dv);
//							dPrevValuesCount[i][prevRow][prevColumn] = dPrevValuesCount[i][prevRow][prevColumn] + 1; 
//						}
//					}
//				} //End dValues.
//			}
//		}
//		
//		//Calculating mean of values.
//		if (CALC_ERROR_MEAN) {
//			for (int i = 0; i < dPrevValues.length; i++) {
//				int rows = dPrevValues[i].rows(), columns = dPrevValues[i].columns();
//				for (int row = 0; row < rows; row++) {
//					for (int column = 0; column < columns; column++) {
//						int count = dPrevValuesCount[i][row][column];
//						if (count <= 0) continue;
//						NeuronValue mean = dPrevValues[i].get(row, column).divide(count);
//						dPrevValues[i].set(row, column, mean);
//					}
//				}
//			}
//		}
//		return new MatrixStack(dPrevValues);
//	}

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevIndexOutputLayers previous index output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevIndexOutputLayers, MatrixStack thisErrorLayers) {
		if (prevInputLayers.depth() != depth() || prevIndexOutputLayers.depth() != depth()) throw new IllegalArgumentException();
		if (prevIndexOutputLayers.rows() != thisErrorLayers.rows() || prevIndexOutputLayers.columns() != thisErrorLayers.columns()) throw new IllegalArgumentException();

		NeuronValue zero = prevInputLayers.get().get(0, 0).zero();
		Matrix[] dPrevValues = new Matrix[this.depth()];
		int[][][] dPrevValuesCount = new int[this.depth()][][];
		for (int i = 0; i < dPrevValues.length; i++) {
			int rows = prevInputLayers.rows(), columns = prevInputLayers.columns();
			dPrevValues[i] = prevInputLayers.get().create(new Size(columns, rows));
			MatrixUtil.fill(dPrevValues[i], zero);
			dPrevValuesCount[i] = new int[rows][columns];
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < columns; k++) dPrevValuesCount[i][j][k] = 0;
			}
		}

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayers.columns(), thisHeight = thisErrorLayers.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				for (int i = 0; i < this.depth(); i++) {
					Matrix dPrevValue = null;
					for (int count = 0; count < thisErrorLayers.depth(); count++) {
						Matrix dValue = this.dValue(thisX, thisY, prevInputLayers.get(i), prevIndexOutputLayers.get(i), thisErrorLayers.get(count));
						dPrevValue = dPrevValue != null ? dPrevValue.add(dValue) : dValue; 
					}
					if (dPrevValue == null) continue;
					for (int j = 0; j < dPrevValue.rows(); j++) {
						int prevRow = prevY + j;
						for (int k = 0; k < dPrevValue.columns(); k++) {
							int prevColumn = prevX + k;
							NeuronValue dv = dPrevValues[i].get(prevRow, prevColumn).add(dPrevValue.get(j, k));
							dPrevValues[i].set(prevRow, prevColumn, dv);
							dPrevValuesCount[i][prevRow][prevColumn] = dPrevValuesCount[i][prevRow][prevColumn] + 1; 
						}
					}
				} //End dValues.
			}
		}
		
		//Calculating mean of values.
		if (CALC_ERROR_MEAN) {
			for (int i = 0; i < dPrevValues.length; i++) {
				int rows = dPrevValues[i].rows(), columns = dPrevValues[i].columns();
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						int count = dPrevValuesCount[i][row][column];
						if (count <= 0) continue;
						NeuronValue mean = dPrevValues[i].get(row, column).divide(count);
						dPrevValues[i].set(row, column, mean);
					}
				}
			}
		}
		return new MatrixStack(dPrevValues);
	}

	
	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevIndexOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		MatrixStack stack = dValue(prevInputLayers, prevIndexOutputLayers, thisErrorLayers);
		return stack.depth() == 1 ? stack.get() : stack;
	}

	
	@Override
	public Kernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		return null;
	}


	/**
	 * Creating max pooling filter with specific kernel size.
	 * @param size specific kernel size.
	 * @return max pooling filter created from specific kernel size.
	 */
	public static MaxPoolFilter create(Size size) {
		return new MaxPoolFilter(size);
	}


}
