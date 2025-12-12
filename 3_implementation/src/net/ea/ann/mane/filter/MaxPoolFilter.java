/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueV;
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

	
	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	private Point apply(int x, int y, Matrix layer) {
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
		NeuronValue zero = thisIndexInputLayer != null ? thisIndexInputLayer.get(0, 0).zero() : (thisOutputLayer != null ? thisOutputLayer.get(0, 0).zero() : prevLayer.get(0, 0).zero());
		Matrix.fill(thisIndexInputLayer, zero);
		Matrix.fill(thisOutputLayer, zero);

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
				Point filteredIndex = this.apply(prevX, prevY, prevLayer);
				if (filteredIndex == null) continue;
				NeuronValue filteredValue = prevLayer.get(filteredIndex.y, filteredIndex.x);
				if (thisIndexInputLayer != null) {
					NeuronValueV index = new NeuronValueV(thisY, thisX);
					thisIndexInputLayer.set(filteredIndex.y, filteredIndex.x, index);
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
		if (thisIndexInputLayers.rows() != prevLayers.rows() || thisIndexInputLayers.columns() != prevLayers.columns()) throw new IllegalArgumentException();
		
		for (int d = 0; d < depth(); d++) {
			forward0(prevLayers.get(d), thisIndexInputLayers.get(d), thisOutputLayers.get(d));
		}
	}
	

	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer) {
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
	private MatrixStack dValue(int thisX, int thisY, Matrix prevInputLayer, Matrix prevIndexOutputLayer, Matrix thisErrorLayer) {
		int kernelWidth = width(), kernelHeight = height(), kernelDepth = depth();
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

		Matrix[] dValues = new Matrix[kernelDepth];
		NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
		NeuronValue zero = thisError.zero();
		for (int i = 0; i < kernelDepth; i++) {
			dValues[i] = prevInputLayer.create(new Size(kernelWidth, kernelHeight));
			for (int j = 0; j < kernelHeight; j++) {
				int row = prevY + j;
				for (int k = 0; k < kernelWidth; k++) {
					int column = prevX + k;
					NeuronValueV index = (NeuronValueV)prevIndexOutputLayer.get(row, column);
					if (index.get(0) == thisY && index.get(1) == thisX)
						dValues[i].set(j, k, thisError);
					else
						dValues[i].set(j, k, zero);
				}
			}
		}
		return new MatrixStack(dValues);
	}

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevIndexOutputLayers previous index output layers.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevIndexOutputLayers, Matrix thisErrorLayer) {
		NeuronValue zero = prevInputLayers.get().get(0, 0).zero();
		Matrix[] dPrevValues = new Matrix[this.depth()];
		int[][][] dPrevValuesCount = new int[this.depth()][][];
		for (int i = 0; i < dPrevValues.length; i++) {
			int rows = prevInputLayers.rows(), columns = prevInputLayers.columns();
			dPrevValues[i] = prevInputLayers.get().create(new Size(columns, rows));
			Matrix.fill(dPrevValues[i], zero);
			dPrevValuesCount[i] = new int[rows][columns];
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < columns; k++) dPrevValuesCount[i][j][k] = 0;
			}
		}

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
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
				MatrixStack dValues = this.dValue(thisX, thisY, prevInputLayers, prevIndexOutputLayers, thisErrorLayer);
				if (dValues == null) continue;
				
				for (int i = 0; i < dValues.depth(); i++) {
					for (int j = 0; j < dValues.get(i).rows(); j++) {
						int row = prevY + j;
						for (int k = 0; k < dValues.get(i).columns(); k++) {
							int column = prevX + k;
							NeuronValue dv = dPrevValues[i].get(row, column).add(dValues.get(i).get(j, k));
							dPrevValues[i].set(row, column, dv);
							dPrevValuesCount[i][row][column] = dPrevValuesCount[i][row][column] + 1; 
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

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param prevInputLayers previous input layers.
	 * @param prevIndexOutputLayers previous input output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevIndexOutputLayers, MatrixStack thisErrorLayers) {
		if (prevInputLayers.depth() != depth() || prevIndexOutputLayers.depth() != depth()) throw new IllegalArgumentException();
		if (prevIndexOutputLayers.rows() != prevInputLayers.rows() || prevIndexOutputLayers.columns() != prevInputLayers.columns()) throw new IllegalArgumentException();
		
		MatrixStack dValueSum = null;
		for (int d = 0; d < thisErrorLayers.depth(); d++) {
			MatrixStack dValue = dValue(prevInputLayers, prevIndexOutputLayers, thisErrorLayers.get(d));
			dValueSum = dValueSum != null ? (MatrixStack)dValueSum.add(dValue) : dValue;
		}
		return dValueSum;
	}
	

	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevIndexOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		MatrixStack stack = dValue(prevInputLayers, prevIndexOutputLayers, thisErrorLayers);
		return stack.depth() == 1 ? stack.get() : stack;
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
