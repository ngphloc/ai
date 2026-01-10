/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.raster.Point;
import net.ea.ann.raster.Size;

/**
 * This class represents max pooling filter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class PoolFilterMax extends PoolFilter {


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
	protected PoolFilterMax(Size size) {
		super(size);
		this.depth = size.depth < 1 ? 1 : size.depth; 
	}


	@Override
	int depth() {return depth;}


	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param y y coordinator.
	 * @param x x coordinator.
	 * @param layer specific layer.
	 * @return the index value resulted from this application.
	 */
	private Point apply(int y, int x, Matrix layer) {
		int width = layer.columns(), height = layer.rows();
		if (y + height() > height) {
			if (isPadZero())
				return y >= height ? null : null;
			else
				y = height - height();
		}
		if (x + width() > width) {
			if (isPadZero())
				return x >= width ? null : null;
			else
				x = width - width();
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


	@Override
	void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer) {
		NeuronValueV zeroV = new NeuronValueV(2, 0);
		MatrixUtil.fill(thisInputLayer, zeroV);
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
				if (thisInputLayer != null) {
					NeuronValueV prevIndex = new NeuronValueV((double)filteredIndex.y, (double)filteredIndex.x);
					thisInputLayer.set(thisY, thisX, prevIndex);
				}
				if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValue);
			}
		}
	}


	@Override
	Matrix dValue(int thisX, int thisY, Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer) {
		int kernelWidth = width(), kernelHeight = height();
		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayer.columns(), prevHeight = prevInputLayer.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
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

		NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
		NeuronValueV thisErrorIndex = (NeuronValueV)prevOutputLayer.get(thisY, thisX);
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

	
	/**
	 * Creating max pooling filter with specific kernel size.
	 * @param size specific kernel size.
	 * @return max pooling filter created from specific kernel size.
	 */
	public static PoolFilterMax create(Size size) {
		return new PoolFilterMax(size);
	}


}
