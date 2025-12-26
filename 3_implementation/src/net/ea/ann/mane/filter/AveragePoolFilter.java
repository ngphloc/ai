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
import net.ea.ann.raster.Size;

/**
 * This class represents average pooling filter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class AveragePoolFilter extends PoolFilter {


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
	protected AveragePoolFilter(Size size) {
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
	 * @param y y coordinator.
	 * @param x x coordinator.
	 * @param layer specific layer.
	 * @return the index value resulted from this application.
	 */
	private NeuronValue apply(int y, int x, Matrix layer) {
		int width = layer.columns();
		int height = layer.rows();
		NeuronValue zero = layer.get(0, 0).zero();
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

		NeuronValue result = zero;
		for (int i = 0; i < height(); i++) {
			for (int j = 0; j < width(); j++) {
				NeuronValue value = layer.get(y+i, x+j);
				result = result.add(value);
			}
		}
		return result.divide(width()*height());
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
				NeuronValue filteredValue = this.apply(prevY, prevX, prevLayer);
				if (filteredValue == null) continue;
				if (thisInputLayer != null) thisInputLayer.set(thisY, thisX, filteredValue);
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

		NeuronValue thisError = thisErrorLayer.get(thisY, thisX).divide(kernelWidth*kernelHeight);
		Matrix dPrevValue = prevInputLayer.create(new Size(kernelWidth, kernelHeight));
		for (int j = 0; j < kernelHeight; j++) {
			for (int k = 0; k < kernelWidth; k++) {
				dPrevValue.set(j, k, thisError);
			}
		}
		return dPrevValue;
	}

	
	/**
	 * Creating average pooling filter with specific kernel size.
	 * @param size specific kernel size.
	 * @return average pooling filter created from specific kernel size.
	 */
	public static AveragePoolFilter create(Size size) {
		return new AveragePoolFilter(size);
	}


}
