/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
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
	 * Constructor with kernel width and height.
	 * @param width kernel width.
	 * @param height kernel height.
	 */
	protected MaxPoolFilter(int width, int height) {
		super(width, height);
	}

	
	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param layer specific layer.
	 * @return the value resulted from this application.
	 */
	NeuronValue apply(int x, int y, Matrix layer) {
		NeuronValue zero = layer.get(0, 0).zero();
		int width = layer.columns();
		int height = layer.rows();
		if (x + width() > width) {
			if (isPadZero())
				return x >= width ? null : zero;
			else
				x = width - width();
		}
		x = x < 0 ? 0 : x;
		if (y + height() > height) {
			if (isPadZero())
				if (y >= height)
					return y >= height ? null : zero;
			else
				y = height - height();
		}
		y = y < 0 ? 0 : y;

		NeuronValue result = layer.get(y, x);
		for (int i = 0; i < height(); i++) {
			for (int j = 0; j < width(); j++) {
				if (i == 0 && j == 0) continue;
				NeuronValue value = layer.get(y+i, x+j);
				result = result.max(value);
			}
		}
		
		return result;
	}

	
	/**
	 * Creating max pooling filter with specific kernel size.
	 * @param size specific kernel size.
	 * @return max pooling filter created from specific kernel size.
	 */
	public static MaxPoolFilter create(Size size) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		return new MaxPoolFilter(size.width, size.height);
	}


}
