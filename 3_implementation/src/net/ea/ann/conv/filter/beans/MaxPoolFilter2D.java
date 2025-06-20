/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.conv.filter.PoolFilter2D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents max pooling filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MaxPoolFilter2D extends PoolFilter2D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Kernel width.
	 */
	protected int width = 1;
	
	
	/**
	 * Kernel height.
	 */
	protected int height = 1;

	
	/**
	 * Constructor with kernel width and height.
	 * @param width kernel width.
	 * @param height kernel height.
	 */
	protected MaxPoolFilter2D(int width, int height) {
		super();
		this.width = width;
		this.height = height;
	}

	
	@Override
	public int width() {
		return width;
	}


	@Override
	public int height() {
		return height;
	}


	@Override
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer) {
		if (layer == null) return null;
		
		int width = layer.getWidth();
		int height = layer.getHeight();
		if (x + width() > width) {
			if (layer.isPadZeroFilter()) {
				if (x >= width)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				x = width - width();
		}
		x = x < 0 ? 0 : x;
		if (y + height() > height) {
			if (layer.isPadZeroFilter()) {
				if (y >= height)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				y = height - height();
		}
		y = y < 0 ? 0 : y;

		NeuronValue result = layer.get(x, y).getValue();
		for (int i = 0; i < height(); i++) {
			for (int j = 0; j < width(); j++) {
				if (i == 0 && j == 0) continue;
				
				NeuronValue value = layer.get(x+j, y+i).getValue();
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
	public static MaxPoolFilter2D create(Size size) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		
		return new MaxPoolFilter2D(size.width, size.height);
	}


}
