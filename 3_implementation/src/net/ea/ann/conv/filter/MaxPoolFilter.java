/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayer;
import net.ea.ann.core.NeuronValue;

/**
 * This class represents max pooling filter.
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
	protected MaxPoolFilter(int width, int height) {
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
	public NeuronValue apply(int x, int y, ConvLayer layer) {
		if (layer == null) return null;
		
		int height = layer.getHeight();
		int width = layer.getWidth();
		if (x + width() > width) x = width - width();
		x = x < 0 ? 0 : x;
		if (y + height() > height) y = height - height();
		y = y < 0 ? 0 : y;

		NeuronValue result = layer.get(x, y).getValue();
		result = result.max(result);
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
	 * Creating max pooling filter with specific kernel width and height.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @return max pooling filter created from specific kernel width and height.
	 */
	public static MaxPoolFilter create(int width, int height) {
		if (width < 1) width = 1;
		if (height < 1) height = 1;
		
		return new MaxPoolFilter(width, height);
	}


}
