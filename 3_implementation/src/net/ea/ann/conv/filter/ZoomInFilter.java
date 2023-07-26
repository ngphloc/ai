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
 * This class represents a zoom in as simple deconvolution filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ZoomInFilter extends AbstractDeconvFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zoom ratio in width.
	 */
	protected int width = 1;
	
	
	/**
	 * Zoom ratio in height.
	 */
	protected int height = 1;
	
	
	/**
	 * Constructor with specific width and height.
	 * @param width specific width.
	 * @param height specific height.
	 */
	public ZoomInFilter(int width, int height) {
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
		if (x >= width) x = width - 1;
		x = x < 0 ? 0 : x;
		if (y >= height) y = height - 1;
		y = y < 0 ? 0 : y;
		
		NeuronValue result = layer.get(x, y).getValue().duplicate();
		return result;
	}
	

	/**
	 * Creating max zoom-in filter with specific width and height.
	 * @param width specific width.
	 * @param height specific height.
	 * @return max zoom-in filter created from specific width and height.
	 */
	public static ZoomInFilter create(int width, int height) {
		if (width < 1) width = 1;
		if (height < 1) height = 1;
		
		return new ZoomInFilter(width, height);
	}


}
