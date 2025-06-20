/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.conv.filter.AbstractDeconvFilter2D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents a zoom in as simple deconvolution filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ZoomInFilter2D extends AbstractDeconvFilter2D {


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
	protected ZoomInFilter2D(int width, int height) {
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
		
		int height = layer.getHeight();
		int width = layer.getWidth();
		if (x >= width) {
			if (layer.isPadZeroFilter())
				return null;
			else
				x = width - 1;
		}
		x = x < 0 ? 0 : x;
		if (y >= height) {
			if (layer.isPadZeroFilter())
				return null;
			else
				y = height - 1;
		}
		y = y < 0 ? 0 : y;
		
		NeuronValue result = layer.get(x, y).getValue();
		return result;
	}
	

	/**
	 * Creating max zoom-in filter with specific size.
	 * @param size specific size.
	 * @return max zoom-in filter created from specific size.
	 */
	public static ZoomInFilter2D create(Size size) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		
		return new ZoomInFilter2D(size.width, size.height);
	}


}
