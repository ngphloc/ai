/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.conv.filter.AbstractDeconvFilter1D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents a zoom in as simple deconvolution filter in 1D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ZoomInFilter1D extends AbstractDeconvFilter1D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zoom ratio in width.
	 */
	protected int width = 1;

	
	
	/**
	 * Constructor with specific width.
	 * @param width specific width.
	 */
	protected ZoomInFilter1D(int width) {
		super();
		this.width = width;
	}

	
	@Override
	public int width() {
		return width;
	}

	
	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer) {
		if (layer == null) return null;
		
		int width = layer.getWidth();
		if (x >= width) {
			if (layer.isPadZeroFilter())
				return null;
			else
				x = width - 1;
		}
		x = x < 0 ? 0 : x;
		
		NeuronValue result = layer.get(x).getValue();
		return result;
	}

	
	/**
	 * Creating max zoom-in filter with specific size.
	 * @param size specific size.
	 * @return max zoom-in filter created from specific size.
	 */
	public static ZoomInFilter1D create(Size size) {
		if (size.width < 1) size.width = 1;
		
		return new ZoomInFilter1D(size.width);
	}


}
