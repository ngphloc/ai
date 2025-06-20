/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.conv.filter.AbstractFilter4D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents a zoom out as simple convolution filter in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ZoomOutFilter4D extends AbstractFilter4D {


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
	 * Zoom ratio in depth.
	 */
	protected int depth = 1;

	
	/**
	 * Zoom ratio in depth.
	 */
	protected int time = 1;

	
	/**
	 * Constructor with specific width, height, depth, and time.
	 * @param width specific width.
	 * @param height specific height.
	 * @param depth specific depth.
	 * @param time specific time.
	 */
	protected ZoomOutFilter4D(int width, int height, int depth, int time) {
		super();
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.time = time;
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
	public int depth() {
		return depth;
	}

	
	@Override
	public int time() {
		return time;
	}

	
	@Override
	public NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer) {
		if (layer == null) return null;
		
		int height = layer.getHeight();
		int width = layer.getWidth();
		int depth = layer.getDepth();
		int time = layer.getTime();
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
		
		if (z + depth() > depth) {
			if (layer.isPadZeroFilter()) {
				if (z >= depth)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				z = depth - depth();
		}
		z = z < 0 ? 0 : z;

		if (t + time() > time) {
			if (layer.isPadZeroFilter()) {
				if (t >= time)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				t = time - time();
		}
		t = t < 0 ? 0 : t;

		NeuronValue result = layer.get(x, y, z, t).getValue();
		return result;
	}
	

	/**
	 * Creating max zoom-out filter with specific size.
	 * @param size specific size.
	 * @return max zoom-out filter created from specific size.
	 */
	public static ZoomOutFilter4D create(Size size) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		if (size.depth < 1) size.depth = 1;
		if (size.time < 1) size.time = 1;
		
		return new ZoomOutFilter4D(size.width, size.height, size.depth, size.time);
	}


}
