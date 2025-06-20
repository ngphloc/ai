/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.conv.filter.PoolFilter4D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents max pooling filter in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MaxPoolFilter4D extends PoolFilter4D {


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
	 * Kernel depth.
	 */
	protected int depth = 1;

	
	/**
	 * Kernel depth.
	 */
	protected int time = 1;

	
	/**
	 * Constructor with kernel width, height, depth, and time.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @param depth kernel depth.
	 * @param time kernel time.
	 */
	protected MaxPoolFilter4D(int width, int height, int depth, int time) {
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
		
		int width = layer.getWidth();
		int height = layer.getHeight();
		int depth = layer.getDepth();
		int time = layer.getTime();
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
		result = result.max(result);
		for (int h = 0; h < time(); h++) {
			for (int i = 0; i < depth(); i++) {
				for (int j = 0; j < height(); j++) {
					for (int k = 0; k < width(); k++) {
						if (i == 0 && j == 0 && k == 0) continue;
						
						NeuronValue value = layer.get(x+k, y+j, z+i, t+h).getValue();
						result = result.max(value);
					}
				}
			}
		}
		return result;
	}

	
	/**
	 * Creating max pooling filter with specific kernel size.
	 * @param size specific kernel size.
	 * @return max pooling filter created from specific kernel size.
	 */
	public static MaxPoolFilter4D create(Size size) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		if (size.depth < 1) size.depth = 1;
		if (size.time < 1) size.time = 1;
		
		return new MaxPoolFilter4D(size.width, size.height, size.depth, size.time);
	}


}
