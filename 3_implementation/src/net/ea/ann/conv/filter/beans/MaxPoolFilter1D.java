/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.conv.filter.PoolFilter1D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents max pooling filter in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MaxPoolFilter1D extends PoolFilter1D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Kernel width.
	 */
	protected int width = 1;

	
	/**
	 * Constructor with kernel width.
	 * @param width kernel width.
	 */
	public MaxPoolFilter1D(int width) {
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

		NeuronValue result = layer.get(x).getValue();
		for (int j = 0; j < width(); j++) {
			if (j == 0) continue;
			
			NeuronValue value = layer.get(x+j).getValue();
			result = result.max(value);
		}
		
		return result;
	}


}
