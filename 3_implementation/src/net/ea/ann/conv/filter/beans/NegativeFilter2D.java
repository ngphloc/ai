/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.beans;

import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.conv.filter.AbstractFilter2D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents negative filter in 1D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NegativeFilter2D extends AbstractFilter2D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Maximum value.
	 */
	protected NeuronValue max = null;
	
	
	/**
	 * Constructor with specified maximum value.
	 * @param max maximum value.
	 */
	protected NegativeFilter2D(NeuronValue max) {
		super();
		this.max = max;
	}

	
	@Override
	public int width() {
		return 1;
	}


	@Override
	public int height() {
		return 1;
	}

	
	@Override
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer) {
		if (layer == null) return null;
		
		int width = layer.getWidth();
		int height = layer.getHeight();
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
		
		NeuronValue result = max.subtract(layer.get(x, y).getValue());
		return result;
	}

	
	/**
	 * Creating negative filter with specific maximum.
	 * @param max specific maximum.
	 * @return negative filter with specific maximum.
	 */
	public static NegativeFilter2D create(NeuronValue max) {
		return max != null ? new NegativeFilter2D(max) : null;
	}


}
