/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents a filter with function in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FunctionFilter2D extends AbstractFilter2D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Internal function.
	 */
	protected Function f = null;
	
	
	/**
	 * Constructor with specified function.
	 * @param f specified function.
	 */
	protected FunctionFilter2D(Function f) {
		super();
		this.f = f;
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
		
		NeuronValue result = f.evaluate(layer.get(x, y).getValue());
		return result;
	}

	
	/**
	 * Creating function filter with specific function.
	 * @param f specific function.
	 * @return function filter created from specific function.
	 */
	public static FunctionFilter2D create(Function f) {
		if (f == null)
			return null;
		else
			return new FunctionFilter2D(f);
	}


}
