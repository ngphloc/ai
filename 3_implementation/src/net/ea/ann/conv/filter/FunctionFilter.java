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
import net.ea.ann.core.function.Function;

/**
 * This class represents a filter with function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FunctionFilter extends AbstractFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Internal function.
	 */
	protected Function f = null;
	
	
	/**
	 * Default constructor.
	 */
	protected FunctionFilter(Function f) {
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
	public NeuronValue apply(int x, int y, ConvLayer layer) {
		if (layer == null) return null;
		
		int height = layer.getHeight();
		int width = layer.getWidth();
		if (x >= width) x = width - 1;
		x = x < 0 ? 0 : x;
		if (y >= height) y = height - 1;
		y = y < 0 ? 0 : y;
		
		NeuronValue result = f.eval(layer.get(x, y).getValue());
		return result;
	}

	
	/**
	 * Creating function filter with specific function.
	 * @param f specific function.
	 * @return function filter created from specific function.
	 */
	public static FunctionFilter create(Function f) {
		if (f == null)
			return null;
		else
			return new FunctionFilter(f);
	}


}
