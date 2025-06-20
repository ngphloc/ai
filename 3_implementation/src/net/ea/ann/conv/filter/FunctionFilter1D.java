/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents a filter with function in 1D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FunctionFilter1D extends AbstractFilter1D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Internal function.
	 */
	protected Function f = null;

	
	/**
	 * Default constructor with specified function.
	 * @param f specified function.
	 */
	protected FunctionFilter1D(Function f) {
		super();
		this.f = f;
	}

	
	@Override
	public int width() {
		return 1;
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
		
		NeuronValue result = f.evaluate(layer.get(x).getValue());
		return result;
	}


	/**
	 * Creating function filter with specific function.
	 * @param f specific function.
	 * @return function filter created from specific function.
	 */
	public static FunctionFilter1D create(Function f) {
		if (f == null)
			return null;
		else
			return new FunctionFilter1D(f);
	}


}
