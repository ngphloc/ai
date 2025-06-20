/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents a filter with function in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FunctionFilter4D extends AbstractFilter4D {


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
	protected FunctionFilter4D(Function f) {
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
	public int depth() {
		return 1;
	}

	
	@Override
	public NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer) {
		if (layer == null) return null;
		
		int width = layer.getWidth();
		int height = layer.getHeight();
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
		
		if (z >= depth) {
			if (layer.isPadZeroFilter())
				return null;
			else
				z = depth - 1;
		}
		z = z < 0 ? 0 : z;

		if (t >= time) {
			if (layer.isPadZeroFilter())
				return null;
			else
				t = time - 1;
		}
		t = t < 0 ? 0 : t;

		NeuronValue result = f.evaluate(layer.get(x, y, z, t).getValue());
		return result;
	}

	
	/**
	 * Creating function filter with specific function.
	 * @param f specific function.
	 * @return function filter created from specific function.
	 */
	public static FunctionFilter4D create(Function f) {
		if (f == null)
			return null;
		else
			return new FunctionFilter4D(f);
	}


}
