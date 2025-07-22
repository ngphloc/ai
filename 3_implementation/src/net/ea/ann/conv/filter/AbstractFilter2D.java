/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is an abstract implementation of filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class AbstractFilter2D extends AbstractFilter1D implements Filter2D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	protected AbstractFilter2D() {
		super();
	}


	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer) {
		if (!(layer instanceof ConvLayerSingle2D)) return null;
		return apply(x, 0, (ConvLayerSingle2D)layer);
	}


	@Override
	public NeuronValue[][] dKernel(int nextX, int nextY, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer) {
		throw new RuntimeException("Method Filter2D::dKernel(int, int, ConvLayerSingle2D, ConvLayerSingle2D) not implemented yet");
	}


	@Override
	public NeuronValue[][] dValue(int nextX, int nextY, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer) {
		throw new RuntimeException("Method Filter2D::dValue(int, int, ConvLayerSingle2D, ConvLayerSingle2D) not implemented yet");
	}


}
