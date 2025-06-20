/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Rectangle;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents the default deconvolutional filter based on a convolutional filter in 1D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DeconvConvFilter1DImpl extends AbstractDeconvFilter1D implements DeconvConvFilter1D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional filter.
	 */
	protected ProductFilter1D convFilter = null;

	
	/**
	 * Constructor with specified product filter.
	 * @param convFilter specified product filter.
	 */
	protected DeconvConvFilter1DImpl(ProductFilter1D convFilter) {
		super();
		this.convFilter = convFilter;
	}

	
	@Override
	public int width() {
		return convFilter.width();
	}

	
	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer) {
		return convFilter.apply(x, layer);
	}

	
	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer, int nextX, ConvLayerSingle1D nextLayer) {
		if (layer == null && nextLayer == null) return null;
		if (nextLayer == null) return apply(x, layer);
		
		int filterStrideWidth = convFilter.getStrideWidth();
		int nextWidth = nextLayer.getWidth();
		
		int kernelWidth = convFilter.width();
		Rectangle nextRegion = new Rectangle(new Point(x * filterStrideWidth, 0),
			new Dimension(kernelWidth, 0));
		if (nextRegion.x + kernelWidth > nextWidth) {
			if (nextLayer.isPadZeroFilter()) {
				if (nextRegion.x >= nextWidth)
					return null;
				else
					return nextLayer.newNeuronValue().zero();
			}
			else
				nextRegion.x = nextWidth - kernelWidth;
		}
		
		if (nextRegion.x < 0 || nextRegion.x >= nextWidth)
			return null;
		if (nextX < nextRegion.x || nextRegion.x + nextRegion.width <= nextX)
			return null;
			
		NeuronValue nextResult = nextLayer.newNeuronValue().zero();
		NeuronValue value0 = layer.get(x).getValue();
		int kernelX = -1;
		for (int j = 0; j < kernelWidth; j++) {
			int X = nextRegion.x + j;
			if (X == nextX) {
				kernelX = j;
				continue;
			}
			
			NeuronValue value = nextLayer.get(X).getValue(); //The null case is special case only for forwarding.
			value = value == null ? value0 : value; //Smoothing trick.
			nextResult = nextResult.add(value.multiply(convFilter.kernel[j]));
		}
		nextResult = nextResult.multiply(convFilter.weight);
		
		NeuronValue xWeight = null;
		if (kernelX >= 0 ) xWeight = convFilter.kernel[kernelX];
		
		NeuronValue result = layer.get(x).getValue();
		if (xWeight != null && xWeight.canInvert()) {
			result = result.subtract(nextResult);
			result = result.divide(xWeight.multiply(convFilter.weight));
		}
		
		return result;
	}

	
	/**
	 * Creating the deconvolutional filter based on a convolutional filter.
	 * @param convFilter specified product filter.
	 * @return the deconvolutional filter based on a convolutional filter.
	 */
	public static DeconvConvFilter1D create(ProductFilter1D convFilter) {
		if (convFilter == null)
			return null;
		else
			return new DeconvConvFilter1DImpl(convFilter);
	}


}
