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
import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represents the default deconvolutional filter based on a convolutional filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DeconvConvFilter2DImpl extends AbstractDeconvFilter2D implements DeconvConvFilter2D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional filter.
	 */
	protected ProductFilter2D convFilter = null;
	
	
	/**
	 * Constructor with specified product filter.
	 * @param convFilter specified product filter.
	 */
	protected DeconvConvFilter2DImpl(ProductFilter2D convFilter) {
		super();
		this.convFilter = convFilter;
	}

	
	@Override
	public int width() {
		return convFilter.width();
	}

	
	@Override
	public int height() {
		return convFilter.height();
	}

	
	@Override
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer) {
		return convFilter.apply(x, y, layer);
	}


	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer, int nextIndex, ConvLayerSingle1D nextLayer) {
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle2D)) return apply(x, layer);
		if (layer == null || !(layer instanceof ConvLayerSingle2D)) return apply(x, layer);
		return apply(x, 0, (ConvLayerSingle2D)layer, nextIndex, 0, (ConvLayerSingle2D)nextLayer);
	}


	@Override
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer, int nextX, int nextY, ConvLayerSingle2D nextLayer) {
		if (layer == null && nextLayer == null) return null;
		if (nextLayer == null) return apply(x, y, layer);
		
		int filterStrideWidth = convFilter.getStrideWidth();
		int filterStrideHeight = convFilter.getStrideHeight();
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		
		int kernelWidth = convFilter.width();
		int kernelHeight = convFilter.height();
		Rectangle nextRegion = new Rectangle(new Point(x * filterStrideWidth, y * filterStrideHeight),
			new Dimension(kernelWidth, kernelHeight));
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
		if (nextRegion.y + kernelHeight > nextHeight) {
			if (nextLayer.isPadZeroFilter()) {
				if (nextRegion.y >= nextHeight)
					return null;
				else
					return nextLayer.newNeuronValue().zero();
			}
			else
				nextRegion.y = nextHeight - kernelHeight;
		}
		
		if (nextRegion.x < 0 || nextRegion.x >= nextWidth || nextRegion.y < 0 || nextRegion.y >= nextHeight)
			return null;
		if (!nextRegion.contains(nextX, nextY))
			return null;
			
		NeuronValue nextResult = nextLayer.newNeuronValue().zero();
		NeuronValue value0 = layer.get(x, y).getValue();
		int kernelX = -1;
		int kernelY = -1;
		for (int i = 0; i < kernelHeight; i++) {
			for (int j = 0; j < kernelWidth; j++) {
				int X = nextRegion.x + j;
				int Y = nextRegion.y + i;
				if (X == nextX) kernelX = j;
				if (Y == nextY) kernelY = i;
				
				if (X == nextX && Y == nextY)
					continue;
				
				NeuronValue value = nextLayer.get(X, Y).getValue(); //The null case is special case only for forwarding.
				value = value == null ? value0 : value; //Smoothing trick.
				nextResult = nextResult.add(value.multiply(convFilter.kernel[i][j]));
			}
		}
		nextResult = nextResult.multiply(convFilter.weight);
		
		NeuronValue xyWeight = null;
		if (kernelX >= 0 && kernelY >= 0) xyWeight = convFilter.kernel[kernelY][kernelX];
		
		NeuronValue result = layer.get(x, y).getValue();
		if (xyWeight != null && xyWeight.canInvert()) {
			result = result.subtract(nextResult);
			result = result.divide(xyWeight.multiply(convFilter.weight));
		}
		
		return result;
	}


	/**
	 * Creating the deconvolutional filter based on a convolutional filter.
	 * @param convFilter specified product filter.
	 * @return the deconvolutional filter based on a convolutional filter.
	 */
	public static DeconvConvFilter2D create(ProductFilter2D convFilter) {
		if (convFilter == null)
			return null;
		else
			return new DeconvConvFilter2DImpl(convFilter);
	}


}
