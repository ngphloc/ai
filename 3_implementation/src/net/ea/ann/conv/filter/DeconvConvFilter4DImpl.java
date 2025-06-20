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
import net.ea.ann.conv.ConvLayerSingle3D;
import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Cube;

/**
 * This class represents the default deconvolutional filter based on a convolutional filter in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DeconvConvFilter4DImpl extends AbstractDeconvFilter4D implements DeconvConvFilter4D {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional filter.
	 */
	protected ProductFilter4D convFilter = null;
	
	
	/**
	 * Constructor with specified product filter.
	 * @param convFilter specified product filter.
	 */
	protected DeconvConvFilter4DImpl(ProductFilter4D convFilter) {
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
	public int depth() {
		return convFilter.depth();
	}


	@Override
	public int time() {
		return convFilter.time();
	}

	
	@Override
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer) {
		return convFilter.apply(x, y, layer);
	}


	@Override
	public NeuronValue apply(int x, int y, int z, ConvLayerSingle3D layer) {
		return convFilter.apply(x, y, z, layer);
	}


	@Override
	public NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer) {
		return convFilter.apply(x, y, z, t, layer);
	}

	
	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer, int nextIndex, ConvLayerSingle1D nextLayer) {
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle2D)) return apply(x, layer);
		if (layer == null || !(layer instanceof ConvLayerSingle2D)) return apply(x, layer);
		return apply(x, 0, (ConvLayerSingle2D)layer, nextIndex, 0, (ConvLayerSingle2D)nextLayer);
	}


	@Override
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer, int nextX, int nextY, ConvLayerSingle2D nextLayer) {
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle3D)) return apply(x, y, layer);
		if (layer == null || !(layer instanceof ConvLayerSingle3D)) return apply(x, y, layer);
		return apply(x, y, 0, (ConvLayerSingle3D)layer, nextX, nextY, 0, (ConvLayerSingle3D)nextLayer);
	}


	@Override
	public NeuronValue apply(int x, int y, int z, ConvLayerSingle3D layer, int nextX, int nextY, int nextZ, ConvLayerSingle3D nextLayer) {
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle4D)) return apply(x, y, z, layer);
		if (layer == null || !(layer instanceof ConvLayerSingle4D)) return apply(x, y, z, layer);
		return apply(x, y, z, 0, (ConvLayerSingle4D)layer, nextX, nextY, nextZ, 0, (ConvLayerSingle4D)nextLayer);
	}

	
	@Override
	public NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer, int nextX, int nextY, int nextZ, int nextT, ConvLayerSingle4D nextLayer) {
		if (layer == null && nextLayer == null) return null;
		if (nextLayer == null) return apply(x, y, z, t, layer);
		
		int filterStrideWidth = convFilter.getStrideWidth();
		int filterStrideHeight = convFilter.getStrideHeight();
		int filterStrideDepth = convFilter.getStrideDepth();
		int filterStrideTime = convFilter.getStrideTime();
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		int nextDepth = nextLayer.getDepth();
		int nextTime = nextLayer.getTime();
		
		int kernelWidth = convFilter.width();
		int kernelHeight = convFilter.height();
		int kernelDepth = convFilter.depth();
		int kernelTime = convFilter.time();
		Cube nextRegion = new Cube(x*filterStrideWidth, y*filterStrideHeight, z*filterStrideDepth, t*filterStrideTime,
			kernelWidth, kernelHeight, kernelDepth, kernelTime);
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
		if (nextRegion.z + kernelDepth > nextDepth) {
			if (nextLayer.isPadZeroFilter()) {
				if (nextRegion.z >= nextDepth)
					return null;
				else
					return nextLayer.newNeuronValue().zero();
			}
			else
				nextRegion.z = nextDepth - kernelDepth;
		}
		if (nextRegion.t + kernelTime > nextTime) {
			if (nextLayer.isPadZeroFilter()) {
				if (nextRegion.t >= nextTime)
					return null;
				else
					return nextLayer.newNeuronValue().zero();
			}
			else
				nextRegion.t = nextTime - kernelTime;
		}
		
		if (nextRegion.x < 0 || nextRegion.x >= nextWidth || nextRegion.y < 0 || nextRegion.y >= nextHeight || nextRegion.z < 0 || nextRegion.z >= nextDepth || nextRegion.t < 0 || nextRegion.t >= nextTime)
			return null;
		if (!nextRegion.contains(nextX, nextY, nextZ, nextT))
			return null;
			
		NeuronValue nextResult = nextLayer.newNeuronValue().zero();
		NeuronValue value0 = layer.get(x, y, z, t).getValue();
		int kernelX = -1;
		int kernelY = -1;
		int kernelZ = -1;
		int kernelT = -1;
		for (int h = 0; h < kernelTime; h++) {
			int T = nextRegion.t + h;
			if (T == nextT) kernelT = h;
			for (int i = 0; i < kernelDepth; i++) {
				int Z = nextRegion.z + i;
				if (Z == nextZ) kernelZ = i;
				for (int j = 0; j < kernelHeight; j++) {
					int Y = nextRegion.y + j;
					if (Y == nextY) kernelY = j;
					for (int k = 0; k < kernelWidth; k++) {
						int X = nextRegion.x + k;
						if (X == nextX) kernelX = k;
						
						if (X == nextX && Y == nextY && Z == nextZ && T == nextT)
							continue;
						
						NeuronValue value = nextLayer.get(X, Y, Z, T).getValue();
						value = value == null ? value0 : value; //Smoothing trick.
						nextResult = nextResult.add(value.multiply(convFilter.kernel[h][i][j][k]));
					}
				}
			}
		}
		nextResult = nextResult.multiply(convFilter.weight);
		
		NeuronValue xyztWeight = null;
		if (kernelX >= 0 && kernelY >= 0 && kernelZ >= 0 && kernelT >= 0)
			xyztWeight = convFilter.kernel[kernelT][kernelZ][kernelY][kernelX];
		
		NeuronValue result = layer.get(x, y, z, t).getValue();
		if (xyztWeight != null && xyztWeight.canInvert()) {
			result = result.subtract(nextResult);
			result = result.divide(xyztWeight.multiply(convFilter.weight));
		}
		
		return result;
	}


	/**
	 * Creating the deconvolutional filter based on a convolutional filter.
	 * @param convFilter specified product filter.
	 * @return the deconvolutional filter based on a convolutional filter.
	 */
	public static DeconvConvFilter4D create(ProductFilter4D convFilter) {
		if (convFilter == null)
			return null;
		else
			return new DeconvConvFilter4DImpl(convFilter);
	}


}
