/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import java.awt.Dimension;
import java.awt.Rectangle;
import java.io.Serializable;

import net.ea.ann.conv.ConvLayer1DAbstract;
import net.ea.ann.conv.ConvLayer1DImpl;
import net.ea.ann.conv.ConvLayer2DAbstract;
import net.ea.ann.conv.ConvLayer2DImpl;
import net.ea.ann.conv.ConvLayer3DAbstract;
import net.ea.ann.conv.ConvLayer3DImpl;
import net.ea.ann.conv.ConvLayer4DAbstract;
import net.ea.ann.conv.ConvLayer4DImpl;
import net.ea.ann.conv.filter.beans.MaxPoolFilter2D;
import net.ea.ann.conv.filter.beans.NegativeFilter2D;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;

/**
 * This class is an associator of filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FilterAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This class represent a plain raster as a pair of neuron array and size.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public class PlainRaster implements Serializable, Cloneable {

		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Neuron array as data.
		 */
		public NeuronValue[] data = null;
		
		/**
		 * Size of raster.
		 */
		public Size size = null;
		
		/**
		 * Constructor with data and size.
		 * @param data neuron array as data.
		 * @param size raster size
		 */
		public PlainRaster(NeuronValue[] data, Size size) {
			this.data = data;
			this.size = size;
		}
		
		/**
		 * Converting this plain raster to realistic raster.
		 * @param isNorm flag to indicate whether value is normalized in range [0, 1].
		 * @param defaultAlpha default alpha value.
		 * @return realistic raster.
		 */
		public Raster toRaster(boolean isNorm, int defaultAlpha) {
			return RasterAssoc.createRaster(data, neuronChannel, size, isNorm, defaultAlpha);
		}
		
		/**
		 * Converting this plain raster to realistic raster.
		 * @param isNorm flag to indicate whether value is normalized in range [0, 1].
		 * @return realistic raster.
		 */
		public Raster toRaster(boolean isNorm) {
			return toRaster(isNorm, Image.ALPHA_DEFAULT);
		}

	}
	
	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Neuron channel.
	 */
	protected Function activateRef = null;

	
	/**
	 * Internal filter.
	 */
	protected Filter filter = null;
	
	
	/**
	 * Constructor with specified neuron channel, activation channel, and filter.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param filter specified filter.
	 */
	public FilterAssoc(int neuronChannel, Function activateRef, Filter filter) {
		super();
		this.neuronChannel = neuronChannel;
		this.activateRef = activateRef;
		this.filter = filter;
	}

	
	/**
	 * Constructor with specified neuron channel, and filter.
	 * @param neuronChannel neuron channel.
	 * @param filter specified filter.
	 */
	public FilterAssoc(int neuronChannel, Filter filter) {
		this(neuronChannel, null, filter);
	}
	
	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply1D(NeuronValue[] source, int sourceSize, Rectangle filterRegion, boolean affected) {
		if (source == null || sourceSize <= 0) return null;
		
		ConvLayer1DAbstract thisLayer = ConvLayer1DImpl.create(neuronChannel, activateRef, sourceSize);
		SizeZoom zoom = Filter.zoomRatioOf(new Filter[] {filter});
		ConvLayer1DAbstract nextLayer = null;
		if (filter instanceof DeconvFilter)
			nextLayer = ConvLayer1DImpl.create(neuronChannel, activateRef, sourceSize*zoom.widthZoom);
		else
			nextLayer = ConvLayer1DImpl.create(neuronChannel, activateRef, sourceSize/zoom.widthZoom);
		
		NeuronRaster neuronRaster = ConvLayer1DAbstract.forward(source, thisLayer, nextLayer, filter, filterRegion, null, true);
		if (neuronRaster == null) return null;
		
		if (affected && zoom.widthZoom == 1) {
			NeuronValue[] data = nextLayer.getData();
			RasterAssoc.copyRange1D(data, 0, Math.min(data.length, source.length), source);
		}
		return new PlainRaster(neuronRaster.getData(), neuronRaster.getSize());
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @return filtered data.
	 */
	public PlainRaster apply1D(NeuronValue[] source, int sourceSize, Rectangle filterRegion) {
		return apply1D(source, sourceSize, filterRegion, true);
	}
	
	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply1D(NeuronValue[] source, int sourceSize, boolean affected) {
		return apply1D(source, sourceSize, (Rectangle)null, affected);
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @return filtered data.
	 */
	public PlainRaster apply1D(NeuronValue[] source, int sourceSize) {
		return apply1D(source, sourceSize, (Rectangle)null, true);
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply2D(NeuronValue[] source, Dimension sourceSize, Rectangle filterRegion, boolean affected) {
		if (sourceSize.height <= 1) return apply1D(source, sourceSize.width, filterRegion, affected);
		if (source == null || sourceSize == null || sourceSize.width <= 0 || sourceSize.height <= 0) return null;
		
		ConvLayer2DAbstract thisLayer = ConvLayer2DImpl.create(neuronChannel, activateRef, sourceSize.width, sourceSize.height);
		SizeZoom zoom = Filter.zoomRatioOf(new Filter[] {filter});
		ConvLayer2DAbstract nextLayer = null;
		if (filter instanceof DeconvFilter)
			nextLayer = ConvLayer2DImpl.create(neuronChannel, activateRef, sourceSize.width*zoom.widthZoom, sourceSize.height*zoom.heightZoom);
		else
			nextLayer = ConvLayer2DImpl.create(neuronChannel, activateRef, sourceSize.width/zoom.widthZoom, sourceSize.height/zoom.heightZoom);
		
		NeuronRaster neuronRaster = ConvLayer2DAbstract.forward(source, thisLayer, nextLayer, filter, filterRegion, null, true);
		if (neuronRaster == null) return null;
		
		if (affected && zoom.widthZoom == 1 && zoom.heightZoom == 1) {
			NeuronValue[] data = nextLayer.getData();
			RasterAssoc.copyRange1D(data, 0, Math.min(data.length, source.length), source);
		}
		return new PlainRaster(neuronRaster.getData(), neuronRaster.getSize());
	}
	

	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @return filtered data.
	 */
	public PlainRaster apply2D(NeuronValue[] source, Dimension sourceSize, Rectangle filterRegion) {
		return apply2D(source, sourceSize, filterRegion, true);
	}
	
	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply2D(NeuronValue[] source, Dimension sourceSize, boolean affected) {
		return apply2D(source, sourceSize, (Rectangle)null, affected);
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @return filtered data.
	 */
	public PlainRaster apply2D(NeuronValue[] source, Dimension sourceSize) {
		return apply2D(source, sourceSize, (Rectangle)null, true);
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply3D(NeuronValue[] source, Size sourceSize, Cube filterRegion, boolean affected) {
		if (sourceSize.depth <= 1) return apply2D(source, sourceSize,
			filterRegion != null ? filterRegion.toRectangle() : null,
			affected);
		if (source == null || sourceSize == null || sourceSize.width <= 0 || sourceSize.height <= 0 || sourceSize.depth <= 0) return null;
		
		ConvLayer3DAbstract thisLayer = ConvLayer3DImpl.create(neuronChannel, activateRef, sourceSize.width, sourceSize.height, sourceSize.depth, filter);
		SizeZoom zoom = Filter.zoomRatioOf(new Filter[] {filter});
		ConvLayer3DAbstract nextLayer = null;
		if (filter instanceof DeconvFilter)
			nextLayer = ConvLayer3DImpl.create(neuronChannel, activateRef, sourceSize.width*zoom.widthZoom, sourceSize.height*zoom.heightZoom, sourceSize.depth*zoom.depthZoom);
		else
			nextLayer = ConvLayer3DImpl.create(neuronChannel, activateRef, sourceSize.width/zoom.widthZoom, sourceSize.height/zoom.heightZoom, sourceSize.depth/zoom.depthZoom);
		
		NeuronRaster neuronRaster = ConvLayer3DAbstract.forward(source, thisLayer, nextLayer, filter, filterRegion, null, true);
		if (neuronRaster == null) return null;
		
		if (affected && zoom.widthZoom == 1 && zoom.heightZoom == 1 && zoom.depthZoom == 1) {
			NeuronValue[] data = nextLayer.getData();
			RasterAssoc.copyRange1D(data, 0, Math.min(data.length, source.length), source);
		}
		return new PlainRaster(neuronRaster.getData(), neuronRaster.getSize());
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @return filtered data.
	 */
	public PlainRaster apply3D(NeuronValue[] source, Size sourceSize, Cube filterRegion) {
		return apply3D(source, sourceSize, filterRegion, true);
	}
	

	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply3D(NeuronValue[] source, Size sourceSize, boolean affected) {
		return apply3D(source, sourceSize, (Cube)null, affected);
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @return filtered data.
	 */
	public PlainRaster apply3D(NeuronValue[] source, Size sourceSize) {
		return apply3D(source, sourceSize, (Cube)null, true);
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply4D(NeuronValue[] source, Size sourceSize, Cube filterRegion, boolean affected) {
		if (sourceSize.time <= 1) return apply3D(source, sourceSize, filterRegion, affected);
		if (source == null || sourceSize == null || sourceSize.width <= 0 || sourceSize.height <= 0 || sourceSize.depth <= 0 || sourceSize.time <= 0) return null;
		
		ConvLayer4DAbstract thisLayer = ConvLayer4DImpl.create(neuronChannel, activateRef, sourceSize.width, sourceSize.height, sourceSize.depth, sourceSize.time, filter);
		SizeZoom zoom = Filter.zoomRatioOf(new Filter[] {filter});
		ConvLayer4DAbstract nextLayer = null;
		if (filter instanceof DeconvFilter)
			nextLayer = ConvLayer4DImpl.create(neuronChannel, activateRef, sourceSize.width*zoom.widthZoom, sourceSize.height*zoom.heightZoom, sourceSize.depth*zoom.depthZoom, sourceSize.time*zoom.timeZoom);
		else
			nextLayer = ConvLayer4DImpl.create(neuronChannel, activateRef, sourceSize.width/zoom.widthZoom, sourceSize.height/zoom.heightZoom, sourceSize.depth/zoom.depthZoom, sourceSize.time/zoom.timeZoom);
		
		NeuronRaster neuronRaster = ConvLayer4DAbstract.forward(source, thisLayer, nextLayer, filter, filterRegion, null, true);
		if (neuronRaster == null) return null;
		
		if (affected && zoom.widthZoom == 1 && zoom.heightZoom == 1 && zoom.depthZoom == 1 && zoom.timeZoom == 1) {
			NeuronValue[] data = nextLayer.getData();
			RasterAssoc.copyRange1D(data, 0, Math.min(data.length, source.length), source);
		}
		return new PlainRaster(neuronRaster.getData(), neuronRaster.getSize());
	}

	
	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param filterRegion filtered region.
	 * @return filtered data.
	 */
	public PlainRaster apply4D(NeuronValue[] source, Size sourceSize, Cube filterRegion) {
		return apply4D(source, sourceSize, filterRegion, true);
	}


	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @param affected source is affected by filtering if this flag is true.
	 * @return filtered data.
	 */
	public PlainRaster apply4D(NeuronValue[] source, Size sourceSize, boolean affected) {
		return apply4D(source, sourceSize, (Cube)null, affected);
	}


	/**
	 * Applying filter to neuron array as data source.
	 * @param source neuron array as data source.
	 * @param sourceSize size of source.
	 * @return filtered data.
	 */
	public PlainRaster apply4D(NeuronValue[] source, Size sourceSize) {
		return apply4D(source, sourceSize, (Cube)null, true);
	}


	/**
	 * Creating arrays of filters for extracting features in 2D space.
	 * @param creator specified neuron value creator.
	 * @return arrays of filters for extracting features in 2D space.
	 */
	public static Filter[][] createFeatureExtractor2D(NeuronValueCreator creator) {
		ProductFilter2D blur = ProductFilter2D.create(new double[][] {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, 1.0/9.0, creator);
		ProductFilter2D sharpen = ProductFilter2D.create(new double[][] {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}, 1, creator);
//		ProductFilter2D edgeDetect = ProductFilter2D.create(new double[][] {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}, 1.0, creator);
		MaxPoolFilter2D maxPool = MaxPoolFilter2D.create(new Size(3, 3));
		
		blur.setStrideWidth(1); blur.setStrideHeight(1);
		sharpen.setStrideWidth(1); sharpen.setStrideHeight(1);
//		edgeDetect.setStrideWidth(1); edgeDetect.setStrideHeight(1);
		
//		return new Filter[][] {{blur, sharpen, edgeDetect}, {maxPool, maxPool, maxPool}};
		return new Filter[][] {{blur, sharpen}, {maxPool, maxPool}};
	}


	/**
	 * Creating arrays of filters for extracting features in 2D space.
	 * @param creator specified neuron value creator.
	 * @return arrays of filters for extracting features in 2D space.
	 */
	public static Filter[][] createNormFeatureExtractor2D(NeuronValueCreator creator) {
		ProductFilter2D blur = ProductFilter2D.create(new double[][] {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, 1.0/9.0, creator);
		ProductFilter2D sharpen = ProductFilter2D.create(new double[][] {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}, 1, creator);
		NegativeFilter2D negative = NegativeFilter2D.create(creator.newNeuronValue().unit());
		MaxPoolFilter2D maxPool = MaxPoolFilter2D.create(new Size(3, 3));
		
		blur.setStrideWidth(1); blur.setStrideHeight(1);
		sharpen.setStrideWidth(1); sharpen.setStrideHeight(1);
		
		return new Filter[][] {{blur, sharpen, negative}, {maxPool, maxPool, maxPool}};
	}


}
