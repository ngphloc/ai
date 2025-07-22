/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.conv.ConvSupporter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterAssoc;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;

/**
 * This class provides initialization methods to initialize convolutional generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvGenInitializer implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional generative model.
	 */
	protected ConvGenModel convGM = null;

	
	/**
	 * Constructor with convolutional generative model.
	 * @param convGM convolutional generative model.
	 */
	public ConvGenInitializer(ConvGenModel convGM) {
		this.convGM = convGM;
	}

	
	/**
	 * Setting thick-stack property.
	 * @param thickStack thick-stack property. In thick-stack mode (true), every stack in convolutional network should have more than one element layer.
	 * @return this initializer.
	 */
	public ConvGenInitializer setParamThickStack(boolean thickStack) {
		try {
			ConvGenSetting setting = convGM.getSetting();
			setting.thickStack = thickStack;
			convGM.setSetting(setting);
		}
		catch (Exception e) {Util.trace(e);}
		return this;
	}

	
	/**
	 * Concatenating filter arrays and filters.
	 * @param filterArrays arrays of filters. Filters in the same array have the same size.
	 * @param filters other filters.
	 * @param reverse reversed flag.
	 * @return arrays of concatenated filters.
	 */
	private static Filter[][] concat(Filter[][] filterArrays, Filter[] filters, boolean reverse) {
		 List<Filter[]> filtersList = Util.newList(0);
		 if (filterArrays == null || filterArrays.length == 0) {
			 if (filters != null && filters.length > 0) filtersList.add(filters);
		 }
		 else if (filters == null || filters.length == 0) {
			 for (Filter[] filterArray : filterArrays) filtersList.add(filterArray);
		 }
		 else if (reverse) {
			 filtersList.add(filters);
			 for (Filter[] filterArray : filterArrays) filtersList.add(filterArray);
		 }
		 else {
			 for (Filter[] filterArray : filterArrays) filtersList.add(filterArray);
			 filtersList.add(filters);
		 }
		 return filtersList.size() > 0 ? filtersList.toArray(new Filter[][] {}) : null;
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param deconvFilterArrays arrays of deconvolutional filters. Deconvolutional filters in the same array have the same size.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays, Filter[][] deconvFilterArrays, SizeZoom zoomOut, Size minSize) {
		if (sample == null || zDim <= 0) return false;
		Size size = RasterAssoc.getAverageSize(sample);
		if (size.width == 0 && size.height == 0 && size.depth == 0 && size.time == 0) return false;
		
		zoomOut = zoomOut != null ? zoomOut : SizeZoom.zoom(Size.unit());
		minSize = minSize != null ? minSize : Size.unit();
		SizeZoom sizeZ = RasterAssoc.calcFitSize(
			new SizeZoom(size.width, size.height, size.depth, size.time, zoomOut.widthZoom, zoomOut.heightZoom, zoomOut.depthZoom, zoomOut.timeZoom),
			minSize);
		
		try {
			convGM.reset();
			ConvGenSetting setting = convGM.getSetting();
			setting.width = sizeZ.width;
			setting.height = sizeZ.height;
			setting.depth = sizeZ.depth;
			setting.time = sizeZ.time;
			convGM.setSetting(setting);
		}
		catch (Exception e) {Util.trace(e);}

		Filter[] zoomConvFilters = null;
		Filter[] zoomDeconvFilters = null;
		if (convGM instanceof ConvSupporter) {
			ConvSupporter supporter = (ConvSupporter)convGM;
			FilterFactory factory = supporter.getFilterFactory();
			zoomConvFilters = Filter.calcZoomFilters(zoomOut, factory, true);
			zoomDeconvFilters = Filter.calcZoomFilters(zoomOut, factory, false);
		}

		try {
			if ((convFilterArrays == null || convFilterArrays.length == 0) && (deconvFilterArrays == null || deconvFilterArrays.length == 0))
				return convGM.initialize(zDim, zoomConvFilters, zoomDeconvFilters);
			else if ((convFilterArrays != null && convFilterArrays.length > 0) && (deconvFilterArrays != null && deconvFilterArrays.length > 0)) {
				Filter[][] convFilters = concat(convFilterArrays, zoomConvFilters, true);
				Filter[][] deconvFilters = concat(deconvFilterArrays, zoomDeconvFilters, false);
				return convGM.initialize(zDim, convFilters, deconvFilters);
			}
			else if (convFilterArrays != null && convFilterArrays.length > 0) {
				Filter[][] convFilters = concat(convFilterArrays, zoomConvFilters, true);
				return convGM.initialize(zDim, convFilters, null);
			}
			else {
				Filter[][] deconvFilters = concat(deconvFilterArrays, zoomDeconvFilters, false);
				return convGM.initialize(zDim, null, deconvFilters);
			}
		}
		catch (Exception e) {Util.trace(e);}
		
		return false;
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param deconvFilterArrays arrays of deconvolutional filters. Deconvolutional filters in the same array have the same size.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays, Filter[][] deconvFilterArrays, Size minSize) {
		return initialize(sample, zDim, convFilterArrays, deconvFilterArrays, null, minSize);
	}


	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param deconvFilterArrays arrays of deconvolutional filters. Deconvolutional filters in the same array have the same size.
	 * @param zoomOut zoom out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays, Filter[][] deconvFilterArrays, SizeZoom zoomOut) {
		return initialize(sample, zDim, convFilterArrays, deconvFilterArrays, zoomOut, null);
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param deconvFilterArrays arrays of deconvolutional filters. Deconvolutional filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) {
		return initialize(sample, zDim, convFilterArrays, deconvFilterArrays, (SizeZoom)null);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays) {
		return initialize(sample, zDim, convFilterArrays, (Filter[][])null);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays, SizeZoom zoomOut, Size minSize) {
		return initialize(sample, zDim, convFilterArrays, null, zoomOut, minSize);
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilterArrays arrays of convolutional filters. Convolutional filters in the same array have the same size.
	 * @param zoomOut zoom out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[][] convFilterArrays, SizeZoom zoomOut) {
		return initialize(sample, zDim, convFilterArrays, zoomOut, null);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, SizeZoom zoomOut, Size minSize) {
		Filter[][] convFilterArrays = null, deconvFilterArrays = null;
		if (convFilters != null && convFilters.length > 0) convFilterArrays = new Filter[][] {convFilters};
		if (deconvFilters != null && deconvFilters.length > 0) deconvFilterArrays = new Filter[][] {deconvFilters};
		return initialize(sample, zDim, convFilterArrays, deconvFilterArrays, zoomOut, minSize);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize) {
		return initialize(sample, zDim, convFilters, deconvFilters, null, minSize);
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param zoomOut zoom out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, SizeZoom zoomOut) {
		return initialize(sample, zDim, convFilters, deconvFilters, zoomOut, null);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters) {
		return initialize(sample, zDim, convFilters, deconvFilters, (SizeZoom)null);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters) {
		return initialize(sample, zDim, convFilters, (Filter[])null);
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters, SizeZoom zoomOut, Size minSize) {
		return initialize(sample, zDim, convFilters, null, zoomOut, minSize);
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param zoomOut zoom out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, Filter[] convFilters, SizeZoom zoomOut) {
		return initialize(sample, zDim, convFilters, zoomOut, null);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize) {
		return initialize(sample, zDim, (Filter[])null, (Filter[])null, zoomOut, minSize);
	}

	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param zoomOut zoom out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim, SizeZoom zoomOut) {
		return initialize(sample, zDim, zoomOut, null);
	}
	
	
	/**
	 * Initializing model from sample.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Iterable<Raster> sample, int zDim) {
		return initialize(sample, zDim, (SizeZoom)null);
	}
	
	
	/**
	 * Initializing model from sample with 2D feature extractor.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeFeatureExtractor2D(Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize) {
		NeuronValueCreator creator = convGM instanceof ConvSupporter ? ((ConvSupporter)convGM).getConvNeuronValueCreator() : null;
		if (creator == null) return false;
		Filter[][] convFilterArrays = FilterAssoc.createFeatureExtractor2D(creator);
		return initialize(sample, zDim, convFilterArrays, zoomOut, minSize);
	}
	
	
	/**
	 * Initializing model from sample with 2D feature extractor.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param minSize minimum size.
	 * @return true if initialization is successful.
	 */
	public boolean initializeFeatureExtractor2D(Iterable<Raster> sample, int zDim, Size minSize) {
		return initializeFeatureExtractor2D(sample, zDim, null, minSize);
	}
	
	
	/**
	 * Initializing model from sample with 2D feature extractor.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @param zoomOut zoom out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initializeFeatureExtractor2D(Iterable<Raster> sample, int zDim, SizeZoom zoomOut) {
		return initializeFeatureExtractor2D(sample, zDim, zoomOut, null);
		
	}
	
	
	/**
	 * Initializing model from sample with 2D feature extractor.
	 * @param sample raster sample.
	 * @param zDim Z dimension.
	 * @return true if initialization is successful.
	 */
	public boolean initializeFeatureExtractor2D(Iterable<Raster> sample, int zDim) {
		return initializeFeatureExtractor2D(sample, zDim, (SizeZoom)null);
		
	}


}
