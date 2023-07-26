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

/**
 * This interface is the default implementation of factory to create filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FilterFactoryImpl implements FilterFactory {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional layer.
	 */
	protected ConvLayer layer = null;
	
	
	/**
	 * Constructor with convolutional layer. 
	 * @param layer convolutional layer.
	 */
	public FilterFactoryImpl(ConvLayer layer) {
		this.layer = layer;
	}


	@Override
	public Filter product(int width, int height) {
		if (width <= 0 || height <= 0) return null;
		
		NeuronValue source = layer.newNeuronValue();
		NeuronValue[][] kernel = new NeuronValue[height][width];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) kernel[i][j] = source.zero();
		}
		
		NeuronValue weight = source.valueOf(1.0);
		return new ProductFilter(kernel, weight);
	}

	
	@Override
	public Filter zoomOut(int width, int height) {
		ProductFilter filter = (ProductFilter)product(width, height);
		if (filter == null) return null;
		
		NeuronValue source = layer.newNeuronValue();
		int mid = Math.min(width/2, height/2);
		for (int i = 0; i < filter.height(); i++) {
			for (int j = 0; j < filter.width(); j++)
				if (i == j && i == mid)
					filter.kernel[i][j] = source.identity();
				else
					filter.kernel[i][j] = source.zero();
		}
		
		filter.weight = source.valueOf(1);
		return filter;
	}

	
	@Override
	public Filter mean(int width, int height) {
		ProductFilter filter = (ProductFilter)product(width, height);
		if (filter == null) return null;
		
		NeuronValue source = layer.newNeuronValue();
		for (int i = 0; i < filter.height(); i++) {
			for (int j = 0; j < filter.width(); j++)
				filter.kernel[i][j] = source.identity();
		}
		
		filter.weight = source.valueOf(1.0/9.0);
		return filter;
	}


	@Override
	public Filter zoomIn(int width, int height) {
		return ZoomInFilter.create(width, height);
	}

	
}
