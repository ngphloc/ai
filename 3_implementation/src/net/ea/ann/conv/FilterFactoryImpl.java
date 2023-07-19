/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

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
	public Filter productFilter(int width, int height) {
		if (width <= 0 || height <= 0) return null;
		
		NeuronValue[][] kernel = new NeuronValue[height][width];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) kernel[i][j] = layer.newNeuronValue();
		}
		
		NeuronValue weight = layer.newNeuronValue().valueOf(1.0);
		return new ProductFilter(kernel, weight);
	}

	
	@Override
	public Filter meanFilter(int width, int height) {
		ProductFilter filter = (ProductFilter)productFilter(width, height);
		if (filter == null) return null;
		
		for (int i = 0; i < filter.height(); i++) {
			for (int j = 0; j < filter.width(); j++)
				filter.kernel[i][j] = layer.newNeuronValue().identity();
		}
		
		filter.weight = layer.newNeuronValue().valueOf(1.0/9.0);
		return filter;
	}

	
}
