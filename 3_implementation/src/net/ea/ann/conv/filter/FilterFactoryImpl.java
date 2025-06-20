/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.filter.beans.ZoomInFilter1D;
import net.ea.ann.conv.filter.beans.ZoomInFilter2D;
import net.ea.ann.conv.filter.beans.ZoomInFilter3D;
import net.ea.ann.conv.filter.beans.ZoomInFilter4D;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Size;

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
	 * Creator to create new neuron value.
	 */
	protected NeuronValueCreator creator = null;
	
	
	/**
	 * Constructor with creator to create new neuron value. 
	 * @param creator creator to create new neuron value.
	 */
	public FilterFactoryImpl(NeuronValueCreator creator) {
		super();
		this.creator = creator;
	}


	@Override
	public Filter1D product(double[] kernel, double weight) {
		return ProductFilter1D.create(kernel, weight, creator);
	}


	@Override
	public Filter2D product(double[][] kernel, double weight) {
		return ProductFilter2D.create(kernel, weight, creator);
	}


	@Override
	public Filter3D product(double[][][] kernel, double weight) {
		return ProductFilter3D.create(kernel, weight, creator);
	}


	@Override
	public Filter4D product(double[][][][] kernel, double weight) {
		return ProductFilter4D.create(kernel, weight, creator);
	}


	@Override
	public Filter1D product(int width) {
		return ProductFilter1D.create(new Size(width, 0), creator);
	}


	@Override
	public Filter2D product(int width, int height) {
		return ProductFilter2D.create(new Size(width, height), creator);
	}

	
	@Override
	public Filter3D product(int width, int height, int depth) {
		return ProductFilter3D.create(new Size(width, height, depth), creator);
	}


	@Override
	public Filter4D product(int width, int height, int depth, int time) {
		return ProductFilter4D.create(new Size(width, height, depth, time), creator);
	}


	@Override
	public Filter1D zoomOut(int width) {
		ProductFilter1D filter = (ProductFilter1D)product(width);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		NeuronValue zero = creator.newNeuronValue().zero();
		int mid = width/2;
		for (int j = 0; j < filter.width(); j++) {
			if (j == mid)
				filter.kernel[j] = unit;
			else
				filter.kernel[j] = zero;
		}
		
		filter.weight = unit;
		return filter;
	}


	@Override
	public Filter2D zoomOut(int width, int height) {
		ProductFilter2D filter = (ProductFilter2D)product(width, height);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		NeuronValue zero = creator.newNeuronValue().zero();
		int mid = Math.min(width/2, height/2);
		for (int i = 0; i < filter.height(); i++) {
			for (int j = 0; j < filter.width(); j++)
				if (i == j && j == mid)
					filter.kernel[i][j] = unit;
				else
					filter.kernel[i][j] = zero;
		}
		
		filter.weight = unit;
		return filter;
	}

	
	@Override
	public Filter3D zoomOut(int width, int height, int depth) {
		ProductFilter3D filter = (ProductFilter3D)product(width, height, depth);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		NeuronValue zero = creator.newNeuronValue().zero();
		int mid = Math.min(Math.min(width/2, height/2), depth/2);
		for (int i = 0; i < filter.depth(); i++) {
			for (int j = 0; j < filter.height(); j++) {
				for (int k = 0; k < filter.width(); k++)
					if (i == j && j == k && k == mid)
						filter.kernel[i][j][k] = unit;
					else
						filter.kernel[i][j][k] = zero;
			}
		}
		
		filter.weight = unit;
		return filter;
	}


	@Override
	public Filter4D zoomOut(int width, int height, int depth, int time) {
		ProductFilter4D filter = (ProductFilter4D)product(width, height, depth, time);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		NeuronValue zero = creator.newNeuronValue().zero();
		int mid = Math.min(Math.min(Math.min(width/2, height/2), depth/2), time/2);
		for (int h = 0; h < filter.time(); h++) {
			for (int i = 0; i < filter.depth(); i++) {
				for (int j = 0; j < filter.height(); j++) {
					for (int k = 0; k < filter.width(); k++)
						if (h == i && i == j && j == k && k == mid)
							filter.kernel[h][i][j][k] = unit;
						else
							filter.kernel[h][i][j][k] = zero;
				}
			}
		}
		
		filter.weight = unit;
		return filter;
	}


	@Override
	public Filter1D mean(int width) {
		ProductFilter1D filter = (ProductFilter1D)product(width);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		for (int j = 0; j < filter.width(); j++) filter.kernel[j] = unit;
		
		filter.weight = unit.valueOf(1.0/(filter.width()));
		return filter;
	}


	@Override
	public Filter2D mean(int width, int height) {
		ProductFilter2D filter = (ProductFilter2D)product(width, height);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		for (int i = 0; i < filter.height(); i++) {
			for (int j = 0; j < filter.width(); j++)
				filter.kernel[i][j] = unit;
		}
		
		filter.weight = unit.valueOf(1.0/(filter.width()*filter.height()));
		return filter;
	}


	@Override
	public Filter3D mean(int width, int height, int depth) {
		ProductFilter3D filter = (ProductFilter3D)product(width, height, depth);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		for (int i = 0; i < filter.depth(); i++) {
			for (int j = 0; j < filter.height(); j++) {
				for (int k = 0; k < filter.width(); k++)
					filter.kernel[i][j][k] = unit;
			}
		}
		
		filter.weight = unit.valueOf(1.0/(filter.width()*filter.height()*filter.depth()));
		return filter;
	}

	
	@Override
	public Filter4D mean(int width, int height, int depth, int time) {
		ProductFilter4D filter = (ProductFilter4D)product(width, height, depth, time);
		if (filter == null) return null;
		
		NeuronValue unit = creator.newNeuronValue().unit();
		for (int h = 0; h < filter.time(); h++) {
			for (int i = 0; i < filter.depth(); i++) {
				for (int j = 0; j < filter.height(); j++) {
					for (int k = 0; k < filter.width(); k++)
						filter.kernel[h][i][j][k] = unit;
				}
			}
		}
		
		filter.weight = unit.valueOf(1.0/(filter.width()*filter.height()*filter.depth()*filter.time()));
		return filter;
	}


	@Override
	public DeconvFilter1D zoomIn(int width) {
		return ZoomInFilter1D.create(new Size(width, 0));
	}


	@Override
	public DeconvFilter2D zoomIn(int width, int height) {
		return ZoomInFilter2D.create(new Size(width, height));
	}


	@Override
	public DeconvFilter3D zoomIn(int width, int height, int depth) {
		return ZoomInFilter3D.create(new Size(width, height, depth));
	}


	@Override
	public DeconvFilter4D zoomIn(int width, int height, int depth, int time) {
		return ZoomInFilter4D.create(new Size(width, height, depth, time));
	}


	@Override
	public DeconvConvFilter1D deconvConv(ProductFilter1D convFilter) {
		return DeconvConvFilter1DImpl.create(convFilter);
	}


	@Override
	public DeconvConvFilter2D deconvConv(ProductFilter2D convFilter) {
		return DeconvConvFilter2DImpl.create(convFilter);
	}


	@Override
	public DeconvConvFilter3D deconvConv(ProductFilter3D convFilter) {
		return DeconvConvFilter3DImpl.create(convFilter);
	}


	@Override
	public DeconvConvFilter4D deconvConv(ProductFilter4D convFilter) {
		return DeconvConvFilter4DImpl.create(convFilter);
	}

	
}
