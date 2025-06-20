/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import java.io.Serializable;

/**
 * This interface represents a factory to create filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface FilterFactory extends Serializable, Cloneable {

	
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @return product filter created from real kernel and weight.
	 */
	Filter1D product(double[] kernel, double weight);

	
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @return product filter created from real kernel and weight.
	 */
	Filter2D product(double[][] kernel, double weight);

		
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @return product filter created from real kernel and weight.
	 */
	Filter3D product(double[][][] kernel, double weight);

	
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @return product filter created from real kernel and weight.
	 */
	Filter4D product(double[][][][] kernel, double weight);

	
	/**
	 * Creating product filter.
	 * @param width kernel width.
	 * @return product filter.
	 */
	Filter1D product(int width);

	
	/**
	 * Creating product filter.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @return product filter.
	 */
	Filter2D product(int width, int height);

		
	/**
	 * Creating product filter.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @param depth kernel depth.
	 * @return product filter.
	 */
	Filter3D product(int width, int height, int depth);

	
	/**
	 * Creating product filter.
	 * @param width kernel width.
	 * @param height kernel height.
	 * @param depth kernel depth.
	 * @param time kernel time.
	 * @return product filter.
	 */
	Filter4D product(int width, int height, int depth, int time);

	
	/**
	 * Creating zooming out convolutional filter.
	 * @param width filter width.
	 * @return zooming out convolutional filter.
	 */
	Filter1D zoomOut(int width);

	
	/**
	 * Creating zooming out convolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @return zooming out convolutional filter.
	 */
	Filter2D zoomOut(int width, int height);

	
	/**
	 * Creating zooming out convolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @param depth filter depth.
	 * @return zooming out convolutional filter.
	 */
	Filter3D zoomOut(int width, int height, int depth);

	
	/**
	 * Creating zooming out convolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @param depth filter depth.
	 * @param time filter time.
	 * @return zooming out convolutional filter.
	 */
	Filter4D zoomOut(int width, int height, int depth, int time);
	
	
	/**
	 * Creating mean product filter.
	 * @param width filter width.
	 * @return mean product filter.
	 */
	Filter1D mean(int width);

	
	/**
	 * Creating mean product filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @return mean product filter.
	 */
	Filter2D mean(int width, int height);
	
	
	/**
	 * Creating mean product filter in 3D space.
	 * @param width filter width.
	 * @param height filter height.
	 * @param depth filter depth.
	 * @return mean product filter.
	 */
	Filter3D mean(int width, int height, int depth);
	
	
	/**
	 * Creating mean product filter in 4D space.
	 * @param width filter width.
	 * @param height filter height.
	 * @param depth filter depth.
	 * @param time filter time.
	 * @return mean product filter.
	 */
	Filter4D mean(int width, int height, int depth, int time);

	
	/**
	 * Creating zoom-in filter as simple deconvolutional filter.
	 * @param width filter width.
	 * @return zoom-in filter as simple deconvolutional filter.
	 */
	DeconvFilter1D zoomIn(int width);

	
	/**
	 * Creating zoom-in filter as simple deconvolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @return zoom-in filter as simple deconvolutional filter.
	 */
	DeconvFilter2D zoomIn(int width, int height);

	
	/**
	 * Creating zoom-in filter as simple deconvolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @param depth filter depth.
	 * @return zoom-in filter as simple deconvolutional filter.
	 */
	DeconvFilter3D zoomIn(int width, int height, int depth);
	
	
	/**
	 * Creating zoom-in filter as simple deconvolutional filter.
	 * @param width filter width.
	 * @param height filter height.
	 * @param depth filter depth.
	 * @param time filter time.
	 * @return zoom-in filter as simple deconvolutional filter.
	 */
	DeconvFilter4D zoomIn(int width, int height, int depth, int time);
	
	
	/**
	 * Creating deconvolutional filter based on a convolutional filter.
	 * @param convFilter specified convolutional filter.
	 * @return deconvolutional filter based on a convolutional filter.
	 */
	DeconvConvFilter1D deconvConv(ProductFilter1D convFilter);

	
	/**
	 * Creating deconvolutional filter based on a convolutional filter.
	 * @param convFilter specified convolutional filter.
	 * @return deconvolutional filter based on a convolutional filter.
	 */
	DeconvConvFilter2D deconvConv(ProductFilter2D convFilter);
	
	
	/**
	 * Creating deconvolutional filter based on a convolutional filter in 3D space.
	 * @param convFilter specified convolutional filter.
	 * @return deconvolutional filter based on a convolutional filter.
	 */
	DeconvConvFilter3D deconvConv(ProductFilter3D convFilter);


	/**
	 * Creating deconvolutional filter based on a convolutional filter in 3D space.
	 * @param convFilter specified convolutional filter.
	 * @return deconvolutional filter based on a convolutional filter.
	 */
	DeconvConvFilter4D deconvConv(ProductFilter4D convFilter);


}
