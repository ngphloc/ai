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
 * This class represents a product filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProductFilter extends AbstractFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected NeuronValue[][] kernel = null;
	
	
	/**
	 * Kernel weight.
	 */
	protected NeuronValue weight = null;
	
	
	/**
	 * Constructor with kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 */
	public ProductFilter(NeuronValue[][] kernel, NeuronValue weight) {
		super();
		this.kernel = kernel;
		this.weight = weight;
	}


	@Override
	public int width() {
		return kernel[0].length;
	}


	@Override
	public int height() {
		return kernel.length;
	}


	@Override
	public NeuronValue apply(int x, int y, ConvLayer layer) {
		if (layer == null) return null;
		
		int kernelWidth = width();
		int kernelHeight = height();
		int height = layer.getHeight();
		int width = layer.getWidth();
		if (x + kernelWidth > width) x = width - kernelWidth;
		x = x < 0 ? 0 : x;
		if (y + kernelHeight > height) y = height - kernelHeight;
		y = y < 0 ? 0 : y;
		
		NeuronValue result = layer.newNeuronValue().zero();
		for (int i = 0; i < kernelHeight; i++) {
			for (int j = 0; j < kernelWidth; j++) {
				NeuronValue value = layer.get(x+j, y+i).getValue();
				result = result.add(value.multiply(kernel[i][j]));
			}
		}
		
		return result.multiply(weight);
	}
	
	
	/**
	 * Creating kernel filter with specific kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 * @return filter created from specific kernel and weight.
	 */
	public static ProductFilter create(NeuronValue[][] kernel, NeuronValue weight) {
		if (kernel == null || kernel.length == 0 || weight == null) return null;
		if (kernel[0] == null || kernel[0].length == 0) return null;
		
		return new ProductFilter(kernel, weight);
	}
	
	
}
