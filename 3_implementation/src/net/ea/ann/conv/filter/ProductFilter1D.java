/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle1D;
import net.ea.ann.core.TextParsable;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Size;

/**
 * This class represents a product filter in 1D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProductFilter1D extends AbstractFilter1D implements TextParsable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected NeuronValue[] kernel = null;
	
	
	/**
	 * Kernel weight.
	 */
	protected NeuronValue weight = null;
	
	
	/**
	 * Stride width.
	 */
	private int strideWidth = 0;

	
	/**
	 * Constructor with kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 */
	protected ProductFilter1D(NeuronValue[] kernel, NeuronValue weight) {
		super();
		this.kernel = kernel;
		this.weight = weight;
		
		this.strideWidth = kernel.length;
	}

	
	@Override
	public int getStrideWidth() {
		if (!isMoveStride())
			return 1;
		else if (strideWidth <= 0)
			return width();
		else
			return strideWidth;
	}


	/**
	 * Setting stride width.
	 * @param strideWidth specified stride width.
	 * @return true if setting is successful.
	 */
	public boolean setStrideWidth(int strideWidth) {
		if (strideWidth <= 0)
			return false;
		else {
			this.strideWidth = strideWidth;
			return true;
		}
	}

	
	@Override
	public int width() {
		return kernel.length;
	}


	/**
	 * Getting internal kernel.
	 * @return internal kernel.
	 */
	public NeuronValue[] getKernel() {
		return kernel;
	}

	
	/**
	 * Getting internal weight.
	 * @return internal weight.
	 */
	public NeuronValue getWeight() {
		return weight;
	}

	
	@Override
	public NeuronValue apply(int x, ConvLayerSingle1D layer) {
		if (layer == null) return null;
		
		int kernelWidth = width();
		int width = layer.getWidth();
		if (x + kernelWidth > width) {
			if (layer.isPadZeroFilter()) {
				if (x >= width)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				x = width - kernelWidth;
		}
		x = x < 0 ? 0 : x;
		
		NeuronValue result = layer.newNeuronValue().zero();
		for (int j = 0; j < kernelWidth; j++) {
			NeuronValue value = layer.get(x+j).getValue();
			result = result.add(value.multiply(kernel[j]));
		}
		
		return result.multiply(weight);
	}

	
	@Override
	public String toText() {
		if (kernel == null || weight == null) return "";
		StringBuffer buffer = new StringBuffer();
		
		buffer.append("kernel = {");
		for (int j = 0; j < kernel.length; j++) {
			if (j > 0) buffer.append(", ");
			
			if (kernel[j] instanceof TextParsable)
				buffer.append(((TextParsable)kernel[j]).toText());
			else
				buffer.append(kernel[j]);
			
		}
		buffer.append("}");
		
		buffer.append(", weight = " + (weight instanceof TextParsable ? ((TextParsable)weight).toText() : weight.toString()));
		buffer.append(", move stride = " + isMoveStride());
		buffer.append(", stride width = " + getStrideWidth());
		
		return buffer.toString();
	}

	
	/**
	 * Creating product filter with specific kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 * @return product filter created from specific kernel and weight.
	 */
	public static ProductFilter1D create(NeuronValue[] kernel, NeuronValue weight) {
		if (kernel == null || weight == null) return null;
		
		return new ProductFilter1D(kernel, weight);
	}
	
	
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @param creator to create neuron value.
	 * @return product filter created from real kernel and weight.
	 */
	public static ProductFilter1D create(double[] kernel, double weight, NeuronValueCreator creator) {
		if (kernel == null) return null;
		
		int width = kernel.length;
		NeuronValue[] newKernel = new NeuronValue[width];
		NeuronValue source = creator.newNeuronValue();
		for (int j = 0; j < width; j++) newKernel[j] = source.valueOf(kernel[j]);
		
		NeuronValue newWeight = source.valueOf(weight);
		return new ProductFilter1D(newKernel, newWeight);
	}
	
	
	/**
	 * Creating product filter with size.
	 * @param size kernel size.
	 * @param creator to create neuron value.
	 * @return product filter.
	 */
	public static ProductFilter1D create(Size size, NeuronValueCreator creator) {
		if (size.width < 1) size.width = 1;
		
		NeuronValue source = creator.newNeuronValue();
		NeuronValue[] kernel = new NeuronValue[size.width];
		for (int j = 0; j < size.width; j++) kernel[j] = source.zero();
		
		NeuronValue weight = source.valueOf(1.0);
		return new ProductFilter1D(kernel, weight);
	}


}
