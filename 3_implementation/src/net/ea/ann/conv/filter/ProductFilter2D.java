/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.core.TextParsable;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Size;

/**
 * This class represents a product filter in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProductFilter2D extends AbstractFilter2D implements TextParsable {


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
	 * Stride width.
	 */
	private int strideWidth = 0;
	
	
	/**
	 * Stride width.
	 */
	private int strideHeight = 0;
	
	
	/**
	 * Constructor with kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 */
	protected ProductFilter2D(NeuronValue[][] kernel, NeuronValue weight) {
		super();
		this.kernel = kernel;
		this.weight = weight;
		
		this.strideWidth = kernel[0].length;
		this.strideHeight = kernel.length;
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
	public int getStrideHeight() {
		if (!isMoveStride())
			return 1;
		else if (strideHeight <= 0)
			return height();
		else
			return strideHeight;
	}


	/**
	 * Setting stride height.
	 * @param strideHeight specified stride height.
	 * @return true if setting is successful.
	 */
	public boolean setStrideHeight(int strideHeight) {
		if (strideHeight <= 0)
			return false;
		else {
			this.strideHeight = strideHeight;
			return true;
		}
	}
	
	
	@Override
	public int width() {
		return kernel[0].length;
	}


	@Override
	public int height() {
		return kernel.length;
	}


	/**
	 * Getting internal kernel.
	 * @return internal kernel.
	 */
	public NeuronValue[][] getKernel() {
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
	public NeuronValue apply(int x, int y, ConvLayerSingle2D layer) {
		if (layer == null) return null;
		
		int kernelWidth = width();
		int kernelHeight = height();
		int width = layer.getWidth();
		int height = layer.getHeight();
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
		if (y + kernelHeight > height) {
			if (layer.isPadZeroFilter()) {
				if (y >= height)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				y = height - kernelHeight;
		}
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
	
	
	@Override
	public String toText() {
		if (kernel == null || weight == null) return "";
		StringBuffer buffer = new StringBuffer();
		
		buffer.append("kernel = {");
		for (int i = 0; i < kernel.length; i++) {
			if (i > 0) buffer.append(", ");
			buffer.append("{");
			for (int j = 0; j < kernel[i].length; j++) {
				if (j > 0) buffer.append(", ");
				buffer.append("(");
				
				if (kernel[i][j] instanceof TextParsable)
					buffer.append(((TextParsable)kernel[i][j]).toText());
				else
					buffer.append(kernel[i][j]);
				
				buffer.append(")");
			}
			buffer.append("}");
		}
		buffer.append("}");
		
		buffer.append(", weight = (" + (weight instanceof TextParsable ? ((TextParsable)weight).toText() : weight.toString()) + ")");
		buffer.append(", move stride = " + isMoveStride());
		buffer.append(", stride width = " + getStrideWidth());
		buffer.append(", stride height = " + getStrideHeight());
		
		return buffer.toString();
	}
	
	
	/**
	 * Creating product filter with specific kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 * @return product filter created from specific kernel and weight.
	 */
	public static ProductFilter2D create(NeuronValue[][] kernel, NeuronValue weight) {
		if (kernel == null || weight == null) return null;
		
		return new ProductFilter2D(kernel, weight);
	}
	
	
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @param creator to create neuron value.
	 * @return product filter created from real kernel and weight.
	 */
	public static ProductFilter2D create(double[][] kernel, double weight, NeuronValueCreator creator) {
		if (kernel == null) return null;
		
		int height = kernel.length;
		int width = kernel[0].length;
		NeuronValue[][] newKernel = new NeuronValue[height][width];
		NeuronValue source = creator.newNeuronValue();
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) newKernel[i][j] = source.valueOf(kernel[i][j]);
		}
		
		NeuronValue newWeight = source.valueOf(weight);
		return new ProductFilter2D(newKernel, newWeight);
	}
	
	
	/**
	 * Creating product filter with size.
	 * @param size kernel size.
	 * @param creator to create neuron value.
	 * @return product filter.
	 */
	public static ProductFilter2D create(Size size, NeuronValueCreator creator) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		
		NeuronValue source = creator.newNeuronValue();
		NeuronValue[][] kernel = new NeuronValue[size.height][size.width];
		for (int i = 0; i < size.height; i++) {
			for (int j = 0; j < size.width; j++) kernel[i][j] = source.zero();
		}
		
		NeuronValue weight = source.valueOf(1.0);
		return new ProductFilter2D(kernel, weight);
	}


}
