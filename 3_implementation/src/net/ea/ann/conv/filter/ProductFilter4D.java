/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import net.ea.ann.conv.ConvLayerSingle4D;
import net.ea.ann.core.TextParsable;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Size;

/**
 * This class represents a product filter in 4D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProductFilter4D extends AbstractFilter4D implements TextParsable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected NeuronValue[][][][] kernel = null;
	
	
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
	 * Stride depth.
	 */
	private int strideDepth = 0;

	
	/**
	 * Stride time.
	 */
	private int strideTime = 0;

	
	/**
	 * Constructor with kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 */
	protected ProductFilter4D(NeuronValue[][][][] kernel, NeuronValue weight) {
		super();
		this.kernel = kernel;
		this.weight = weight;
		
		this.strideWidth = kernel[0][0][0].length;
		this.strideHeight = kernel[0][0].length;
		this.strideDepth = kernel[0].length;
		this.strideTime = kernel.length;
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
	public int getStrideDepth() {
		if (!isMoveStride())
			return 1;
		else if (strideDepth <= 0)
			return depth();
		else
			return strideDepth;
	}


	/**
	 * Setting stride depth.
	 * @param strideDepth specified stride depth.
	 * @return true if setting is successful.
	 */
	public boolean setStrideDepth(int strideDepth) {
		if (strideDepth <= 0)
			return false;
		else {
			this.strideDepth = strideDepth;
			return true;
		}
	}

	
	@Override
	public int getStrideTime() {
		if (!isMoveStride())
			return 1;
		else if (strideTime <= 0)
			return time();
		else
			return strideTime;
	}

	
	/**
	 * Setting stride time.
	 * @param strideTime specified stride time.
	 * @return true if setting is successful.
	 */
	public boolean setStrideTime(int strideTime) {
		if (strideTime <= 0)
			return false;
		else {
			this.strideTime = strideTime;
			return true;
		}
	}

	
	@Override
	public int width() {
		return kernel[0][0][0].length;
	}


	@Override
	public int height() {
		return kernel[0][0].length;
	}


	@Override
	public int depth() {
		return kernel[0].length;
	}


	@Override
	public int time() {
		return kernel.length;
	}

	
	/**
	 * Getting internal kernel.
	 * @return internal kernel.
	 */
	public NeuronValue[][][][] getKernel() {
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
	public NeuronValue apply(int x, int y, int z, int t, ConvLayerSingle4D layer) {
		if (layer == null) return null;
		
		int kernelWidth = width();
		int kernelHeight = height();
		int kernelDepth = depth();
		int kernelTime = time();
		int width = layer.getWidth();
		int height = layer.getHeight();
		int depth = layer.getDepth();
		int time = layer.getTime();
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
		

		if (z + kernelDepth > depth) {
			if (layer.isPadZeroFilter()) {
				if (z >= depth)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				z = depth - kernelDepth;
		}
		z = z < 0 ? 0 : z;
		
		if (t + kernelTime > time) {
			if (layer.isPadZeroFilter()) {
				if (t >= time)
					return null;
				else
					return layer.newNeuronValue().zero();
			}
			else
				t = time - kernelTime;
		}
		t = t < 0 ? 0 : t;
		
		NeuronValue result = layer.newNeuronValue().zero();
		for (int h = 0; h < kernelTime; h++) {
			for (int i = 0; i < kernelDepth; i++) {
				for (int j = 0; j < kernelHeight; j++) {
					for (int k = 0; k < kernelWidth; k++) {
						NeuronValue value = layer.get(x+k, y+j, z+i, t+h).getValue();
						result = result.add(value.multiply(kernel[h][i][j][k]));
					}
				}
			}
		}
		return result.multiply(weight);
	}
	
	
	@Override
	public String toText() {
		if (kernel == null || weight == null) return "";
		StringBuffer buffer = new StringBuffer();
		
		buffer.append("kernel = {");
		for (int h = 0; h < time(); h++) {
			if (h > 0) buffer.append(", ");
			buffer.append("{");

			for (int i = 0; i < depth(); i++) {
				if (i > 0) buffer.append(", ");
				buffer.append("{");
				
				for (int j = 0; j < height(); j++) {
					if (j > 0) buffer.append(", ");
					buffer.append("{");
					
					for (int k = 0; k < width(); k++) {
						if (k > 0) buffer.append(", ");
						buffer.append("(");
						
						if (kernel[h][i][j][k] instanceof TextParsable)
							buffer.append(((TextParsable)kernel[h][i][j][k]).toText());
						else
							buffer.append(kernel[h][i][j][k]);
						
						buffer.append(")");
					}
					buffer.append("}");
				}
				buffer.append("}");
			}
			buffer.append("}");
		}
		buffer.append("}");
		
		buffer.append(", weight = (" + (weight instanceof TextParsable ? ((TextParsable)weight).toText() : weight.toString()) + ")");
		buffer.append(", move stride = " + isMoveStride());
		buffer.append(", stride width = " + getStrideWidth());
		buffer.append(", stride height = " + getStrideHeight());
		buffer.append(", stride depth = " + getStrideDepth());
		buffer.append(", stride time = " + getStrideTime());
		
		return buffer.toString();
	}
	
	
	/**
	 * Creating product filter with specific kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 * @return product filter created from specific kernel and weight.
	 */
	public static ProductFilter4D create(NeuronValue[][][][] kernel, NeuronValue weight) {
		if (kernel == null || weight == null) return null;
		
		return new ProductFilter4D(kernel, weight);
	}
	
	
	/**
	 * Creating product filter with real kernel and weight.
	 * @param kernel real kernel.
	 * @param weight real weight.
	 * @param creator to create neuron value.
	 * @return product filter created from real kernel and weight.
	 */
	public static ProductFilter4D create(double[][][][] kernel, double weight, NeuronValueCreator creator) {
		if (kernel == null) return null;
		
		int time = kernel.length;
		int depth = kernel[0].length;
		int height = kernel[0][0].length;
		int width = kernel[0][0][0].length;
		NeuronValue[][][][] newKernel = new NeuronValue[time][depth][height][width];
		NeuronValue source = creator.newNeuronValue();
		for (int h = 0; h < time; h++) {
			for (int i = 0; i < depth; i++) {
				for (int j = 0; j < height; j++) {
					for (int k = 0; k < width; k++) newKernel[h][i][j][k] = source.valueOf(kernel[h][i][j][k]);
				}
			}
		}
		
		NeuronValue newWeight = source.valueOf(weight);
		return new ProductFilter4D(newKernel, newWeight);
	}
	
	
	/**
	 * Creating product filter with size.
	 * @param size kernel size.
	 * @param creator to create neuron value.
	 * @return product filter.
	 */
	public static ProductFilter4D create(Size size, NeuronValueCreator creator) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		if (size.depth < 1) size.depth = 1;
		if (size.time < 1) size.time = 1;
		
		NeuronValue source = creator.newNeuronValue();
		NeuronValue[][][][] kernel = new NeuronValue[size.time][size.depth][size.height][size.width];
		for (int h = 0; h < size.time; h++) {
			for (int i = 0; i < size.depth; i++) {
				for (int j = 0; j < size.height; j++) {
					for (int k = 0; k < size.width; k++) kernel[h][i][j][k] = source.zero();
				}
			}
		}
		
		NeuronValue weight = source.valueOf(1.0);
		return new ProductFilter4D(kernel, weight);
	}


}
