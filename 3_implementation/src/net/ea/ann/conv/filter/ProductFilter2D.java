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
import net.ea.ann.core.function.Function;
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
	 * Adding accumulatively kernel.  
	 * @param kernel specified kernel.
	 */
	public void accumKernel(NeuronValue[][] kernel) {
		for (int i = 0; i < kernel.length; i++) {
			for (int j = 0; j < kernel[i].length; j++) {
				this.kernel[i][j] = this.kernel[i][j].add(kernel[i][j]);
			}
		}
	}

	
	/**
	 * Calculating sum of array of kernels.
	 * @param kernels array of kernels.
	 * @return sum of array of kernels.
	 */
	public static NeuronValue[][] kernelAdd(NeuronValue[][]...kernels) {
		if (kernels == null || kernels.length == 0) return null;
		if (kernels.length == 1) return kernels[0];
		NeuronValue[][] sum = kernels[0];
		for (int i = 1; i < kernels.length; i++) sum = NeuronValue.add(sum, kernels[i]);
		return sum;
	}

	
	/**
	 * Calculating mean of array of kernels.
	 * @param kernels array of kernels.
	 * @return mean of array of kernels.
	 */
	public static NeuronValue[][] kernelMean(NeuronValue[][]...kernels) {
		if (kernels == null || kernels.length == 0) return null;
		if (kernels.length == 1) return kernels[0];
		NeuronValue[][] mean = kernelAdd(kernels);
		return NeuronValue.divide(mean, (double)kernels.length);
	}
	
	
	/**
	 * Getting internal weight.
	 * @return internal weight.
	 */
	public NeuronValue getWeight() {
		return weight;
	}

	
	/**
	 * Setting internal weight.
	 * @param weight weight.
	 */
	public void setWeight(NeuronValue weight) {
		if (weight != null) this.weight = weight;
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
	public NeuronValue[][] dKernel(int nextX, int nextY, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer) {
		int kernelWidth = width();
		int kernelHeight = height();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		if (nextX + kernelWidth > thisWidth) {
			if (thisLayer.isPadZeroFilter()) {
				if (nextX >= thisWidth)
					return null;
				else
					return null;
			}
			else
				nextX = thisWidth - kernelWidth;
		}
		nextX = nextX < 0 ? 0 : nextX;
		if (nextY + kernelHeight > thisHeight) {
			if (thisLayer.isPadZeroFilter()) {
				if (nextY >= thisHeight)
					return null;
				else
					return null;
			}
			else
				nextY = thisHeight - kernelHeight;
		}
		nextY = nextY < 0 ? 0 : nextY;

		Function activateRef = nextLayer.getActivateRef();
		activateRef = activateRef == null ? thisLayer.getActivateRef() : activateRef;
		NeuronValue[][] dKernel = new NeuronValue[kernelHeight][kernelWidth];
		for (int i = 0; i < kernelHeight; i++) {
			for (int j = 0; j < kernelWidth; j++) {
				NeuronValue value = thisLayer.get(nextX+j, nextY+i).getValue().multiply(nextLayer.get(nextX, nextY).getValue());
				if (activateRef != null) {
					NeuronValue input = thisLayer.get(nextX+j, nextY+i).getInput();
					if (input != null) value = value.multiply(activateRef.derivative(input));
				}
				dKernel[i][j] = value.multiply(weight);
			}
		}
		return dKernel;
	}
	

	@Override
	public NeuronValue[][] dValue(int nextX, int nextY, ConvLayerSingle2D thisLayer, ConvLayerSingle2D nextLayer) {
		if (nextX < 0 || nextX >= nextLayer.getWidth()) return null;
		if (nextY < 0 || nextY >= nextLayer.getHeight()) return null;
		
		int kernelWidth = width();
		int kernelHeight = height();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		if (nextX + kernelWidth > thisWidth) {
			if (thisLayer.isPadZeroFilter()) {
				if (nextX >= thisWidth)
					return null;
				else
					return null;
			}
			else
				nextX = thisWidth - kernelWidth;
		}
		nextX = nextX < 0 ? 0 : nextX;
		if (nextY + kernelHeight > thisHeight) {
			if (thisLayer.isPadZeroFilter()) {
				if (nextY >= thisHeight)
					return null;
				else
					return null;
			}
			else
				nextY = thisHeight - kernelHeight;
		}
		nextY = nextY < 0 ? 0 : nextY;

		Function activateRef = nextLayer.getActivateRef();
		activateRef = activateRef == null ? thisLayer.getActivateRef() : activateRef;
		NeuronValue[][] dValue = new NeuronValue[kernelHeight][kernelWidth];
		for (int i = 0; i < kernelHeight; i++) {
			for (int j = 0; j < kernelWidth; j++) {
				NeuronValue value = kernel[i][j].multiply(nextLayer.get(nextX, nextY).getValue()); //Ignoring partial derivative due to ReLU by default has always derivative 1.
				if (activateRef != null) {
					NeuronValue input = thisLayer.get(nextX+j, nextY+i).getInput();
					if (input != null) value = value.multiply(activateRef.derivative(input));
				}
				dValue[i][j] = value.multiply(weight);
			}
		}
		return dValue;
	}

	
	/**
	 * Shallow cloning specified filter.
	 * @return cloned filter.
	 */
	public ProductFilter2D shallowClone() {
		NeuronValue[][] newKernel = new NeuronValue[this.height()][this.width()];
		for (int i = 0; i < kernel.length; i++) {
			for (int j = 0; j < kernel[i].length; j++)
				newKernel[i][j] = this.kernel[i][j];
		}
		ProductFilter2D newFilter = new ProductFilter2D(newKernel, this.weight);
		newFilter.strideWidth = this.strideWidth;
		newFilter.strideHeight = this.strideHeight;
		return newFilter;
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
	 * @param v specified value.
	 * @return product filter.
	 */
	public static ProductFilter2D create(Size size, NeuronValueCreator creator, double v) {
		if (size.width < 1) size.width = 1;
		if (size.height < 1) size.height = 1;
		
		NeuronValue source = creator.newNeuronValue();
		NeuronValue value = v == 0 ? source.zero() : source.valueOf(v);
		NeuronValue[][] kernel = new NeuronValue[size.height][size.width];
		for (int i = 0; i < size.height; i++) {
			for (int j = 0; j < size.width; j++) kernel[i][j] = value;
		}
		
		NeuronValue weight = source.valueOf(1.0);
		return new ProductFilter2D(kernel, weight);
	}
	
	
	/**
	 * Creating product filter with size.
	 * @param size kernel size.
	 * @param creator to create neuron value.
	 * @return product filter.
	 */
	public static ProductFilter2D create(Size size, NeuronValueCreator creator) {
		return create(size, creator, 0);
	}


}
