/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.util.Random;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class represents product filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class KernelFilterProduct extends KernelFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected FKernel kernel = null;
	
	
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
	protected KernelFilterProduct(FKernel kernel, NeuronValue weight) {
		super();
		if (!checkValid(kernel)) throw new IllegalArgumentException();
		this.kernel = kernel;
		this.weight = weight;
		this.strideWidth = kernel.width();
		this.strideHeight = kernel.height();
	}

	
	/**
	 * Checking kernel.
	 * @param kernel specific kernel.
	 * @return true if kernel is valid.
	 */
	private static boolean checkValid(FKernel kernel) {
		return kernel != null;
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
	public int width() {return kernel.width();}


	@Override
	public int height() {return kernel.height();}


	@Override
	int depth() {return kernel.depth();}


	@Override
	int time() {return kernel.time();}


	@Override
	FKernel kernel() {return kernel;}
	
	
	/**
	 * Setting internal kernel.
	 * @param otherKernel internal kernel.
	 * @return true if setting is successful.
	 */
	boolean setKernel(FKernel otherKernel) {
		if (!checkValid(otherKernel)) throw new IllegalArgumentException();
		this.kernel = otherKernel;
		this.strideWidth = otherKernel.width();
		this.strideHeight = otherKernel.height();
		return true;
	}
	

	@Override
	public KernelFilterProduct accumKernel(Kernel dKernel, double factor) {
		this.kernel = this.kernel.add(dKernel.multiply(factor));
		return this;
	}
	
	
	/**
	 * Getting internal weight.
	 * @return internal weight.
	 */
	NeuronValue getWeight() {return weight;}

	
	/**
	 * Setting internal weight.
	 * @param weight weight.
	 */
	void setWeight(NeuronValue weight) {
		if (weight != null) this.weight = weight;
	}
	

	@Override
	NeuronValue apply(int time, int y, int x, MatrixStack layers) {
		int kernelWidth = width(), kernelHeight = height(), kernelDepth = depth();
		int width = layers.columns(), height = layers.rows();
		NeuronValue zero = layers.get().get(0, 0).zero();
		if (x + kernelWidth > width) {
			if (isPadZero())
				return x >= width ? null : zero;
			else
				x = width - kernelWidth;
		}
		if (y + kernelHeight > height) {
			if (isPadZero())
				return y >= height ? null : zero;
			else
				y = height - kernelHeight;
		}
		
		NeuronValue result = zero;
		MatrixStack[] kernel = this.kernel.W;
		for (int i = 0; i < kernelDepth; i++) {
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue value = layers.get(i).get(y+j, x+k);
					result = result.add(value.multiply(kernel[time].get(i).get(j, k)));
				}
			}
		}
		return result.multiply(this.weight);
	}
	

	@Override
	MatrixStack dValue(int time, int thisX, int thisY, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		int kernelWidth = width(), kernelHeight = height(), kernelDepth = depth();
		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
		int prevX = xBlock*strideWidth;
		if (prevX + kernelWidth > prevWidth) {
			if (isPadZero())
				return prevX >= prevWidth ? null : null;
			else {
				prevX = prevWidth - kernelWidth;
				thisX = prevX/strideWidth;
			}
		}
		int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
		int prevY = yBlock*strideHeight;
		if (prevY + kernelHeight > prevHeight) {
			if (isPadZero())
				return prevY >= prevHeight ? null : null;
			else {
				prevY = prevHeight - kernelHeight;
				thisY = prevY/strideHeight;
			}
		}

		Matrix[] dValues = new Matrix[kernelDepth];
		NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
		MatrixStack[] kernel = this.kernel.W;
		for (int i = 0; i < kernelDepth; i++) {
			dValues[i] = prevInputLayers.get().create(new Size(kernelWidth, kernelHeight));
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue kernelValue = kernel[time].get(i).get(j, k);
					NeuronValue prevError = kernelValue.multiply(thisError);
					if (thisActivateRef != null) {
						NeuronValue prevOutput = prevOutputLayer.get(thisY, thisX);
						prevError = prevError.multiply(thisActivateRef.derivative(prevOutput));
					}
					dValues[i].set(j, k, prevError.multiply(this.weight));
				}
			}
		}
		return new MatrixStack(dValues);
	}


	@Override
	MatrixStack dKernel(int time, int thisX, int thisY, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		int kernelWidth = width(), kernelHeight = height(), kernelDepth = depth();
		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
		int prevX = xBlock*strideWidth;
		if (prevX + kernelWidth > prevWidth) {
			if (isPadZero())
				return prevX >= prevWidth ? null : null;
			else {
				prevX = prevWidth - kernelWidth;
				thisX = prevX/strideWidth;
			}
		}
		int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
		int prevY = yBlock*strideHeight;
		if (prevY + kernelHeight > prevHeight) {
			if (isPadZero())
				return prevY >= prevHeight ? null : null;
			else {
				prevY = prevHeight - kernelHeight;
				thisY = prevY/strideHeight;
			}
		}

		Matrix[] dKernels = new Matrix[kernelDepth];
		NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
		MatrixStack[] kernel = this.kernel.W;
		for (int i = 0; i < kernelDepth; i++) {
			dKernels[i] = kernel[time].get().create(new Size(kernelWidth, kernelHeight));
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue prevInput = prevInputLayers.get(i).get(thisY+j, thisX+k);
					NeuronValue dKernel = prevInput.multiply(thisError);
					if (thisActivateRef != null) {
						NeuronValue prevOutput = prevOutputLayer.get(thisY, thisX);
						dKernel = dKernel.multiply(thisActivateRef.derivative(prevOutput));
					}
					dKernels[i].set(j, k, dKernel.multiply(this.weight));
				}
			}
		}
		return new MatrixStack(dKernels);
	}
	

	@Override
	public void initParams(double v) {
		MatrixStack[] kernel = this.kernel.W;
		for (MatrixStack ker : kernel) MatrixUtil.fill(ker, v);
		this.weight = this.weight.unit();
	}


	@Override
	public void initParams(Random rnd) {
		MatrixStack[] kernel = this.kernel.W;
		for (MatrixStack ker : kernel) MatrixUtil.fill(ker, rnd);
		this.weight= this.weight.unit();
	}

	
	@Override
	public int sizeOfParams() {
		int size = 0;
		MatrixStack[] kernel = this.kernel.W;
		for (MatrixStack ker : kernel) size += MatrixUtil.capacity(ker);
		return size;
	}


	/**
	 * Creating kernel with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return kernel created from kernel value.
	 */
	static FKernel createKernel(double kernelValue, Size size, NeuronValue hint) {
		if (size.width < 1 || size.height < 1 || hint == null) return null;
		int depth = 1, time = 1;
		if (size.depth < 1)
			time = depth = 1;
		else if (size.time < 1) {
			depth = size.depth;
			time = 1;
		}
		else {
			depth = size.depth;
			time = size.time;
		}
		MatrixStack[] W = new MatrixStack[time];
		NeuronValue value = hint.valueOf(kernelValue);
		for (int t = 0; t < time; t++) {
			Matrix matrix = MatrixUtil.create(new Size(size.width, size.height, depth, 1), hint); 
			W[t] = matrix instanceof MatrixStack ? (MatrixStack)matrix : new MatrixStack(matrix);
			MatrixUtil.fill(W[t], value);
		}
		return new FKernel(W);
	}
	
	
	/**
	 * Creating product filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return product filter created from kernel value.
	 */
	public static KernelFilterProduct create(double kernelValue, Size size, NeuronValue hint) {
		return new KernelFilterProduct(createKernel(kernelValue, size, hint), hint.unit());
	}
	
	
}


