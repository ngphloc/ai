/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.vector.NeuronValueVector;
import net.ea.ann.core.value.vector.NeuronValueVectorImpl;
import net.ea.ann.raster.Size;

/**
 * This class represents max-pooling kernel filter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class KernelFilterMax extends KernelFilterProduct {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 */
	protected KernelFilterMax(FKernel kernel, NeuronValue weight) {
		super(kernel, weight);
	}


	/**
	 * Creating pair of index and value.
	 * @param index index.
	 * @param value value.
	 * @return pair of index and value.
	 */
	private static NeuronValueVector indexValue(int index, NeuronValue value) {
		return new NeuronValueVectorImpl(new NeuronValue1(index), value);
	}
	
	
	/**
	 * Getting index of the pair of index and value.
	 * @param indexValue pair of index and value.
	 * @return index.
	 */
	private static int index(NeuronValueVector indexValue) {
		return (int)((NeuronValue1)indexValue.get(0)).get();
	}
	
	
	/**
	 * Getting value of the pair of index and value.
	 * @param indexValue pair of index and value.
	 * @return value.
	 */
	private static NeuronValue value(NeuronValueVector indexValue) {
		return indexValue.get(1);
	}

	
	@Override
	NeuronValue apply(int time, int y, int x, MatrixStack layers) {
		int kernelWidth = width(), kernelHeight = height(), kernelDepth = depth();
		int width = layers.columns(), height = layers.rows();
		NeuronValue zero = layers.get().get(0, 0).zero();
		if (x + kernelWidth > width) {
			if (isPadZero())
				return x >= width ? null : null/*zero*/;
			else
				x = width - kernelWidth;
		}
		if (y + kernelHeight > height) {
			if (isPadZero())
				return y >= height ? null : null/*zero*/;
			else
				y = height - kernelHeight;
		}
		
		NeuronValue[] result = new NeuronValue[kernelDepth];
		MatrixStack[] kernel = this.kernel.W;
		for (int i = 0; i < kernelDepth; i++) {
			result[i] = zero;
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue value = layers.get(i).get(y+j, x+k);
					result[i] = result[i].add(value.multiply(kernel[time].get(i).get(j, k)));
				}
			}
			result[i] = result[i].multiply(this.weight);
		}
		return new NeuronValueVectorImpl(result);
	}


	@Override
	void forward(int time, MatrixStack prevLayers, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		NeuronValue zero = thisInputLayer != null ? thisInputLayer.get(0, 0).zero() : (thisOutputLayer != null ? thisOutputLayer.get(0, 0).zero() : prevLayers.get().get(0, 0).zero());
		MatrixUtil.fill(thisInputLayer, zero);
		MatrixUtil.fill(thisOutputLayer, zero);

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevLayers.columns(), prevHeight = prevLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisOutputLayer.columns(), thisHeight = thisOutputLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Filtering
				NeuronValueVector filteredValue = (NeuronValueVector)this.apply(time, prevY, prevX, prevLayers);
				if (filteredValue == null) continue;
				int maxIndex = 0;
				for (int index = 1; index < filteredValue.dim(); index++) {
					if (filteredValue.get(index).mean() > filteredValue.get(maxIndex).mean()) maxIndex = index;
				}
				NeuronValue filteredValueMax = filteredValue.get(maxIndex);
				if (thisInputLayer != null) thisInputLayer.set(thisY, thisX, indexValue(maxIndex, filteredValueMax));
				if (bias != null) filteredValueMax = filteredValueMax.add(bias);
				if (thisActivateRef != null) filteredValueMax = filteredValueMax.evaluate(thisActivateRef);
				if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValueMax);
			}
		}
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
		NeuronValue zero = thisError.zero();
		for (int i = 0; i < kernelDepth; i++) {
			dValues[i] = prevOutputLayer/*prevInputLayers.get()*/.create(new Size(kernelWidth, kernelHeight));
			int maxIndex = index((NeuronValueVector)prevInputLayers.get(i).get(thisY, thisX));
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					if (maxIndex != i) {
						dValues[i].set(j, k, zero);
						continue;
					}
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
		NeuronValue zero = thisError.zero();
		for (int i = 0; i < kernelDepth; i++) {
			dKernels[i] = kernel[time].get().create(new Size(kernelWidth, kernelHeight));
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValueVector prevInputV = (NeuronValueVector)prevInputLayers.get(i).get(thisY+j, thisX+k);
					int maxIndex = index(prevInputV);
					if (maxIndex != i) {
						dKernels[i].set(j, k, zero);
						continue;
					}
					NeuronValue prevInput = value(prevInputV);
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


	/**
	 * Creating max-pooling kernel filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return product filter created from kernel value.
	 */
	public static KernelFilterMax create(double kernelValue, Size size, NeuronValue hint) {
		return new KernelFilterMax(createKernel(kernelValue, size, hint), hint.unit());
	}


}
