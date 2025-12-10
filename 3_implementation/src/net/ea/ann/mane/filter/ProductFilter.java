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
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents product filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProductFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected MatrixStack[] kernel = null;
	
	
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
	protected ProductFilter(MatrixStack[] kernel, NeuronValue weight) {
		super();
		if (!checkValid(kernel)) throw new IllegalArgumentException();
		this.kernel = kernel;
		this.weight = weight;
		this.strideWidth = kernel[0].columns();
		this.strideHeight = kernel[0].rows();
	}

	
	/**
	 * Checking kernel.
	 * @param kernel specific kernel.
	 * @return true if kernel is valid.
	 */
	private static boolean checkValid(MatrixStack[] kernel) {
		if (kernel == null || kernel.length == 0) return false;
		int columns = kernel[0].columns();
		int rows = kernel[0].rows();
		int depth = kernel[0].depth();
		if (columns <= 0 || rows <= 0 || depth <= 0) return false;
		for (int t = 0; t < kernel.length; t++) {
			if (kernel[t].columns() != columns || kernel[t].rows() != rows || kernel[t].depth() != depth) return false;
		}
		return true;
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
		return kernel[0].columns();
	}


	@Override
	public int height() {
		return kernel[0].rows();
	}


	/**
	 * Getting filter depth.
	 * @return filter depth.
	 */
	public int depth() {
		return kernel[0].depth();
	}


	/**
	 * Getting filter time.
	 * @return filter time.
	 */
	public int time() {
		return kernel.length;
	}

	
	/**
	 * Getting internal kernel.
	 * @return internal kernel.
	 */
	public Matrix[] getKernel() {return kernel;}
	

	/**
	 * Setting internal kernel.
	 * @param otherKernel internal kernel.
	 * @return true if setting is successful.
	 */
	private boolean setKernel(MatrixStack[] otherKernel) {
		if (!checkValid(otherKernel)) throw new IllegalArgumentException();
		this.kernel = otherKernel;
		this.strideWidth = otherKernel[0].columns();
		this.strideHeight = otherKernel[0].rows();
		return true;
	}
	
	
	/**
	 * Setting internal kernel.
	 * @param otherKernel internal kernel.
	 * @return true if setting is successful.
	 */
	public boolean setKernel(Matrix[] otherKernel) {
		if (otherKernel instanceof MatrixStack[]) return setKernel((MatrixStack[])otherKernel);
		MatrixStack[] stacks = new MatrixStack[otherKernel.length];
		for (int t = 0; t < stacks.length; t++) {
			stacks[t] = otherKernel[t] instanceof MatrixStack ? (MatrixStack)otherKernel[t] : new MatrixStack(otherKernel[t]);
		}
		return setKernel(stacks);
	}
	

	/**
	 * Accumulating kernel.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 */
	public void accumKernel(Matrix[] dKernel, double factor) {
		MatrixStack.accum(this.kernel, dKernel, factor);
	}
	
	
	/**
	 * Getting internal weight.
	 * @return internal weight.
	 */
	public NeuronValue getWeight() {return weight;}

	
	/**
	 * Setting internal weight.
	 * @param weight weight.
	 */
	public void setWeight(NeuronValue weight) {
		if (weight != null) this.weight = weight;
	}
	
	
	/**
	 * Applying this filter to specific layers. Please attention to this important method.
	 * @param time time.
	 * @param x x coordinator.
	 * @param y y coordinator.
	 * @param layers specific layers.
	 * @return the value resulted from this application.
	 */
	private NeuronValue apply(int time, int x, int y, MatrixStack layers) {
		int kernelWidth = width();
		int kernelHeight = height();
		int kernelDepth = depth();
		if (kernelDepth != layers.depth()) throw new IllegalArgumentException();
		
		NeuronValue zero = layers.get().get(0, 0).zero();
		int width = layers.columns();
		int height = layers.rows();
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
		for (int i = 0; i < kernelDepth; i++) {
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue value = layers.get(i).get(y+j, x+k);
					result = result.add(value.multiply(this.kernel[time].get(i).get(j, k)));
				}
			}
		}
		return result.multiply(this.weight);
	}
	
	
	/**
	 * Forwarding evaluation from previous layers to current layers.
	 * @param time time.
	 * @param prevLayers previous layers.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 * @param bias bias.
	 * @param thisActivateRef current activation function.
	 */
	private void forward(int time, MatrixStack prevLayers, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		if (depth() != prevLayers.depth()) throw new IllegalArgumentException();
		NeuronValue zero = thisInputLayer != null ? thisInputLayer.get(0, 0).zero() : (thisOutputLayer != null ? thisOutputLayer.get(0, 0).zero() : prevLayers.get().get(0, 0).zero());
		Matrix.fill(thisInputLayer, zero);
		Matrix.fill(thisOutputLayer, zero);

		int strideWidth = this.getStrideWidth();
		int strideHeight = this.getStrideHeight();
		int prevWidth = prevLayers.columns();
		int prevHeight = prevLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisInputLayer.columns();
		int thisHeight = thisInputLayer.rows();
		
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Filtering
				NeuronValue filteredValue = this.apply(time, prevX, prevY, prevLayers);
				if (filteredValue == null) continue;
				if (bias != null) filteredValue = filteredValue.add(bias);
				if (thisInputLayer != null) thisInputLayer.set(prevY, prevX, filteredValue);
				if (thisActivateRef != null) filteredValue = thisActivateRef.evaluate(filteredValue);
				if (thisOutputLayer != null) thisOutputLayer.set(prevY, prevX, filteredValue);
			}
		}
	}
	
	
	/**
	 * Forwarding evaluation from previous layers to this layers.
	 * @param time time.
	 * @param prevLayers current layers.
	 * @param thisInputLayers next layers 1
	 * @param thisOutputLayers next layers 2.
	 * @param bias bias.
	 * @param thisActivateRef current activation function.
	 */
	private void forward(MatrixStack prevLayers, MatrixStack thisInputLayers, MatrixStack thisOutputLayers, NeuronValue bias, Function thisActivateRef) {
		if (thisInputLayers.depth() != time() || thisOutputLayers.depth() != time()) throw new IllegalArgumentException();
		for (int t = 0; t < time(); t++) {
			forward(t, prevLayers, thisInputLayers.get(t), thisOutputLayers.get(t), bias, thisActivateRef);
		}
	}
	
	
	/**
	 * Forwarding evaluation from previous layers to current layers.
	 * @param time time.
	 * @param prevLayer current layer.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 * @param bias bias.
	 * @param thisActivateRef activation function.
	 */
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		MatrixStack prevLayers = prevLayer instanceof MatrixStack ? (MatrixStack)prevLayer : new MatrixStack(prevLayer);
		MatrixStack thisInputLayers = thisInputLayer instanceof MatrixStack ? (MatrixStack)thisInputLayer : new MatrixStack(thisInputLayer);
		MatrixStack thisOutputLayers = thisOutputLayer instanceof MatrixStack ? (MatrixStack)thisOutputLayer : new MatrixStack(thisOutputLayer);
		forward(prevLayers, thisInputLayers, thisOutputLayers, bias, thisActivateRef);
	}

	
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layer at specified coordinator.
	 * @param time time.
	 * @param thisX current X coordinator.
	 * @param thisY current Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layers.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	private MatrixStack dKernel(int time, int thisX, int thisY, MatrixStack prevInputLayers, MatrixStack prevOutputLayers, Matrix thisErrorLayer, Function thisActivateRef) {
		int kernelWidth = width();
		int kernelHeight = height();
		int kernelDepth = depth();
		if (kernelDepth != prevInputLayers.depth() || kernelDepth != prevOutputLayers.depth()) throw new IllegalArgumentException();
		
		int strideWidth = this.getStrideWidth();
		int strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns();
		int prevHeight = prevInputLayers.rows();
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
		for (int i = 0; i < kernelDepth; i++) {
			dKernels[i] = this.kernel[time].get().create(new Size(kernelWidth, kernelHeight));
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue prevInput = prevInputLayers.get(i).get(thisY+j, thisX+k);
					NeuronValue thisKernelError = prevInput.multiply(thisError);
					if (thisActivateRef != null) {
						NeuronValue prevOutput = prevOutputLayers.get(i).get(thisY+j, thisX+k);
						thisKernelError = thisKernelError.multiply(thisActivateRef.derivative(prevOutput));
					}
					dKernels[i].set(j, k, thisKernelError.multiply(this.weight));
				}
			}
		}
		return new MatrixStack(dKernels);
	}
	

	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layer.
	 * @param time time.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	private MatrixStack dKernel(int time, MatrixStack prevInputLayers, MatrixStack prevOutputLayers, Matrix thisErrorLayer, Function thisActivateRef) {
		if (depth() != prevInputLayers.depth() || depth() != prevOutputLayers.depth()) throw new IllegalArgumentException();
		NeuronValue zero = this.kernel[time].get().get(0, 0).zero();
		Matrix[] dPrevKernelArray = new Matrix[this.depth()];
		for (int i = 0; i < dPrevKernelArray.length; i++) {
			dPrevKernelArray[i] = this.kernel[time].get().create(new Size(width(), height()));
			Matrix.fill(dPrevKernelArray[i], zero);
		}
		MatrixStack dPrevKernels = new MatrixStack(dPrevKernelArray);
		int dPrevKernelsCount = 0;

		int strideWidth = this.getStrideWidth();
		int strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns();
		int prevHeight = prevInputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayer.columns();
		int thisHeight = thisErrorLayer.rows();
		
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				MatrixStack dKernels = this.dKernel(time, thisX, thisY, prevInputLayers, prevOutputLayers, thisErrorLayer, thisActivateRef);
				if (dKernels == null) continue;
				dPrevKernels = (MatrixStack)dPrevKernels.add(dKernels);
				dPrevKernelsCount++;
			}
		}
		if (dPrevKernelsCount <= 0) return dPrevKernels;
		
		//Calculating mean of kernel.
		if (CALC_ERROR_MEAN) dPrevKernels = (MatrixStack)dPrevKernels.divide0(dPrevKernelsCount);
		return dPrevKernels;
	}
	
	
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @param thisActivateRef activation function of current layers.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	private MatrixStack[] dKernel(MatrixStack prevInputLayers, MatrixStack prevOutputLayers, MatrixStack thisErrorLayers, Function thisActivateRef) {
		if (thisErrorLayers.depth() != time()) throw new IllegalArgumentException();
		MatrixStack[] dKernels = new MatrixStack[time()];
		for (int t = 0; t < time(); t++) {
			dKernels[t] = dKernel(t, prevInputLayers, prevOutputLayers, thisErrorLayers.get(t), thisActivateRef);
		}
		return dKernels;
	}
	
	
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	public Matrix[] dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		return dKernel(prevInputLayers, prevOutputLayers, thisErrorLayers, thisActivateRef);
	}
	
	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers at specified coordinator.
	 * @param time time.
	 * @param thisX current X coordinator.
	 * @param thisY current Y coordinator.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(int time, int thisX, int thisY, MatrixStack prevOutputLayers, Matrix thisErrorLayer, Function thisActivateRef) {
		int kernelWidth = width();
		int kernelHeight = height();
		int kernelDepth = depth();
		if (kernelDepth != prevOutputLayers.depth()) throw new IllegalArgumentException();
		
		int strideWidth = this.getStrideWidth();
		int strideHeight = this.getStrideHeight();
		int prevWidth = prevOutputLayers.columns();
		int prevHeight = prevOutputLayers.rows();
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
		for (int i = 0; i < kernelDepth; i++) {
			dValues[i] = prevOutputLayers.get(0).create(new Size(kernelWidth, kernelHeight));
			for (int j = 0; j < kernelHeight; j++) {
				for (int k = 0; k < kernelWidth; k++) {
					NeuronValue thisKernel = this.kernel[time].get(i).get(j, k);
					NeuronValue thisValueError = thisKernel.multiply(thisError);
					if (thisActivateRef != null) {
						NeuronValue prevOutput = prevOutputLayers.get(i).get(thisY+j, thisX+k);
						thisValueError = thisValueError.multiply(thisActivateRef.derivative(prevOutput));
					}
					dValues[i].set(j, k, thisValueError.multiply(this.weight));
				}
			}
		}
		return new MatrixStack(dValues);
	}


	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(int time, MatrixStack prevOutputLayers, Matrix thisErrorLayer, Function thisActivateRef) {
		if (depth() != prevOutputLayers.depth()) throw new IllegalArgumentException();
		NeuronValue zero = prevOutputLayers.get().get(0, 0).zero();
		Matrix[] dPrevValues = new Matrix[this.depth()];
		int[][][] dPrevValuesCount = new int[this.depth()][][];
		for (int i = 0; i < dPrevValues.length; i++) {
			int rows = prevOutputLayers.rows(), columns = prevOutputLayers.columns();
			dPrevValues[i] = prevOutputLayers.get().create(new Size(columns, rows));
			Matrix.fill(dPrevValues[i], zero);
			dPrevValuesCount[i] = new int[rows][columns];
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < columns; k++) dPrevValuesCount[i][j][k] = 0;
			}
		}

		int strideWidth = this.getStrideWidth();
		int strideHeight = this.getStrideHeight();
		int prevWidth = prevOutputLayers.columns();
		int prevHeight = prevOutputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayer.columns();
		int thisHeight = thisErrorLayer.rows();
		
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				MatrixStack dValues = this.dValue(time, thisX, thisY, prevOutputLayers, thisErrorLayer, thisActivateRef);
				if (dValues == null) continue;
				
				for (int i = 0; i < dValues.depth(); i++) {
					for (int j = 0; j < dValues.get(i).rows(); j++) {
						int row = prevY + j;
						for (int k = 0; k < dValues.get(i).columns(); k++) {
							int column = prevX + k;
							NeuronValue dv = dPrevValues[i].get(row, column).add(dValues.get(i).get(j, k));
							dPrevValues[i].set(row, column, dv);
							dPrevValuesCount[i][row][column] = dPrevValuesCount[i][row][column] + 1; 
						}
					}
				} //End dValues.
			}
		}
		
		//Calculating mean of values.
		if (CALC_ERROR_MEAN) {
			for (int i = 0; i < dPrevValues.length; i++) {
				int rows = dPrevValues[i].rows(), columns = dPrevValues[i].columns();
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						int count = dPrevValuesCount[i][row][column];
						if (count <= 0) continue;
						NeuronValue mean = dPrevValues[i].get(row, column).divide(count);
						dPrevValues[i].set(row, column, mean);
					}
				}
			}
		}
		return new MatrixStack(dPrevValues);
	}


	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @param thisActivateRef activation function of current layers.
	 * @return derivative of this previous given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevOutputLayers, MatrixStack thisErrorLayers, Function thisActivateRef) {
		if (thisErrorLayers.depth() != time()) throw new IllegalArgumentException();
		MatrixStack dValueSum = null;
		for (int t = 0; t < time(); t++) {
			MatrixStack dValue = dValue(t, prevOutputLayers, thisErrorLayers.get(t), thisActivateRef);
			dValueSum = dValueSum != null ? (MatrixStack)dValueSum.add(dValue) : dValue;
		}
		return dValueSum;
	}
	
	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	public Matrix dValue(Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack prevOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		MatrixStack stack = dValue(prevOutputLayers, thisErrorLayers, thisActivateRef);
		return stack.depth() == 1 ? stack.get() : stack;
	}
	
	
	@Override
	public void initialize(double v) {
		for (MatrixStack ker : this.kernel) MatrixStack.fill(ker, v);
		this.weight = this.weight.unit();
	}


	@Override
	public void initialize(Random rnd) {
		for (MatrixStack ker : this.kernel) MatrixStack.fill(ker, rnd);
		this.weight= this.weight.unit();
	}

	
	@Override
	public int sizeOfParams() {
		int size = 0;
		for (MatrixStack ker : this.kernel) size += Matrix.capacity(ker);
		return size;
	}


	/**
	 * Creating product filter with specific kernel and weight.
	 * @param kernel specific kernel.
	 * @param weight specific weight.
	 * @return product filter created from specific kernel and weight.
	 */
	public static ProductFilter create(MatrixStack[] kernel, NeuronValue weight) {
		return checkValid(kernel) ? new ProductFilter(kernel, weight) : null; 
	}
	
	
	/**
	 * Creating product filter with real kernel.
	 * @param kernelValue real kernel value.
	 * @param size size of kernel.
	 * @param hint matrix hint.
	 * @return product filter created from real kernel and weight.
	 */
	public static ProductFilter create(double kernelValue, Size size, Matrix hint) {
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
		MatrixStack[] kernel = new MatrixStack[time];
		NeuronValue value = hint.get(0, 0).valueOf(kernelValue);
		for (int t = 0; t < time; t++) {
			Matrix matrix = hint.create(new Size(size.width, size.height, depth, 1)); 
			kernel[t] = matrix instanceof MatrixStack ? (MatrixStack)matrix : new MatrixStack(matrix);
			Matrix.fill(kernel[t], value);
			if (kernel[t] == null) return null;
		}
		return create(kernel, value.unit());
	}
	
	
}
