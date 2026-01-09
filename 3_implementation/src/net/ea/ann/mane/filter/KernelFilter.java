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
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class represents a kernel filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class KernelFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This class represents filter kernel.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class FKernel implements Kernel {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * The weight.
		 */
		protected MatrixStack[] W = null;
		
		/**
		 * Constructor with weight.
		 * @param W weight.
		 */
		public FKernel(MatrixStack[] W) {
			if (!checkValid(W)) throw new IllegalArgumentException();
			this.W = W;
		}

		/**
		 * Checking the weight.
		 * @param W the weight.
		 * @return true if the weight is valid.
		 */
		private static boolean checkValid(MatrixStack[] W) {
			return W != null && W.length > 0;
		}

		/**
		 * Getting width.
		 * @return kernel width.
		 */
		public int width() {
			return W[0].columns();
		}

		/**
		 * Getting kernel height.
		 * @return kernel height.
		 */
		public int height() {
			return W[0].rows();
		}


		/**
		 * Getting kernel depth.
		 * @return kernel depth.
		 */
		public int depth() {
			return W[0].depth();
		}


		/**
		 * Getting kernel time.
		 * @return kernel time.
		 */
		public int time() {
			return W.length;
		}

		@Override
		public FKernel add(Kernel kernel) {
			MatrixStack[] sum = this.W != null ? MatrixStack.sum2(this.W, ((FKernel)kernel).W) : null;
			return new FKernel(sum);
		}

		@Override
		public FKernel multiply(double value) {
			MatrixStack[] d = this.W != null ? MatrixStack.multiply(this.W, value) : null;
			return new FKernel(d);
		}

		@Override
		public FKernel divide(double value) {
			MatrixStack[] d = this.W != null ? MatrixStack.divide(this.W, value) : null;
			return new FKernel(d);
		}

		/**
		 * Calculating sum.
		 * @param kernels kernels.
		 * @return sum.
		 */
		@Deprecated
		private static FKernel sum(FKernel[] kernels) {
			FKernel sum = kernels[0];
			for (int i = 1; i < kernels.length; i++) sum = sum.add(kernels[i]);
			return sum;
		}
		
		/**
		 * Calculating mean.
		 * @param kernels kernels.
		 * @return mean.
		 */
		@SuppressWarnings("unused")
		@Deprecated
		private static FKernel mean(FKernel[] kernels) {
			FKernel sum = sum(kernels);
			return sum.divide(kernels.length);
		}
		
	}

	
	/**
	 * Default constructor.
	 */
	protected KernelFilter() {
		super();
	}

	
	/**
	 * Getting filter depth.
	 * @return filter depth.
	 */
	abstract int depth();


	/**
	 * Getting filter time.
	 * @return filter time.
	 */
	abstract int time();

	
	/**
	 * Getting kernel.
	 * @return kernel.
	 */
	abstract FKernel kernel();

	
	/**
	 * Applying this filter to specific layers. Please attention to this important method.
	 * @param time time.
	 * @param y y coordinator.
	 * @param x x coordinator.
	 * @param layers specific layers.
	 * @return the value resulted from this application.
	 */
	abstract NeuronValue apply(int time, int y, int x, MatrixStack layers);

		
	/**
	 * Forwarding evaluation from previous layer to current layer.
	 * @param time time.
	 * @param prevLayers previous layers.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 * @param bias bias.
	 * @param thisActivateRef current activation function.
	 */
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
				NeuronValue filteredValue = this.apply(time, prevY, prevX, prevLayers);
				if (filteredValue == null) continue;
				if (bias != null) filteredValue = filteredValue.add(bias);
				if (thisInputLayer != null) thisInputLayer.set(thisY, thisX, filteredValue);
				if (thisActivateRef != null) filteredValue = filteredValue.evaluate(thisActivateRef);
				if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValue);
			}
		}
	}
	
	
	/**
	 * Forwarding evaluation from previous layers to this layers.
	 * @param time time.
	 * @param prevLayers previous layers.
	 * @param thisInputLayers current input layers.
	 * @param thisOutputLayers current output layers.
	 * @param bias bias.
	 * @param thisActivateRef current activation function.
	 */
	private void forward(MatrixStack prevLayers, MatrixStack thisInputLayers, MatrixStack thisOutputLayers, NeuronValue bias, Function thisActivateRef) {
		if (prevLayers.depth() != depth() || thisInputLayers.depth() != time() || thisOutputLayers.depth() != time()) throw new IllegalArgumentException();
		if (thisInputLayers.rows() != thisOutputLayers.rows() || thisInputLayers.columns() != thisOutputLayers.columns()) throw new IllegalArgumentException();
		
		for (int t = 0; t < time(); t++) {
			forward(t, prevLayers, thisInputLayers.get(t), thisOutputLayers.get(t), bias, thisActivateRef);
		}
	}
	

	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		MatrixStack prevLayers = prevLayer instanceof MatrixStack ? (MatrixStack)prevLayer : new MatrixStack(prevLayer);
		MatrixStack thisInputLayers = thisInputLayer instanceof MatrixStack ? (MatrixStack)thisInputLayer : new MatrixStack(thisInputLayer);
		MatrixStack thisOutputLayers = thisOutputLayer instanceof MatrixStack ? (MatrixStack)thisOutputLayer : new MatrixStack(thisOutputLayer);
		forward(prevLayers, thisInputLayers, thisOutputLayers, bias, thisActivateRef);
	}

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers at specified coordinator.
	 * @param time time.
	 * @param thisX current X coordinator.
	 * @param thisY current Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	abstract MatrixStack dValue(int time, int thisX, int thisY, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef);

		
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(int time, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		NeuronValue zero = prevInputLayers.get().get(0, 0).zero();
		Matrix[] dPrevValues = new Matrix[this.depth()];
		int[][][] dPrevValuesCount = new int[this.depth()][][];
		for (int i = 0; i < dPrevValues.length; i++) {
			int rows = prevInputLayers.rows(), columns = prevInputLayers.columns();
			dPrevValues[i] = prevInputLayers.get().create(new Size(columns, rows));
			MatrixUtil.fill(dPrevValues[i], zero);
			dPrevValuesCount[i] = new int[rows][columns];
			for (int j = 0; j < rows; j++) {
				for (int k = 0; k < columns; k++) dPrevValuesCount[i][j][k] = 0;
			}
		}

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayer.columns(), thisHeight = thisErrorLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				MatrixStack dPrevValue = this.dValue(time, thisX, thisY, prevInputLayers, prevOutputLayer, thisErrorLayer, thisActivateRef);
				if (dPrevValue == null) continue;
				
				for (int i = 0; i < dPrevValue.depth(); i++) {
					for (int j = 0; j < dPrevValue.get(i).rows(); j++) {
						int prevRow = prevY + j;
						for (int k = 0; k < dPrevValue.get(i).columns(); k++) {
							int prevColumn = prevX + k;
							NeuronValue dv = dPrevValues[i].get(prevRow, prevColumn).add(dPrevValue.get(i).get(j, k));
							dPrevValues[i].set(prevRow, prevColumn, dv);
							dPrevValuesCount[i][prevRow][prevColumn] = dPrevValuesCount[i][prevRow][prevColumn] + 1; 
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
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @param thisActivateRef activation function of current layers.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevOutputLayers, MatrixStack thisErrorLayers, Function thisActivateRef) {
		if (prevInputLayers.depth() != depth() || prevOutputLayers.depth() != time() || thisErrorLayers.depth() != time()) throw new IllegalArgumentException();
		if (prevOutputLayers.rows() != thisErrorLayers.rows() || prevOutputLayers.columns() != thisErrorLayers.columns()) throw new IllegalArgumentException();
		
		MatrixStack dValueSum = null;
		for (int t = 0; t < time(); t++) {
			MatrixStack dValue = dValue(t, prevInputLayers, prevOutputLayers.get(t), thisErrorLayers.get(t), thisActivateRef);
			dValueSum = dValueSum != null ? (MatrixStack)dValueSum.add(dValue) : dValue;
		}
		return dValueSum;
	}
	

	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		MatrixStack stack = dValue(prevInputLayers, prevOutputLayers, thisErrorLayers, thisActivateRef);
		return stack.depth() == 1 ? stack.get() : stack;
	}
	
	
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layer at specified coordinator.
	 * @param time time.
	 * @param thisX current X coordinator.
	 * @param thisY current Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layers.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	abstract MatrixStack dKernel(int time, int thisX, int thisY, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef);

		
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layer.
	 * @param time time.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	private MatrixStack dKernel(int time, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack[] kernel = this.kernel().W;
		NeuronValue zero = kernel[time].get().get(0, 0).zero();
		Matrix[] dKernelArray = new Matrix[this.depth()];
		for (int i = 0; i < dKernelArray.length; i++) {
			dKernelArray[i] = kernel[time].get().create(new Size(width(), height()));
			MatrixUtil.fill(dKernelArray[i], zero);
		}
		MatrixStack dKernels = new MatrixStack(dKernelArray);
		int dKernelCount = 0;

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayer.columns(), thisHeight = thisErrorLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				MatrixStack dKernel = this.dKernel(time, thisX, thisY, prevInputLayers, prevOutputLayer, thisErrorLayer, thisActivateRef);
				if (dKernel == null) continue;
				dKernels = (MatrixStack)dKernels.add(dKernel);
				dKernelCount++;
			}
		}
		if (dKernelCount <= 0) return dKernels;
		
		//Calculating mean of kernel.
		if (CALC_ERROR_MEAN) dKernels = (MatrixStack)dKernels.divide0(dKernelCount);
		return dKernels;
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
		if (prevInputLayers.depth() != depth() || prevOutputLayers.depth() != time() || thisErrorLayers.depth() != time()) throw new IllegalArgumentException();
		if (prevOutputLayers.rows() != thisErrorLayers.rows() || prevOutputLayers.columns() != thisErrorLayers.columns()) throw new IllegalArgumentException();
		
		MatrixStack[] dKernels = new MatrixStack[time()];
		for (int t = 0; t < time(); t++) {
			dKernels[t] = dKernel(t, prevInputLayers, prevOutputLayers.get(t), thisErrorLayers.get(t), thisActivateRef);
		}
		return dKernels;
	}
	

	@Override
	public FKernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		return new FKernel(dKernel(prevInputLayers, prevOutputLayers, thisErrorLayers, thisActivateRef));
	}
	
	
}
