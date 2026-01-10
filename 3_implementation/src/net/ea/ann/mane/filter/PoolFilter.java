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
import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class represents a pooling filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class PoolFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Kernel width.
	 */
	protected int width = 1;
	
	
	/**
	 * Kernel height.
	 */
	protected int height = 1;

	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	protected PoolFilter(Size size) {
		super();
		if (size.width < 1 || size.height < 1) throw new IllegalArgumentException();
		this.width = size.width;
		this.height = size.height;
	}

	
	@Override
	public int width() {return width;}


	@Override
	public int height() {return height;}
	
	
	/**
	 * Getting filter depth.
	 * @return filter depth.
	 */
	abstract int depth();

	
	@Override
	public boolean doesApplyActivate() {return false;}

	
	@Override
	public Filter accumKernel(Kernel dKernel, double factor) {return this;}


	/**
	 * Forwarding evaluation from previous layer to current layer.
	 * @param prevLayer previous layer.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 */
	abstract void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer);

		
	/**
	 * Forwarding evaluation from previous layers to this layers.
	 * @param time time.
	 * @param prevLayers previous layers.
	 * @param thisInputLayers current input layers.
	 * @param thisOutputLayers current output layers.
	 */
	private void forward(MatrixStack prevLayers, MatrixStack thisInputLayers, MatrixStack thisOutputLayers) {
		if (prevLayers.depth() != depth() || thisInputLayers.depth() != depth() || thisOutputLayers.depth() != depth()) throw new IllegalArgumentException();
		if (thisInputLayers.rows() != thisOutputLayers.rows() || thisInputLayers.columns() != thisOutputLayers.columns()) throw new IllegalArgumentException();
		
		for (int d = 0; d < depth(); d++) {
			forward(prevLayers.get(d), thisInputLayers.get(d), thisOutputLayers.get(d));
		}
	}
	

	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		MatrixStack prevLayers = prevLayer instanceof MatrixStack ? (MatrixStack)prevLayer : new MatrixStack(prevLayer);
		MatrixStack thisIndexInputLayers = thisInputLayer instanceof MatrixStack ? (MatrixStack)thisInputLayer : new MatrixStack(thisInputLayer);
		MatrixStack thisOutputLayers = thisOutputLayer instanceof MatrixStack ? (MatrixStack)thisOutputLayer : new MatrixStack(thisOutputLayer);
		forward(prevLayers, thisIndexInputLayers, thisOutputLayers);
	}

	
	/**
	 * Calculating derivative of previous layer given current layer as bias layer at specified coordinator.
	 * @param time time.
	 * @param thisX current X coordinator.
	 * @param thisY current Y coordinator.
	 * @param prevInputLayer previous input layers.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layer given current layer as bias layers.
	 */
	abstract Matrix dValue(int thisX, int thisY, Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer);

		
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param nextX next X coordinator.
	 * @param nextY next Y coordinator.
	 * @param prevInputLayers previous input layers.
	 * @param prevOutputLayers previous output layers.
	 * @param thisErrorLayers current layers as bias layers.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputLayers, MatrixStack prevOutputLayers, MatrixStack thisErrorLayers) {
		if (prevInputLayers.depth() != depth() || prevOutputLayers.depth() != depth()) throw new IllegalArgumentException();
		if (prevOutputLayers.rows() != thisErrorLayers.rows() || prevOutputLayers.columns() != thisErrorLayers.columns()) throw new IllegalArgumentException();

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
		int thisWidth = thisErrorLayers.columns(), thisHeight = thisErrorLayers.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				
				//Calculating gradient.
				for (int i = 0; i < this.depth(); i++) {
					Matrix dPrevValue = null;
					for (int count = 0; count < thisErrorLayers.depth(); count++) {
						Matrix dValue = this.dValue(thisX, thisY, prevInputLayers.get(i), prevOutputLayers.get(i), thisErrorLayers.get(count));
						dPrevValue = dPrevValue != null ? dPrevValue.add(dValue) : dValue; 
					}
					if (dPrevValue == null) continue;
					for (int j = 0; j < dPrevValue.rows(); j++) {
						int prevRow = prevY + j;
						for (int k = 0; k < dPrevValue.columns(); k++) {
							int prevColumn = prevX + k;
							NeuronValue dv = dPrevValues[i].get(prevRow, prevColumn).add(dPrevValue.get(j, k));
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

	
	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		MatrixStack prevIndexOutputLayers = prevOutputLayer instanceof MatrixStack ? (MatrixStack)prevOutputLayer : new MatrixStack(prevOutputLayer);
		MatrixStack thisErrorLayers = thisErrorLayer instanceof MatrixStack ? (MatrixStack)thisErrorLayer : new MatrixStack(thisErrorLayer);
		MatrixStack stack = dValue(prevInputLayers, prevIndexOutputLayers, thisErrorLayers);
		return stack.depth() == 1 ? stack.get() : stack;
	}

	
	@Override
	public Kernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		return null;
	}


}
