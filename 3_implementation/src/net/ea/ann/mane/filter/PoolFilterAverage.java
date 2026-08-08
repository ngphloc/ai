/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class represents average pooling filter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class PoolFilterAverage extends PoolFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Depth.
	 */
	protected int depth = 1;
	
	
	/**
	 * Constructor with size.
	 * @param size size.
	 */
	protected PoolFilterAverage(Size size) {
		super(size);
		this.depth = size.depth < 1 ? 1 : size.depth; 
	}

	
	/**
	 * Getting filter depth.
	 * @return filter depth.
	 */
	public int depth() {return depth;}


	/**
	 * Applying this filter to specific layer. Please attention to this important method.
	 * @param y y coordinator.
	 * @param x x coordinator.
	 * @param layer specific layer.
	 * @return the index value resulted from this application.
	 */
	private NeuronValue apply(int y, int x, Matrix layer) {
		NeuronValue zero = layer.get(0, 0).zero();
		int layerWidth = layer.columns(), layerHeight = layer.rows();
		
		if (Kernel.speedMode(zero)) {
			double result = 0;
			int N = 0;
			for (int i = 0; i < height(); i++) {
				int Y = y + i;
				if (Y >= layerHeight) continue;
				for (int j = 0; j < width(); j++) {
					int X = x + j;
					if (X >= layerWidth) continue;
					double value = layer.getv(Y, X);
					result += value;
					N++;
				}
			}
			return zero.valueOf(result/(double)N);
		}
		else {
			NeuronValue result = zero;
			int N = 0;
			for (int i = 0; i < height(); i++) {
				int Y = y + i;
				if (Y >= layerHeight) continue;
				for (int j = 0; j < width(); j++) {
					int X = x + j;
					if (X >= layerWidth) continue;
					NeuronValue value = layer.get(Y, X);
					result = result.add(value);
					N++;
				}
			}
			return result.divide(N);
		}
	}


	@Override
	void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer) {
		NeuronValue zero = thisOutputLayer != null ? thisOutputLayer.get(0, 0).zero() : prevLayer.get(0, 0).zero();
		MatrixUtil.fill(thisInputLayer, zero);
		MatrixUtil.fill(thisOutputLayer, zero);

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevLayer.columns(), prevHeight = prevLayer.rows();
		int thisWidth = thisOutputLayer.columns(), thisHeight = thisOutputLayer.rows();
		
		if (Kernel.speedMode(zero)) {
			for (int thisY = 0; thisY < thisHeight; thisY++) {
				int prevY = thisY*strideHeight;
				if (prevY >= prevHeight) continue;
				
				for (int thisX = 0; thisX < thisWidth; thisX++) {
					int prevX = thisX*strideWidth;
					if (prevX >= prevWidth) continue;
					
					//Filtering
					NeuronValue filteredValue = this.apply(prevY, prevX, prevLayer);
					if (filteredValue == null) continue;
					
					double filteredValueV = ((NeuronValue1)filteredValue).get();
					if (thisInputLayer != null) thisInputLayer.setv(thisY, thisX, filteredValueV);
					if (thisOutputLayer != null) thisOutputLayer.setv(thisY, thisX, filteredValueV);
				}
			}
		}
		else {
			for (int thisY = 0; thisY < thisHeight; thisY++) {
				int prevY = thisY*strideHeight;
				if (prevY >= prevHeight) continue;
				
				for (int thisX = 0; thisX < thisWidth; thisX++) {
					int prevX = thisX*strideWidth;
					if (prevX >= prevWidth) continue;
					
					//Filtering
					NeuronValue filteredValue = this.apply(prevY, prevX, prevLayer);
					if (filteredValue == null) continue;
					
					if (thisInputLayer != null) thisInputLayer.set(thisY, thisX, filteredValue);
					if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValue);
				}
			}
		}
	}


	@Override
	Matrix dValue(int thisY, int thisX, Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer) {
		int kernelWidth = width(), kernelHeight = height();
		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayer.columns(), prevHeight = prevInputLayer.rows();
		int prevY = thisY*strideHeight;
		int prevX = thisX*strideWidth;
		
		int m = Math.min(kernelHeight, prevHeight-prevY), n = Math.min(kernelWidth, prevWidth-prevX);
		if (m <= 0 || n <= 0) throw new IllegalArgumentException();
		NeuronValue thisError = thisErrorLayer.get(thisY, thisX).divide(m*n);
		Matrix dPrevValue = prevInputLayer.create(new Size(kernelWidth, kernelHeight));
		NeuronValue zero = thisError.zero();
		MatrixUtil.fill(dPrevValue, zero);
		
		if (Kernel.speedMode(zero)) {
			double thisErrorV = ((NeuronValue1)thisError).get();
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < n; k++) {
					dPrevValue.setv(j, k, thisErrorV);
				}
			}
		}
		else {
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < n; k++) {
					dPrevValue.set(j, k, thisError);
				}
			}
		}
		return dPrevValue;
	}

	
	@Override
	public Object clone() throws CloneNotSupportedException {
		return super.clone();
	}


	/**
	 * Creating average pooling filter with specific kernel size.
	 * @param size specific kernel size.
	 * @return average pooling filter created from specific kernel size.
	 */
	public static PoolFilterAverage create(Size size) {
		return new PoolFilterAverage(size);
	}


}
