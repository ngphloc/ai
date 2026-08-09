/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import java.awt.Dimension;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class implements macro filter derived from micro filter developed by Min Lin, Qiang Chen, Shuicheng Yan.
 * Macro filter aims to improve accuracy due to diversity and so it is less stable.
 * @author Min Lin, Qiang Chen, Shuicheng Yan, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public class MacroFilter extends KernelFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected FKernel kernel = null;
	
	
	/**
	 * Constructor with kernel and weight.
	 * @param kernel specific kernel.
	 */
	protected MacroFilter(FKernel kernel) {
		super();
		if (!checkValid(kernel)) throw new IllegalArgumentException();
		this.kernel = kernel;
		
		if (Kernel.OPTIMIZER) this.kernel.setOptimizer(this.kernel.createOptimizer());
	}

	
	/**
	 * Checking kernel.
	 * @param kernel specific kernel.
	 * @return true if kernel is valid.
	 */
	private static boolean checkValid(FKernel kernel) {return kernel != null;}

	
	@Override
	public int width() {return kernel.width();}


	@Override
	public int height() {return kernel.height();}


	@Override
	int depth() {return kernel.depth();}


	@Override
	int time() {return kernel.time();}


	@Override
	public FKernel kernel() {return kernel;}

	
	@Override
	public MacroFilter accumKernel(Kernel dKernel, double factor) {
		assert (factor > 0 && factor <= 1);
		if (dKernel == this.kernel) throw new IllegalArgumentException();
		if (dKernel.getOptimizer() == null) dKernel.setOptimizer(this.kernel.getOptimizer());
		if (dKernel.getOptimizer() == this.kernel.getOptimizer()) dKernel = dKernel.optimize();
		
		this.kernel = this.kernel.add(dKernel.multiply(factor));
		return this;
	}
	
	
	@Override
	public Filter accumKernel(Kernel dKernel, double factor, double decay) {
		assert (factor > 0 && factor <= 1);
		if (dKernel == this.kernel) throw new IllegalArgumentException();
		if (dKernel.getOptimizer() == null) dKernel.setOptimizer(this.kernel.getOptimizer());
		if (dKernel.getOptimizer() == this.kernel.getOptimizer()) dKernel = dKernel.optimize();
		
		this.kernel = this.kernel.L2(decay).add(dKernel.multiply(factor));
		return this;
	}

	
	@Override
	NeuronValue apply(int time, int y, int x, MatrixStack layers) {
		throw new RuntimeException("This method is discarded");
	}


	@Override
	void forward(int time, MatrixStack prevLayers, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		assert (thisOutputLayer.rows() == this.height() && thisOutputLayer.columns() == this.width());
		NeuronValue zero = thisInputLayer != null ? thisInputLayer.get(0, 0).zero() : (thisOutputLayer != null ? thisOutputLayer.get(0, 0).zero() : prevLayers.get().get(0, 0).zero());
		MatrixUtil.fill(thisInputLayer, zero);
		MatrixUtil.fill(thisOutputLayer, zero);

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevLayers.columns(), prevHeight = prevLayers.rows();
		int thisWidth = thisOutputLayer.columns(), thisHeight = thisOutputLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int prevY = thisY*strideHeight;
			if (prevY >= prevHeight) continue;
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int prevX = thisX*strideWidth;
				if (prevX >= prevWidth) continue;
				
				if (Kernel.speedMode(zero)) {
					//Filtering
					MatrixStack[] kernel = this.kernel.W;
					double filteredValue = 0;
					for (int i = 0; i < depth(); i++) {
						double value = summode ? prevLayers.get(i).getv(prevY, prevX) :
							prevLayers.get(time).getv(prevY, prevX); //Please pay attention to this code line.
						filteredValue += value*kernel[time].get(i).getv(prevY, prevX);
					}
					
					//Adding bias.
					NeuronValue thisBias = this.bias(time, thisY, thisX);
					if (thisBias != null)
						filteredValue += ((NeuronValue1)thisBias).get();
					if (bias != null) {
						if (thisBias == null || Kernel.GLOBAL_BIAS) filteredValue += ((NeuronValue1)bias).get();
					}
					
					if (thisInputLayer != null) thisInputLayer.setv(thisY, thisX, filteredValue);
					if (thisActivateRef != null) filteredValue = ((NeuronValue1)(new NeuronValue1(filteredValue).evaluate(thisActivateRef))).get();
					if (thisOutputLayer != null) thisOutputLayer.setv(thisY, thisX, filteredValue);
				}
				else {
					//Filtering
					MatrixStack[] kernel = this.kernel.W;
					NeuronValue filteredValue = zero;
					for (int i = 0; i < depth(); i++) {
						NeuronValue value = summode ? prevLayers.get(i).get(prevY, prevX) :
							prevLayers.get(time).get(prevY, prevX); //Please pay attention to this code line.
						filteredValue = filteredValue.add(value.multiply(kernel[time].get(i).get(prevY, prevX)));
					}
					
					//Adding bias.
					NeuronValue thisBias = this.bias(time, thisY, thisX);
					if (thisBias != null)
						filteredValue = filteredValue.add(thisBias);
					if (bias != null) {
						if (thisBias == null || Kernel.GLOBAL_BIAS) filteredValue = filteredValue.add(bias);
					}
					
					if (thisInputLayer != null) thisInputLayer.set(thisY, thisX, filteredValue);
					if (thisActivateRef != null) filteredValue = filteredValue.evaluate(thisActivateRef);
					if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValue);
				}
			}
		}
	}


	@Override
	MatrixStack dValue(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		throw new RuntimeException("This method is discarded");
	}


	@Override
	MatrixStack dValue(int time, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		assert (prevInputLayers.rows() == this.height() && prevInputLayers.columns() == this.width());
		assert (prevOutputLayer.rows() == this.height() && prevOutputLayer.columns() == this.width());
		assert (thisErrorLayer.rows() == this.height() && thisErrorLayer.columns() == this.width());
		NeuronValue zero = prevInputLayers.get().get(0, 0).zero();
		Matrix[] dPrevValues = new Matrix[this.depth()];
		for (int i = 0; i < dPrevValues.length; i++) {
			int rows = prevInputLayers.rows(), columns = prevInputLayers.columns();
			dPrevValues[i] = prevInputLayers.get().create(new Size(columns, rows));
			MatrixUtil.fill(dPrevValues[i], zero);
		}

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int thisWidth = thisErrorLayer.columns(), thisHeight = thisErrorLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int prevY = thisY*strideHeight;
			if (prevY >= prevHeight) continue;
			assert (prevY == thisY);
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int prevX = thisX*strideWidth;
				if (prevX >= prevWidth) continue;
				assert (prevX == thisX);
				
				//Calculating gradient.
				NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
				NeuronValue derivative = thisActivateRef != null ? prevOutputLayer.get(thisY, thisX).derivativeWiseBy(thisActivateRef) : null;
				if (derivative != null) thisError = derivative.multiplyWise(thisError);
				MatrixStack[] kernel = this.kernel.W;
				
				if (Kernel.speedMode(zero)) {
					double thisErrorV = ((NeuronValue1)thisError).get();
					for (int i = 0; i < depth(); i++) {
						double kernelValue = kernel[time].get(i).getv(thisY, thisX);
						double prevError = kernelValue*thisErrorV;
						dPrevValues[i].setv(prevY, prevX, prevError);
					}
				}
				else {
					for (int i = 0; i < depth(); i++) {
						NeuronValue kernelValue = kernel[time].get(i).get(thisY, thisX);
						NeuronValue prevError = kernelValue.multiply(thisError);
						dPrevValues[i].set(prevY, prevX, prevError);
					}
				}
			}
		}
		
		return new MatrixStack(dPrevValues);
	}

	
	@Override
	BiasWeight dKernel(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		throw new RuntimeException("This method is discarded");
	}


	@Override
	BiasWeight dKernel(int time, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		assert (prevInputLayers.rows() == this.height() && prevInputLayers.columns() == this.width());
		assert (prevOutputLayer.rows() == this.height() && prevOutputLayer.columns() == this.width());
		assert (thisErrorLayer.rows() == this.height() && thisErrorLayer.columns() == this.width());
		assert (this.kernel.Bias != null && this.kernel.bias == null);
		assert (this.kernel.Bias.length == time());
		assert (this.kernel.Bias[0].rows() == this.height() && this.kernel.Bias[0].columns() == this.width());
		
		MatrixStack[] kernel = this.kernel().W;
		NeuronValue zero = kernel[time].get().get(0, 0).zero();
		Matrix[] dKernels = new Matrix[this.depth()];
		for (int i = 0; i < dKernels.length; i++) {
			dKernels[i] = kernel[time].get().create(new Size(width(), height()));
			MatrixUtil.fill(dKernels[i], zero);
		}
		Matrix dBiases = this.kernel().Bias != null ? thisErrorLayer.create(new Size(thisErrorLayer.columns(), thisErrorLayer.rows())) : null;
		NeuronValue dbiases = this.kernel().bias != null ? zero : null;

		int strideWidth = this.getStrideWidth(), strideHeight = this.getStrideHeight();
		int prevWidth = prevInputLayers.columns(), prevHeight = prevInputLayers.rows();
		int thisWidth = thisErrorLayer.columns(), thisHeight = thisErrorLayer.rows();
		
		if (Kernel.speedMode(zero)) {
			double dbiasesV = 0;
			for (int thisY = 0; thisY < thisHeight; thisY++) {
				int prevY = thisY*strideHeight;
				if (prevY >= prevHeight) continue;
				assert (prevY == thisY);
				
				for (int thisX = 0; thisX < thisWidth; thisX++) {
					int prevX = thisX*strideWidth;
					if (prevX >= prevWidth) continue;
					assert (prevX == thisX);
					
					//Calculating weight gradient.
					NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
					NeuronValue derivative = thisActivateRef != null ? prevOutputLayer.get(thisY, thisX).derivativeWiseBy(thisActivateRef) : null;
					if (derivative != null) thisError = derivative.multiplyWise(thisError);
					
					double thisErrorV = ((NeuronValue1)thisError).get();
					for (int i = 0; i < depth(); i++) {
						double prevInput = summode ? prevInputLayers.get(i).getv(prevY, prevX) :
							prevInputLayers.get(time).getv(prevY, prevX); //Please pay attention to this code line.
						double dK = prevInput*thisErrorV;
						dKernels[i].setv(thisY, thisX, dK);
					}
					
					//Calculating bias gradient.
					if (dBiases != null) dBiases.setv(thisY, thisX, thisErrorV);
					if (dbiases != null) dbiasesV += thisErrorV;
				}
			}
			if (dbiases != null) dbiases = dbiases.valueOf(dbiasesV);
		}
		else {
			for (int thisY = 0; thisY < thisHeight; thisY++) {
				int prevY = thisY*strideHeight;
				if (prevY >= prevHeight) continue;
				assert (prevY == thisY);
				
				for (int thisX = 0; thisX < thisWidth; thisX++) {
					int prevX = thisX*strideWidth;
					if (prevX >= prevWidth) continue;
					assert (prevX == thisX);
					
					//Calculating weight gradient.
					NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
					NeuronValue derivative = thisActivateRef != null ? prevOutputLayer.get(thisY, thisX).derivativeWiseBy(thisActivateRef) : null;
					if (derivative != null) thisError = derivative.multiplyWise(thisError);
					
					for (int i = 0; i < depth(); i++) {
						NeuronValue prevInput = summode ? prevInputLayers.get(i).get(prevY, prevX) :
							prevInputLayers.get(time).get(prevY, prevX); //Please pay attention to this code line.
						NeuronValue dK = prevInput.multiply(thisError);
						dKernels[i].set(thisY, thisX, dK);
					}
					
					//Calculating bias gradient.
					if (dBiases != null) dBiases.set(thisY, thisX, thisError);
					if (dbiases != null) dbiases = dbiases.add(thisError);
				}
			}
		}
		
		return new BiasWeight(new MatrixStack(dKernels), dBiases, dbiases);
	}

	
	/**
	 * Creating kernel with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @param bilinear bilinear mode.
	 * @return kernel created from kernel value.
	 */
	static FKernel createKernel(double kernelValue, Size size, NeuronValue hint, boolean bilinear) {
		if (size.width < 1 || size.height < 1 || hint == null) return null;
		size = KernelFilterProduct.adjustSize(size);
		
		int depth = size.depth;
		if (bilinear) if (size.depth == size.time) depth = 1; //Please pay attention to this code line.
		
		MatrixStack[] W = new MatrixStack[size.time];
		Matrix[] bias = new Matrix[size.time];
		NeuronValue value = hint.valueOf(kernelValue);
		NeuronValue zero = hint.zero();
		for (int t = 0; t < size.time; t++) {
			Matrix matrix = MatrixUtil.create(new Size(size.width, size.height, depth, 1), hint); 
			W[t] = matrix instanceof MatrixStack ? (MatrixStack)matrix : new MatrixStack(matrix);
			MatrixUtil.fill(W[t], value);
			
			bias[t] = MatrixUtil.create(new Size(size.width, size.height, 1, 1), hint); 
			MatrixUtil.fill(bias[t], zero);
		}
		return new FKernel(W, bias, null);
	}

	
	/**
	 * Creating macro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @param bilinear bilinear mode.
	 * @return product filter created from kernel value.
	 */
	public static MacroFilter create(double kernelValue, Size size, NeuronValue hint, boolean bilinear) {
		MacroFilter filter = new MacroFilter(createKernel(kernelValue, size, hint, bilinear));
		size = KernelFilterProduct.adjustSize(size);
		filter.summode = size.depth != size.time || !bilinear;
		return filter;
	}
	
	
	/**
	 * Creating macro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return product filter created from kernel value.
	 */
	static MacroFilter create(double kernelValue, Size size, NeuronValue hint) {
		return create(kernelValue, size, hint, Kernel.BILINEAR);
	}
	
	
	/**
	 * Creating macro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param depth depth of kernel.
	 * @param hint hint value.
	 * @param bilinear bilinear mode.
	 * @return micro filter created from kernel value.
	 */
	static MacroFilter create(double kernelValue, Dimension size, int depth, NeuronValue hint, boolean bilinear) {
		return create(kernelValue, new Size(size.width, size.height, depth, 1), hint, bilinear);
	}

	
	/**
	 * Creating macro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @param bilinear bilinear mode.
	 * @return micro filter created from kernel value.
	 */
	static MacroFilter create(double kernelValue, Dimension size, NeuronValue hint, boolean bilinear) {
		return create(kernelValue, new Size(size.width, size.height, 1, 1), hint, bilinear);
	}


}
