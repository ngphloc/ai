/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter.deprecated;

import java.awt.Dimension;
import java.util.Random;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class implements micro filter developed by Min Lin, Qiang Chen, Shuicheng Yan.
 * @author Min Lin, Qiang Chen, Shuicheng Yan, implemented by Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class MicroFilter extends KernelFilter {


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
	protected MicroFilter(FKernel kernel) {
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
	public MicroFilter accumKernel(Kernel dKernel, double factor) {
		assert (factor > 0 && factor < 1);
		if (dKernel == this.kernel) throw new IllegalArgumentException();
		if (dKernel.getOptimizer() == null) dKernel.setOptimizer(this.kernel.getOptimizer());
		if (dKernel.getOptimizer() == this.kernel.getOptimizer()) dKernel = dKernel.optimize();
		
		this.kernel = this.kernel.add(dKernel.multiply(factor));
		return this;
	}
	
	
	@Override
	public Filter accumKernel(Kernel dKernel, double factor, double decay) {
		assert (factor > 0 && factor < 1);
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
				MatrixStack[] kernel = this.kernel.W;
				NeuronValue filteredValue = zero;
				for (int i = 0; i < depth(); i++) {
					NeuronValue value = summode ? prevLayers.get(i).get(prevY, prevX) :
						prevLayers.get(time).get(prevY, prevX); //Please pay attention to this code line.
					filteredValue = filteredValue.add(value.multiply(kernel[time].get(i).get(prevY, prevX)));
				}
				if (bias != null) filteredValue = filteredValue.add(bias);
				if (thisInputLayer != null) thisInputLayer.set(thisY, thisX, filteredValue);
				if (thisActivateRef != null) filteredValue = filteredValue.evaluate(thisActivateRef);
				if (thisOutputLayer != null) thisOutputLayer.set(thisY, thisX, filteredValue);
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
		int prevBlockWidth = this.isMoveStride() ? prevWidth / strideWidth : prevWidth;
		int prevBlockHeight = this.isMoveStride() ? prevHeight / strideHeight : prevHeight;
		int thisWidth = thisErrorLayer.columns(), thisHeight = thisErrorLayer.rows();
		for (int thisY = 0; thisY < thisHeight; thisY++) {
			int yBlock = this.isPadZero() ? thisY : (thisY < prevBlockHeight ? thisY : prevBlockHeight-1);
			int prevY = yBlock*strideHeight;
			if (prevY >= prevHeight) continue;
			assert (prevY == thisY);
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				assert (prevX == thisX);
				
				//Calculating gradient.
				NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
				NeuronValue derivative = thisActivateRef != null ? prevOutputLayer.get(thisY, thisX).derivativeWiseBy(thisActivateRef) : null;
				MatrixStack[] kernel = this.kernel.W;
				for (int i = 0; i < depth(); i++) {
					NeuronValue kernelValue = kernel[time].get(i).get(thisY, thisX);
					NeuronValue prevError = kernelValue.multiply(thisError);
					if (derivative != null) prevError = derivative.multiplyWise(prevError);
					dPrevValues[i].set(prevY, prevX, prevError);
				}
			}
		}
		
		return new MatrixStack(dPrevValues);
	}

	
	@Override
	MatrixStack dKernel(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		throw new RuntimeException("This method is discarded");
	}


	@Override
	MatrixStack dKernel(int time, MatrixStack prevInputLayers, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		assert (prevInputLayers.rows() == this.height() && prevInputLayers.columns() == this.width());
		assert (prevOutputLayer.rows() == this.height() && prevOutputLayer.columns() == this.width());
		assert (thisErrorLayer.rows() == this.height() && thisErrorLayer.columns() == this.width());

		MatrixStack[] kernel = this.kernel().W;
		NeuronValue zero = kernel[time].get().get(0, 0).zero();
		Matrix[] dKernels = new Matrix[this.depth()];
		for (int i = 0; i < dKernels.length; i++) {
			dKernels[i] = kernel[time].get().create(new Size(width(), height()));
			MatrixUtil.fill(dKernels[i], zero);
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
			assert (prevY == thisY);
			
			for (int thisX = 0; thisX < thisWidth; thisX++) {
				int xBlock = this.isPadZero() ? thisX : (thisX < prevBlockWidth ? thisX : prevBlockWidth-1);
				int prevX = xBlock*strideWidth;
				if (prevX >= prevWidth) continue;
				assert (prevX == thisX);
				
				//Calculating gradient.
				NeuronValue thisError = thisErrorLayer.get(thisY, thisX);
				NeuronValue derivative = thisActivateRef != null ? prevOutputLayer.get(thisY, thisX).derivativeWiseBy(thisActivateRef) : null;
				for (int i = 0; i < depth(); i++) {
					NeuronValue prevInput = summode ? prevInputLayers.get(i).get(prevY, prevX) :
						prevInputLayers.get(time).get(prevY, prevX); //Please pay attention to this code line.
					NeuronValue dKernel = prevInput.multiply(thisError);
					if (derivative != null) dKernel = derivative.multiplyWise(dKernel);
					dKernels[i].set(thisY, thisX, dKernel);
				}
			}
		}
		
		return new MatrixStack(dKernels);
	}

	
	@Override
	public void initParams(double v) {
		MatrixStack[] kernel = this.kernel.W;
		for (MatrixStack ker : kernel) MatrixUtil.fill(ker, v);
	}


	@Override
	public void initParams(Random rnd) {
		MatrixStack[] kernel = this.kernel.W;
		int fanIn = kernel[0].width()*kernel[0].height();
		for (MatrixStack ker : kernel) MatrixUtil.fill(ker, rnd, fanIn);
	}


	@Override
	public int sizeOfParams() {
		int size = 0;
		MatrixStack[] kernel = this.kernel.W;
		for (MatrixStack ker : kernel) size += MatrixUtil.capacity(ker);
		return size;
	}


	/**
	 * Creating micro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return product filter created from kernel value.
	 */
	public static MicroFilter create(double kernelValue, Size size, NeuronValue hint) {
		MicroFilter filter = new MicroFilter(KernelFilterProduct.createKernel(kernelValue, size, hint));
		size = KernelFilterProduct.adjustSize(size);
		filter.summode = size.depth != size.time || !Kernel.BILINEAR;
		return filter;
	}
	
	
	/**
	 * Creating micro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param depth depth of kernel.
	 * @param hint hint value.
	 * @return micro filter created from kernel value.
	 */
	public static MicroFilter create(double kernelValue, Dimension size, int depth, NeuronValue hint) {
		return create(kernelValue, new Size(size.width, size.height, depth, 1), hint);
	}

	
	/**
	 * Creating micro filter with kernel value.
	 * @param kernelValue kernel value.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return micro filter created from kernel value.
	 */
	public static MicroFilter create(double kernelValue, Dimension size, NeuronValue hint) {
		return create(kernelValue, new Size(size.width, size.height, 1, 1), hint);
	}


}
