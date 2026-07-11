/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;

/**
 * This class implements micro filter developed by Min Lin, Qiang Chen, Shuicheng Yan.
 * @author Min Lin, Qiang Chen, Shuicheng Yan, implemented by Loc Nguyen
 * @version 1.0
 *
 */
public abstract class MicroFilter extends KernelFilter {


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

	
//	@Override
//	NeuronValue apply(int time, int y, int x, MatrixStack layers) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	
//	@Override
//	MatrixStack dValue(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer,
//			Matrix thisErrorLayer, Function thisActivateRef) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	
//	@Override
//	MatrixStack dKernel(int time, int thisY, int thisX, MatrixStack prevInputLayers, Matrix prevOutputLayer,
//			Matrix thisErrorLayer, Function thisActivateRef) {
//		// TODO Auto-generated method stub
//		return null;
//	}

	
}
