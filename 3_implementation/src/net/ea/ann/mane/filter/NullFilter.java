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
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Filter;
import net.ea.ann.mane.Kernel;

/**
 * This class represents null kernel.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NullFilter extends FilterAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public NullFilter() {}

	
	@Override
	public boolean doesApplyActivate() {return false;}


	@Override
	public Filter accumKernel(Kernel dKernel, double factor) {return this;}

	
	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		if (MatrixUtil.depth(prevLayer) != MatrixUtil.depth(thisInputLayer) || MatrixUtil.depth(prevLayer) != MatrixUtil.depth(thisOutputLayer)) throw new IllegalArgumentException();
		if (prevLayer.rows() != thisInputLayer.rows() || prevLayer.rows() != thisOutputLayer.rows() || prevLayer.columns() != thisInputLayer.columns() || prevLayer.columns() != thisOutputLayer.columns()) throw new IllegalArgumentException();
		
		MatrixUtil.copy(prevLayer, thisInputLayer);
		MatrixUtil.copy(prevLayer, thisOutputLayer);
	}

	
	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		if (MatrixUtil.depth(prevInputLayer) != MatrixUtil.depth(prevOutputLayer) || MatrixUtil.depth(prevInputLayer) != MatrixUtil.depth(thisErrorLayer)) throw new IllegalArgumentException();
		if (prevInputLayer.rows() != prevOutputLayer.rows() || prevInputLayer.rows() != thisErrorLayer.rows() || prevInputLayer.columns() != prevOutputLayer.columns() || prevInputLayer.columns() != thisErrorLayer.columns()) throw new IllegalArgumentException();

		return thisErrorLayer;
	}

	
	@Override
	public Kernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		return new Kernel.NullKernel();
	}

	
}
