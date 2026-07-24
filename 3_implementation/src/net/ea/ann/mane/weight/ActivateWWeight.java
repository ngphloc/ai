/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.Weight;

/**
 * This class represents weight activation weight which is like null weight but having weight activation function.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ActivateWWeight implements Weight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public ActivateWWeight() {

	}

	
	@Override
	public Kernel kernel() {return new Kernel.NullKernel();}

	
	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {return this;}

	
	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {return input;}

	
	@Override
	public Matrix dValue(Matrix prevOutput, Matrix thisError) {
		if (MatrixUtil.depth(prevOutput) != MatrixUtil.depth(thisError)) throw new IllegalArgumentException();
		if (prevOutput.rows() != thisError.rows() || prevOutput.columns() != thisError.columns()) throw new IllegalArgumentException();
		return thisError;
	}

	
	@Override
	public Kernel dKernel(Matrix prevOutput, Matrix thisError) {
		if (MatrixUtil.depth(prevOutput) != MatrixUtil.depth(thisError)) throw new IllegalArgumentException();
		if (prevOutput.rows() != prevOutput.rows() || prevOutput.columns() != thisError.columns()) throw new IllegalArgumentException();
		return new Kernel.NullKernel();
	}

	
}
