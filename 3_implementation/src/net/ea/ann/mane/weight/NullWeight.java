/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.Weight;

/**
 * This class represents null weight.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NullWeight implements Weight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public NullWeight() {

	}

	
	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {
		return this;
	}

	
	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		return input;
	}

	
	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		if (MatrixUtil.depth(prevInput) != MatrixUtil.depth(prevOutput) || MatrixUtil.depth(prevInput) != MatrixUtil.depth(thisError)) throw new IllegalArgumentException();
		if (prevInput.rows() != prevOutput.rows() || prevInput.rows() != thisError.rows() || prevInput.columns() != prevOutput.columns() || prevInput.columns() != thisError.columns()) throw new IllegalArgumentException();
		return thisError;
	}

	
	@Override
	public Kernel dKernel(Matrix prevOutput, Matrix thisError) {
		if (MatrixUtil.depth(prevOutput) != MatrixUtil.depth(thisError)) throw new IllegalArgumentException();
		if (prevOutput.rows() != prevOutput.rows() || prevOutput.columns() != thisError.columns()) throw new IllegalArgumentException();
		return new Kernel.NullKernel();
	}

	
}
