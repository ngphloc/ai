/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;

/**
 * This class represents parametric weight with transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightTrans implements Weight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public WeightTrans() {

	}

	
	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public Kernel dKernel(Matrix prevOutput, Matrix thisError) {
		throw new RuntimeException("Not implemented yet");
	}

	
}
