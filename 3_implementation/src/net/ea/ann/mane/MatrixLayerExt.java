/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.function.Function;

/**
 * This interface extensive represents layer in matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface MatrixLayerExt extends MatrixLayer {

	
	/**
	 * Entering inputs.
	 * @param record record.
	 */
	void enterInputs(Record record);
	
	
	/**
	 * Getting output layer.
	 * @return output layer.
	 */
	MatrixLayer getOutputLayer();


	/**
	 * Getting output activation function.
	 * @return output activation function.
	 */
	Function getOutputActivateRef();

	
}
