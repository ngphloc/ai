/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.value.Matrix;

/**
 * This class implements soft-max task trainer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TaskTrainerLossEntropy extends TaskTrainerAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public TaskTrainerLossEntropy() {
		super();
	}

	
	@Override
	protected Matrix gradient(Matrix output, Matrix realOutput) {
		return LikelihoodGradient.lossEntropyGradientByColumn(output, realOutput);
	}

	
}
