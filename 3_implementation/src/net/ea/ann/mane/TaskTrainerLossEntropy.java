/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.function.Softmax;
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
	 * Calculating by column.
	 */
	public final static boolean BYCOLUMN = true;
	
	
	/**
	 * Flag to calculating by column.
	 */
	protected boolean byColumn = BYCOLUMN;
	
	
	/**
	 * Default constructor.
	 */
	public TaskTrainerLossEntropy() {
		super();
	}

	
	@Override
	protected Matrix gradient(Matrix output, Matrix realOutput, Object...params) {
		return byColumn ? LikelihoodGradient.lossEntropyGradientByColumn(output, realOutput, params) :
			LikelihoodGradient.lossEntropyGradientByRow(output, realOutput, params);
	}


	@Override
	public Matrix convert(Matrix output) {
		return byColumn ? Softmax.softmaxByColumn(output) : Softmax.softmaxByRow(output);
	}

	
	/**
	 * Checking by-column flag.
	 * @return by-column flag.
	 */
	public boolean isByColumn() {return byColumn;}
	
	
	/**
	 * Setting by-column flag.
	 * @param byColumn by-column flag.
	 * @return this task trainer.
	 */
	public TaskTrainerLossEntropy setByColumn(boolean byColumn) {
		this.byColumn = byColumn;
		return this;
	}
	
	
}
