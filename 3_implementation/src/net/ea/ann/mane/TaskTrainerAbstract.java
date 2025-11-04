/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;

/**
 * This class implements partially specifies trainer applying into learning matrix neural network for specific task such as classification and reinforcement learning.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class TaskTrainerAbstract implements TaskTrainer, OutputConverter, Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public TaskTrainerAbstract() {
		super();
	}

	
	@Override
	public Error[] train(MatrixLayer layer, Iterable<Record> sample, boolean propagate, double learningRate, Object...params) {
		List<Error> biases = Util.newList(0);
		for (Record record : sample) {
			if (record == null) continue;
			Matrix realOutput = record.output();
			if (layer instanceof MatrixLayerExt) {
				((MatrixLayerExt)layer).enterInputs(record);
			}
			else {
				Matrix input = record.input();
				if (input != null) Matrix.copy(input, layer.getInput());
			}
			
			Error bias = new Error((Matrix)null);
			Matrix output = layer.evaluate(bias);
			Matrix err = gradient(output, realOutput, params);
			if (err != null) {
				bias.errorSet(err);
				biases.add(bias);
			}
		}
		Error[] biasArray = biases.toArray(new Error[] {});
		
		return layer.backward(biasArray, propagate?layer:null, true, learningRate);
	}

	
	/**
	 * Calculating the optimal derivative given computed output and real output.
	 * @param output computed or predicted output.
	 * @param realOutput real output from environment. It can be null.
	 * @param params additional parameters.
	 * @return optimal derivative.
	 */
	protected abstract Matrix gradient(Matrix output, Matrix realOutput, Object...params);
	
	
}
