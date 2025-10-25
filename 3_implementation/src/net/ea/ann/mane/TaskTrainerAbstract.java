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
import net.ea.ann.core.function.Function;
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
	public Matrix[] train(MatrixLayer layer, Iterable<Record> inouts, boolean propagate, double learningRate) {
		List<Matrix> biases = Util.newList(0);
		for (Record inout : inouts) {
			if (inout == null) continue;
			Matrix realOutput = inout.output();
			if (layer instanceof MatrixLayerExt) {
				((MatrixLayerExt)layer).enterInputs(inout);
			}
			else {
				Matrix input = inout.input();
				if (input != null) Matrix.copy(input, layer.getInput());
			}
			
			Matrix output = layer.evaluate();
			Matrix bias = gradient(output, realOutput);
			if (bias == null) continue;
			
			if (!(layer instanceof MatrixNetworkCore)) {
				biases.add(bias);
				continue;
			}
			
			MatrixNetworkCore mnc = (MatrixNetworkCore)layer;
			Matrix oinput = mnc.getOutputLayer().getInput();
			Function activateRef = mnc.getOutputLayer().getActivateRef();
			if (oinput != null && activateRef != null) {
				Matrix derivative = oinput.derivativeWise(activateRef);
				bias = derivative.multiplyWise(bias);
			}
			biases.add(bias);
		}
		Matrix[] biasArray = biases.toArray(new Matrix[] {});
		
		return layer.backward(biasArray, propagate?layer:null, true, learningRate);
	}

	
	/**
	 * Calculating the optimal derivative given computed output and real output.
	 * @param output computed or predicted output.
	 * @param realOutput real output from environment. It can be null.
	 * @return optimal derivative.
	 */
	protected abstract Matrix gradient(Matrix output, Matrix realOutput);
	
	
}
