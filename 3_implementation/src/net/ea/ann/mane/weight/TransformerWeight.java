/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import net.ea.ann.core.Network;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.Kernel.NullKernel;
import net.ea.ann.mane.Weight;
import net.ea.ann.raster.Size;
import net.ea.ann.transformer.TransformerImpl;
import net.ea.ann.transformer.TransformerInitializer;

/**
 * This class represents parametric weight based on transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerWeight implements NetworkWeight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal kernel.
	 */
	protected TKernel kernel = null;
	
	
	/**
	 * Default constructor with neuron channel, previous size, and current size.
	 * @param neuronChannel neuron channel.
	 * @param prevSize previous size.
	 * @param thisSize current size.
	 */
	protected TransformerWeight(int neuronChannel, Size prevSize, Size thisSize) {
		Size size = new Size(thisSize.width, thisSize.height, prevSize.depth, thisSize.depth);
		this.kernel = new TKernel(neuronChannel, size);
	}

	
	/**
	 * Getting width.
	 * @return width.
	 */
	int width() {return kernel.width();}
	
	
	/**
	 * Getting height.
	 * @return height.
	 */
	int height() {return kernel.height();}

	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	int depth() {return kernel.depth();}

	
	/**
	 * Getting time.
	 * @return time.
	 */
	int time() {return kernel.time();}

	
	/**
	 * Getting transformer at specified time and depth.
	 * @param time time.
	 * @param depth depth.
	 * @return transformer at specified time and depth.
	 */
	TransformerImpl tra(int time, int depth) {return kernel.transformer(time, depth);}
	
	
	@Override
	public boolean backwardErrorMode() {return false;}


	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {
		this.kernel = (TKernel)this.kernel.add(dKernel);
		return this;
	}

	
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param inputs inputs.
	 * @param bias bias.
	 * @return evaluated value.
	 */
	private Matrix evaluate(int time, MatrixStack inputs, Matrix bias) {
		int depth = depth();
		Matrix sum = null;
		for (int d = 0; d < depth; d++) {
			Matrix value = tra(time, d).evaluate(inputs.get(d));
			sum = sum != null ? sum.add(value) : value;
		}
		return sum.add(bias);
	}
	
	
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param inputs inputs.
	 * @param biases biases.
	 * @return evaluated value.
	 */
	private MatrixStack evaluate(MatrixStack inputs, MatrixStack biases) {
		if (inputs.depth() != depth() || biases.depth() != time()) throw new IllegalArgumentException();
		int time = time();
		Matrix[] values = new Matrix[time];
		for (int t = 0; t < time; t++) {
			values[t] = evaluate(t, inputs, biases.get(t));
		}
		return new MatrixStack(values);

	}
	

	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		MatrixStack inputs = input instanceof MatrixStack ? (MatrixStack)input : new MatrixStack(input);
		MatrixStack biases = bias instanceof MatrixStack ? (MatrixStack)bias : new MatrixStack(bias);
		MatrixStack values = evaluate(inputs, biases);
		return values.depth() == 1 ? values.get() : values;
	}

	
//	/**
//	 * Calculate gradient of previous layers.
//	 * @param time time.
//	 * @param prevInputs previous inputs.
//	 * @param prevOutputs previous outputs.
//	 * @param thisError current error.
//	 * @param prevActivateRef previous activation function.
//	 * @param learning learning mode.
//	 * @param learningRate learning rate.
//	 * @return gradient of previous layers.
//	 */
//	private MatrixStack dValue(int time, MatrixStack prevInputs, MatrixStack prevOutputs, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate) {
//		int depth = depth();
//		Matrix[] dValues = new Matrix[depth];
//		Matrix derivative = prevOutputs.get(time) != null && prevActivateRef != null ? prevOutputs.get(time).derivativeWise(prevActivateRef) : null;
//		for (int d = 0; d < depth; d++) {
//			dValues[d] = tra(time, d).backward(new Error[] {new Error(thisError)}, null, learning, learningRate)[0].error();
//			if (derivative != null) dValues[d] = derivative.multiplyWise(dValues[d]);
//		}
//		return new MatrixStack(dValues);
//	}
	
	
	/**
	 * Calculate gradient of previous layers.
	 * @param time time.
	 * @param prevInputs previous inputs.
	 * @param prevOutput previous output.
	 * @param thisError current error.
	 * @param prevActivateRef previous activation function.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return gradient of previous layers.
	 */
	private Matrix dValue(int time, MatrixStack prevInputs, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate) {
		int depth = depth();
		Matrix sum = null;
		Matrix derivative = prevOutput != null && prevActivateRef != null ? prevOutput.derivativeWise(prevActivateRef) : null;
		for (int d = 0; d < depth; d++) {
			Matrix dValue = tra(time, d).backward(new Error[] {new Error(thisError)}, null, learning, learningRate)[0].error();
			if (derivative != null) dValue = derivative.multiplyWise(dValue);
			sum = sum != null ? sum.add(dValue) : dValue;
		}
		return sum;
	}

	
//	/**
//	 * Calculate gradient of previous layers.
//	 * @param prevInputs previous inputs.
//	 * @param prevOutputs previous outputs.
//	 * @param thisErrors current errors.
//	 * @param prevActivateRef previous activation function.
//	 * @param learning learning mode.
//	 * @param learningRate learning rate.
//	 * @return gradient of previous layers.
//	 */
//	private MatrixStack dValue(MatrixStack prevInputs, MatrixStack prevOutputs, MatrixStack thisErrors, Function prevActivateRef, boolean learning, double learningRate) {
//		if (prevInputs.depth() != depth() || prevOutputs.depth() != time() || thisErrors.depth() != time()) throw new IllegalArgumentException();
//		int time = time();
//		MatrixStack sum = null;
//		for (int t = 0; t < time; t++) {
//			MatrixStack dValue = dValue(t, prevInputs, prevOutputs, thisErrors.get(t), prevActivateRef, learning, learningRate);
//			sum = sum != null ? (MatrixStack)sum.add(dValue) : dValue;
//		}
//		return sum;
//	}


	/**
	 * Calculate gradient of previous layers.
	 * @param prevInputs previous inputs.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @param prevActivateRef previous activation function.
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @return gradient of previous layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputs, MatrixStack prevOutputs, MatrixStack thisErrors, Function prevActivateRef, boolean learning, double learningRate) {
		if (prevInputs.depth() != depth() || prevOutputs.depth() != time() || thisErrors.depth() != time()) throw new IllegalArgumentException();
		int time = time();
		Matrix[] dValues = new Matrix[time];
		for (int t = 0; t < time; t++) {
			dValues[t] = dValue(t, prevInputs, prevOutputs.get(t), thisErrors.get(t), prevActivateRef, learning, learningRate);
		}
		return new MatrixStack(dValues);
	}

	
	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate) {
		MatrixStack prevInputs = prevInput instanceof MatrixStack ? (MatrixStack)prevInput : new MatrixStack(prevInput);
		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dValue = dValue(prevInputs, prevOutputs, thisErrors, prevActivateRef, learning, learningRate);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}
	
	
	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		return dValue(prevInput, prevOutput, thisError, prevActivateRef, true, Network.LEARN_RATE_DEFAULT);
	}

	
	/**
	 * Updating parameters from backward information.
	 * @param recordCount count of records in sample.
	 * @param learningRate learning rate.
	 */
	public void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		for (int t = 0; t < time(); t++) {
			for (int d = 0; d < depth(); d++) {
				tra(t, d).updateParametersFromBackwardInfo(recordCount, learningRate);
			}
		}
	}
	
	
	/**
	 * Resetting backward information.
	 */
	public void resetBackwardInfo() {
		for (int t = 0; t < time(); t++) {
			for (int d = 0; d < depth(); d++) {
				tra(t, d).resetBackwardInfo();
			}
		}
	}

	
	/**
	 * Creating transformer-based weight with neuron channel, previous size, and current size.
	 * @param neuronChannel neuron channel.
	 * @param prevSize previous size.
	 * @param thisSize current size.
	 */
	public static TransformerWeight create(int neuronChannel, Size prevSize, Size thisSize) { 
		return new TransformerWeight(neuronChannel, prevSize, thisSize);
	}
	
	
}



/**
 * This class represents kernel of transformer-based weight.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class TKernel extends NullKernel {
	
	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal transformers.
	 */
	protected TransformerImpl[][] transformers = null;
	
	
	/**
	 * COnstructor with size.
	 * @param size size.
	 */
	TKernel(int neuronChannel, Size size) {
		int n = size.height, dm = size.width;
		if (n <= 0 || dm <= 0) throw new IllegalArgumentException();
		int depth = size.depth < 1 ? 1 : size.depth;
		int time = size.time < 1 ? 1 : size.time;
		this.transformers = new TransformerImpl[time][depth];
		for (int t = 0; t < time; t++) {
			for (int d = 0; d < depth; d++) {
				this.transformers[t][d] = new TransformerImpl(neuronChannel);
				new TransformerInitializer(this.transformers[t][d]).initializeOnlyEncoder(n, dm);
				this.transformers[t][d].removeOutputFFN();
			}
		}
	}
	

	/**
	 * Getting width.
	 * @return width.
	 */
	int width() {
		return transformers[0][0].encoder().getInput().columns();
	}
	
	
	/**
	 * Getting height.
	 * @return height.
	 */
	int height() {
		return transformers[0][0].encoder().getInput().rows();
	}

	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	int depth() {return transformers[0].length;}

	
	/**
	 * Getting time.
	 * @return time.
	 */
	int time() {return transformers.length;}

	
	/**
	 * Getting transformer at specified time and depth.
	 * @param time time.
	 * @param depth depth.
	 * @return transformer at specified time and depth.
	 */
	TransformerImpl transformer(int time, int depth) {return transformers[time][depth];}


}

