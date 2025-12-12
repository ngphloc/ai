/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents parametric weight with matrix stack.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightImpl implements Weight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * This kernel consists of both the first weight and the second weight.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class Kernel implements net.ea.ann.mane.Weight.Kernel {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * The first weight.
		 */
		protected MatrixStack[] W1 = null;
		
		/**
		 * The second weight.
		 */
		protected MatrixStack[] W2 = null;
		
		/**
		 * Constructor with the first weight and the second weight.
		 * @param W1 the first weight.
		 * @param W2 the second weight.
		 */
		public Kernel(MatrixStack[] W1, MatrixStack[] W2) {
			if (!checkValid(W1, W1)) throw new IllegalArgumentException();
			this.W1 = W1;
			this.W2 = W2;
		}

		/**
		 * Checking the first weight and the second weight.
		 * @param W1 the first weight.
		 * @param W2 the second weight.
		 * @return true if the two weights are mutually valid.
		 */
		private static boolean checkValid(MatrixStack[] W1, MatrixStack[] W2) {
			if (W1 == null && W2 == null) return false;
			MatrixStack[] W = W1 != null ? W1 : W2;
			if (W.length == 0) return false;
			if (W1 != null && W2 != null) return W1.length == W2.length && W1[0].depth() == W2[0].depth();
			return true;
		}

		@Override
		public net.ea.ann.mane.Weight.Kernel add(net.ea.ann.mane.Weight.Kernel kernel) {
			MatrixStack[] sum1 = this.W1 != null ? MatrixStack.sum(this.W1, ((Kernel)kernel).W1) : null;
			MatrixStack[] sum2 = this.W2 != null ? MatrixStack.sum(this.W2, ((Kernel)kernel).W2) : null;
			return new Kernel(sum1, sum2);
		}

		@Override
		public net.ea.ann.mane.Weight.Kernel multiply(double value) {
			MatrixStack[] d1 = this.W1 != null ? MatrixStack.multiply(this.W1, value) : null;
			MatrixStack[] d2 = this.W2 != null ? MatrixStack.multiply(this.W2, value) : null;
			return new Kernel(d1, d2);
		}

		@Override
		public net.ea.ann.mane.Weight.Kernel divide(double value) {
			MatrixStack[] d1 = this.W1 != null ? MatrixStack.divide(this.W1, value) : null;
			MatrixStack[] d2 = this.W2 != null ? MatrixStack.divide(this.W2, value) : null;
			return new Kernel(d1, d2);
		}

	}
	

	/**
	 * The kernel.
	 */
	protected Kernel kernel = null;
	
	
	/**
	 * Constructor with the first weight and the second weight.
	 * @param W1 the first weight.
	 * @param W2 the second weight.
	 */
	public WeightImpl(Kernel w) {
		this.kernel = w;
	}

	
	/**
	 * Getting the first weight.
	 * @return the second weight.
	 */
	private MatrixStack[] W1() {return kernel.W1;}
	
	
	/**
	 * Getting the first weight.
	 * @return the second weight.
	 */
	private MatrixStack[] W2() {return kernel.W2;}

	
	/**
	 * Getting non-null weight.
	 * @return non-null weight.
	 */
	private MatrixStack[] nonnull() {return W1() != null ? W1() : W2();}
	
	
	/**
	 * Getting internal weight.
	 * @return internal weight.
	 */
	public Kernel kernel() {return kernel;}
	
	
	/**
	 * Getting weight depth.
	 * @return weight depth.
	 */
	public int depth() {return nonnull()[0].depth();}

	
	/**
	 * Getting weight stack size.
	 * @return weight stack size.
	 */
	public int time() {return nonnull().length;}

	
	/**
	 * Getting first weight.
	 * @param time time.
	 * @param depth depth.
	 * @return first weight.
	 */
	private Matrix W1(int time, int depth) {
		return W1() != null ? W1()[time].get(depth) : null;
	}
	
	
	/**
	 * Getting second weight.
	 * @param time time.
	 * @param depth depth.
	 * @return second weight.
	 */
	private Matrix W2(int time, int depth) {
		return W2() != null ? W2()[time].get(depth) : null;
	}


	@Override
	public Weight accumKernel(net.ea.ann.mane.Weight.Kernel dKernel, double factor) {
		this.kernel = (Kernel)this.kernel.add(dKernel.multiply(factor));
		return this;
	}

	
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param inputs inputs.
	 * @param biases biases.
	 * @return evaluated value.
	 */
	private Matrix evaluate(int time, MatrixStack inputs, MatrixStack biases) {
		int depth = depth();
		if (inputs.depth() != depth || inputs.depth() != depth) throw new IllegalArgumentException();
		Matrix sum = null;
		for (int d = 0; d < depth; d++) {
			Matrix value = new WCore(W1(time, d), W2(time, d)).evaluate(inputs.get(d), biases.get(d));
			sum = sum != null ? sum.add(value) : value;
		}
		return sum;
	}
	
	
	/**
	 * Evaluating inputs.
	 * @param time time.
	 * @param inputs inputs.
	 * @param biases biases.
	 * @return evaluated value.
	 */
	private MatrixStack evaluate(MatrixStack inputs, MatrixStack biases) {
		int time = time();
		Matrix[] values = new Matrix[time];
		for (int t = 0; t < time; t++) {
			values[t] = evaluate(t, inputs, biases);
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

	
	/**
	 * Calculate gradient of previous layers.
	 * @param time time.
	 * @param prevInputs previous inputs.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layers.
	 */
	private MatrixStack dValue(int time, MatrixStack prevInputs, MatrixStack prevOutputs, MatrixStack thisErrors, Function prevActivateRef) {
		int depth = depth();
		if (prevInputs.depth() != depth || prevOutputs.depth() != depth) throw new IllegalArgumentException();
		Matrix[] dValues = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			dValues[d] = new WCore(W1(time, d), W2(time, d)).
				dValue(prevInputs.get(d), prevOutputs.get(d), thisErrors.get(d), prevActivateRef);
		}
		return new MatrixStack(dValues);
	}
	
	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevInputs previous inputs.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputs, MatrixStack prevOutputs, MatrixStack thisErrors, Function prevActivateRef) {
		int time = time();
		MatrixStack sum = null;
		for (int t = 0; t < time; t++) {
			MatrixStack dValues = dValue(t, prevInputs, prevOutputs, thisErrors, prevActivateRef);
			sum = sum != null ? (MatrixStack)sum.add(dValues) : dValues;
		}
		return sum;
	}


	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		MatrixStack prevInputs = prevInput instanceof MatrixStack ? (MatrixStack)prevInput : new MatrixStack(prevInput);
		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dValue = dValue(prevInputs, prevOutputs, thisErrors, prevActivateRef);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}
	
	
	/**
	 * Calculating gradient of the current first weight.
	 * @param time time.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of the current first weight.
	 */
	private MatrixStack dW1(int time, MatrixStack prevOutputs, MatrixStack thisErrors) {
		if (this.W1() == null) return null;
		int depth = depth();
		if (prevOutputs.depth() != depth) throw new IllegalArgumentException();
		Matrix[] dW1s = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			dW1s[d] = new WCore(W1(time, d), W2(time, d)).dW1(prevOutputs.get(d), thisErrors.get(d));
		}
		return new MatrixStack(dW1s);
	}
	
	
	/**
	 * Calculating gradient of the current first weight.
	 * @param time time.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of the current first weight.
	 */
	private MatrixStack[] dW1(MatrixStack prevOutputs, MatrixStack thisErrors) {
		if (this.W1() == null) return null;
		int time = time();
		MatrixStack[] dW1s = new MatrixStack[time];
		for (int t = 0; t < time; t++) {
			dW1s[t] = dW1(t, prevOutputs, thisErrors);
		}
		return dW1s;
	}
	
	
	/**
	 * Calculating gradient of the current second weight.
	 * @param time time.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of the current second weight.
	 */
	private MatrixStack dW2(int time, MatrixStack prevOutputs, MatrixStack thisErrors) {
		if (this.W2() == null) return null;
		int depth = depth();
		if (prevOutputs.depth() != depth) throw new IllegalArgumentException();
		Matrix[] dW2s = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			dW2s[d] = new WCore(W1(time, d), W2(time, d)).dW2(prevOutputs.get(d), thisErrors.get(d));
		}
		return new MatrixStack(dW2s);
	}


	/**
	 * Calculating gradient of the current first weight.
	 * @param time time.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of the current first weight.
	 */
	private MatrixStack[] dW2(MatrixStack prevOutputs, MatrixStack thisErrors) {
		if (this.W2() == null) return null;
		int time = time();
		MatrixStack[] dW2s = new MatrixStack[time];
		for (int t = 0; t < time; t++) {
			dW2s[t] = dW2(t, prevOutputs, thisErrors);
		}
		return dW2s;
	}


	@Override
	public Kernel dKernel(Matrix prevOutput, Matrix thisError) {
		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack[] dW1 = dW1(prevOutputs, thisErrors);
		MatrixStack[] dW2 = dW2(prevOutputs, thisErrors);
		return new Kernel(dW1, dW2);
	}


	@Override
	public void fill(double v) {
		MatrixStack[] W1 = W1();
		MatrixStack[] W2 = W1();
		if (W1 != null) {
			for (MatrixStack w1 : W1) MatrixStack.fill(w1, v);
		}
		if (W2 != null) {
			for (MatrixStack w2 : W2) MatrixStack.fill(w2, v);
		}
	}
	

	@Override
	public void fill(Random rnd) {
		MatrixStack[] W1 = W1();
		MatrixStack[] W2 = W1();
		if (W1 != null) {
			for (MatrixStack w1 : W1) MatrixStack.fill(w1, rnd);
		}
		if (W2 != null) {
			for (MatrixStack w2 : W2) MatrixStack.fill(w2, rnd);
		}
	}


	@Override
	public int sizeOfParams() {
		int size = 0;
		MatrixStack[] W1 = W1();
		MatrixStack[] W2 = W1();
		if (W1 != null) {
			for (MatrixStack w1 : W1) size += Matrix.capacity(w1);
		}
		if (W2 != null) {
			for (MatrixStack w2 : W2) size += Matrix.capacity(w2);
		}
		return size;
	}
	

	/**
	 * Creating weight.
	 * @param size size of kernel.
	 * @param hint hint value.
	 * @return weight.
	 */
	private static MatrixStack[] createW(Size size, NeuronValue hint) {
		if (size.width < 1 || size.height < 1 || hint == null) return null;
		int depth = 1, time = 1;
		if (size.depth < 1)
			time = depth = 1;
		else if (size.time < 1) {
			depth = size.depth;
			time = 1;
		}
		else {
			depth = size.depth;
			time = size.time;
		}
		MatrixStack[] W = new MatrixStack[time];
		for (int t = 0; t < time; t++) {
			Matrix matrix = Matrix.create(new Size(size.width, size.height, depth, 1), hint); 
			W[t] = matrix instanceof MatrixStack ? (MatrixStack)matrix : new MatrixStack(matrix);
		}
		return W;
	}
	
	
	/**
	 * Creating weight.
	 * @param sizeW1 hint of first weight.
	 * @param sizeW2 hint of second weight.
	 * @param hint hint.
	 * @return weight.
	 */
	public static WeightImpl create(Size sizeW1, Size sizeW2, NeuronValue hint) {
		MatrixStack[] W1 = sizeW1 != null ? createW(sizeW1, hint) : null;
		MatrixStack[] W2 = sizeW2 != null ? createW(sizeW2, hint) : null;
		return new WeightImpl(new Kernel(W1, W2));
	}


}



/**
 * This class represents specification of standard parametric weight.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class WCore implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * The first matrix.
	 */
	Matrix W1 = null;
	
	
	/**
	 * The second matrix.
	 */
	Matrix W2 = null;
	
	
	/**
	 * Constructor with the first weight and the second weight.
	 * @param W1 the first weight.
	 * @param W2 the second weight.
	 */
	WCore(Matrix W1, Matrix W2) {
		if (!checkValid(W1, W2)) throw new IllegalArgumentException();
		this.W1 = W1;
		this.W2 = W2;
	}
	

	/**
	 * Checking the first weight and the second weight.
	 * @param W1 the first weight.
	 * @param W2 the second weight.
	 * @return true if the two weights are mutually valid.
	 */
	private static boolean checkValid(Matrix W1, Matrix W2) {
		return W1 != null || W2 != null;
	}
	
	
	/**
	 * Evaluating input.
	 * @param input input.
	 * @param bias bias.
	 * @return evaluated value.
	 */
	Matrix evaluate(Matrix input, Matrix bias) {
		if (this.W1 != null) input = this.W1.multiply(input);
		if (this.W2 != null) input = input.multiply(this.W2);
		return input.add(bias);
	}
	
	
	/**
	 * Calculate gradient of previous layer.
	 * @param prevInput previous input.
	 * @param prevOutput previous output.
	 * @param thisError current error.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layer.
	 */
	Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		Matrix derivative = prevInput != null && prevActivateRef != null ? prevInput.derivativeWise(prevActivateRef) : null;

		//Updating errors based on weights.
		Matrix nextW1T = this.W1;
		Matrix nextW2 = this.W2;
		nextW1T = (nextW1T != null) ? nextW1T.transpose() : prevOutput.createIdentity(prevOutput.rows());
		nextW2 = (nextW2 != null) ? nextW2 : prevOutput.createIdentity(prevOutput.columns());
		
		Matrix[] errorArray = new Matrix[nextW2.rows()];
		Matrix vecNextError = thisError.vec(); //Please pay attention to this code line.
		for (int row = 0; row < errorArray.length; row++) {
			//errorArray[row] = Matrix.kroneckerProductMutilply(nextW2, nextW1T, row, vecNextError); //Lower but consuming less memory.
			errorArray[row] = nextW2.kroneckerProductRowOf(nextW1T, row).multiply(vecNextError); //Faster.
		}
		Matrix prevError = Matrix.concatV(errorArray);
		return derivative != null ? derivative.multiplyWise(prevError) : prevError;
	}
	
	
	/**
	 * Calculating gradient of the current first weight.
	 * @param prevOutput previous output.
	 * @param thisError current error.
	 * @return gradient of the current first weight.
	 */
	Matrix dW1(Matrix prevOutput, Matrix thisError) {
		if (this.W1 == null) return null;
		
		Matrix vecError = thisError.vec();
		Matrix XW2 = this.W2 != null ? prevOutput.multiply(this.W2) : prevOutput;
		Matrix I = this.W1.createIdentity(this.W1.rows());
		Matrix[] W1s = new Matrix[XW2.rows()];
		for (int row = 0; row < W1s.length; row++) {
			//W1s[row] = Matrix.kroneckerProductMutilply(XW2, I, row, vecError); //Lower but consuming less memory.
			W1s[row] = XW2.kroneckerProductRowOf(I, row).multiply(vecError); //Faster.
		}
		return Matrix.concatV(W1s);
	}
	
	
	/**
	 * Calculating gradient of the current second weight.
	 * @param prevOutput previous output.
	 * @param thisError current error.
	 * @return gradient of the current second weight.
	 */
	Matrix dW2(Matrix prevOutput, Matrix thisError) {
		if (this.W2 == null) return null;
		
		Matrix vecError = thisError.vec();
		Matrix W1XT = this.W1 != null ? this.W1.multiply(prevOutput) : prevOutput;
		W1XT = W1XT.transpose();
		Matrix I = this.W2.createIdentity(this.W2.columns());
		Matrix[] W2s = new Matrix[I.rows()];
		for (int row = 0; row < W2s.length; row++) {
			//W2s[row] = Matrix.kroneckerProductMutilply(I, W1XT, row, vecError); //Lower but consuming less memory.
			W2s[row] = I.kroneckerProductRowOf(W1XT, row).multiply(vecError); //Faster.
		}
		return Matrix.concatV(W2s);
	}


}
