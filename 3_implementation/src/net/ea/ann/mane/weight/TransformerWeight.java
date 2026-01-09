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
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.Kernel.NullKernel;
import net.ea.ann.mane.Weight;
import net.ea.ann.raster.Size;
import net.ea.ann.transformer.TransformerAssoc;
import net.ea.ann.transformer.TransformerImpl;
import net.ea.ann.transformer.TransformerInitializer;

/**
 * This class represents parametric weight based on transformer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerWeight extends NetworkWeightAbstract {


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
		super();
		Size size = new Size(thisSize.width, thisSize.height, prevSize.depth, thisSize.depth);
		this.kernel = new TKernel(neuronChannel, size);
	}

	
	@Override
	int width() {return kernel.width();}
	

	@Override
	int height() {return kernel.height();}


	@Override
	int depth() {return kernel.depth();}


	@Override
	int time() {return kernel.time();}

	
	/**
	 * Getting transformer at specified time and depth.
	 * @param time time.
	 * @param depth depth.
	 * @return transformer at specified time and depth.
	 */
	private TransformerImpl tra(int time, int depth) {return kernel.transformer(time, depth);}
	
	
	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {
		this.kernel = (TKernel)this.kernel.add(dKernel);
		return this;
	}


	@Override
	Matrix evaluate(int time, MatrixStack inputs, Matrix bias) {
		int depth = depth();
		Matrix sum = null;
		for (int d = 0; d < depth; d++) {
			Matrix value = tra(time, d).evaluate(inputs.get(d));
			sum = sum != null ? sum.add(value) : value;
		}
		return bias != null ? sum.add(bias) : sum;
	}
	

	@Override
	Matrix dValue(int time, MatrixStack prevInputs, Matrix prevOutput, Matrix thisError, Function prevActivateRef, boolean learning, double learningRate) {
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

	
	@Override
	public void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		for (int t = 0; t < time(); t++) {
			for (int d = 0; d < depth(); d++) {
				tra(t, d).updateParametersFromBackwardInfo(recordCount, learningRate);
			}
		}
	}
	

	@Override
	public void resetBackwardInfo() {
		for (int t = 0; t < time(); t++) {
			for (int d = 0; d < depth(); d++) {
				tra(t, d).resetBackwardInfo();
			}
		}
	}

	
	@Override
	public int sizeOfParams() {
		int size = 0;
		for (int t = 0; t < time(); t++) {
			for (int d = 0; d < depth(); d++) {
				size += new TransformerAssoc(tra(t, d)).sizeOfParams();
			}
		}
		return size;
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
				if (!new TransformerInitializer(this.transformers[t][d]).initializeOnlyEncoder(n, dm)) throw new IllegalArgumentException();
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

