/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixNetworkImpl.TrainingFlag;
import net.ea.ann.mane.weight.NullWeight;
import net.ea.ann.raster.Size;

/**
 * This class implements layer in matrix neural network with dropout technique.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class DropoutLayer extends MatrixLayerImpl implements MatrixLayerExt {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for dropout mode.
	 */
	public final static String DROPOUT_MODE_FIELD = "mane_dropout";
	
	
	/**
	 * Default value for dropout mode.
	 */
	public final static boolean DROPOUT_MODE_DEFAULT = true;

	
	/**
	 * Field for dropout rate.
	 */
	public final static String DROPOUT_RATE_FIELD = "mane_dropout_rate";
	
	
	/**
	 * Default value for dropout rate.
	 */
	public final static double DROPOUT_RATE_DEFAULT = 0.2;

	
	/**
	 * Field for inverted mode.
	 */
	public final static String DROPOUT_INVERTED_FIELD = "mane_dropout_inverted";
	
	
	/**
	 * Default value for inverted mode.
	 */
	public final static boolean DROPOUT_INVERTED_DEFAULT = true;

	
	/**
	 * Field for dropout all.
	 */
	public final static String DROPOUT_ALL_FIELD = "mane_dropout_all";
	
	
	/**
	 * Default value for dropout all.
	 */
	public final static boolean DROPOUT_ALL_DEFAULT = true;

	
	/**
	 * Dropout mask.
	 */
	protected Matrix dropoutMask = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public DropoutLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}


	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public DropoutLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public DropoutLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public DropoutLayer(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	@Override
	public void reset() {
		super.reset();
		this.dropoutMask = null;
	}

	
	/**
	 * Checking dropout mode.
	 * @return dropout mode.
	 */
	boolean isDropoutMode() {
		if ((getNetwork() == null) || !(getNetwork() instanceof DropoutNetwork)) return DROPOUT_MODE_DEFAULT;
		return ((DropoutNetwork)getNetwork()).paramIsDropoutMode();
	}
	
	
	/**
	 * Getting dropout rate.
	 * @return dropout rate.
	 */
	private double getDropoutRate() {
		if ((getNetwork() == null) || !(getNetwork() instanceof DropoutNetwork)) return DROPOUT_RATE_DEFAULT;
		double dropoutRate = ((DropoutNetwork)getNetwork()).paramGetDropoutRate();
		return dropoutRate > 0 && dropoutRate < 1 ? dropoutRate : 0;
	}
	
	
	/**
	 * Checking whether to be in mode of inverted dropout.
	 * @return whether to be in mode of inverted dropout.
	 */
	private boolean isDropoutInverted() {
		if ((getNetwork() == null) || !(getNetwork() instanceof DropoutNetwork)) return DROPOUT_INVERTED_DEFAULT;
		return ((DropoutNetwork)getNetwork()).paramIsDropoutInverted();
	}
	
	
	/**
	 * Getting flag of dropouting all layers.
	 * @return flag of dropouting all layers.
	 */
	private boolean isDropoutAll() {
		if ((getNetwork() == null) || !(getNetwork() instanceof DropoutNetwork)) return DROPOUT_ALL_DEFAULT;
		return ((DropoutNetwork)getNetwork()).paramIsDropoutAll();
	}
	
	
	/**
	 * Getting dropout mask.
	 * @return dropout mask.
	 */
	public Matrix getDropoutMask() {return dropoutMask;}
	
	
	@Override
	public boolean initialize(Size size, Size prevSize, LayerSpec layerSpec) {
		this.dropoutMask = null;
		return super.initialize(size, prevSize, layerSpec);
	}

	
	/**
	 * Setting up dropout mask.
	 */
	void setupMask() {
		this.dropoutMask = null;
		if (!isDropoutMode()) return;
		if (!isDropoutAll() && this.getNetwork() != null && this.getNetwork().getOutputLayer() != this.getNextLayer()) return;//Dropout the layer whose next layer is output layer is enough because the back-propagation mechanism.
		if (getDropoutRate() <= 0 || getDropoutRate() >= 1) return;
		if (getWeight() == null && getFilter() == null) return; //Do not dropout the input layer.
		if (this.getNetwork() != null && this.getNetwork().getOutputLayer() == this) return; //Do not dropout the output layer.
		
		if ( (getWeight() != null && !(getWeight() instanceof NullWeight)) || (getFilter() != null && !getFilter().isIndexMode()) ) {
			Matrix thisOutput = queryOutput();
			this.dropoutMask = thisOutput.create(new Size(thisOutput.columns(), thisOutput.rows()));
		}
		if (this.dropoutMask == null) return;
		
		double keepProb = 1.0 - getDropoutRate();
        double scale = isDropoutInverted() ? 1.0/keepProb : 1.0;
        Random rnd = new Random();
        for (int row = 0; row < this.dropoutMask.rows(); row++) {
        	for (int column = 0; column < this.dropoutMask.columns(); column++) {
                if (rnd.nextDouble() < keepProb)
                	setValue(this.dropoutMask, row, column, scale);
                else
                	setValue(this.dropoutMask, row, column, 0);
        	}
        }
	}

	
	@Override
	public Matrix evaluate(Object...params) {
		if (!isDropoutMode()) return super.evaluate(params);
		if (!extractTrainingFlag(params)) {
			this.dropoutMask = null;
			return super.evaluate(params);
		}
		setupMask();
		if (this.dropoutMask == null) return super.evaluate(params);
        
		Matrix thisOutput = super.evaluate(params);
		Matrix maskedOutput = this.dropoutMask.multiplyWise(thisOutput);
		if (thisOutput == this.output) this.output = maskedOutput;
        return maskedOutput;
	}

	
	/**
	 * Extracting training flag.
	 * @param params parameters.
	 * @return training flag.
	 */
	static boolean extractTrainingFlag(Object[] params) {
		if (params == null || params.length == 0) return false;
		for (Object param : params) {
			if (param != null && param instanceof TrainingFlag) return true;
		}
		return false;
	}
	
	
	@Override
	Matrix adjustError(Matrix error, Error ERROR) {
		Matrix mask = ERROR.oinputDropoutMaskOfLayer(this);
		return mask != null ? mask.multiplyWise(error) : error;
	}


	/**
	 * Setting value of matrix.
	 * @param matrix matrix.
	 * @param row row.
	 * @param column column.
	 * @param v value.
	 */
	private static void setValue(Matrix matrix, int row, int column, double v) {
		if (matrix instanceof MatrixStack) {
			MatrixStack stack = (MatrixStack)matrix;
			NeuronValue zero = stack.get().get(0, 0).zero();
			NeuronValue value = v == 0 ? zero : zero.valueOf(v);
			for (int d = 0; d < stack.depth(); d++) stack.get(d).set(row, column, value);
		}
		else {
			NeuronValue zero = matrix.get(0, 0).zero();
			NeuronValue value = v == 0 ? zero : zero.valueOf(v);
			matrix.set(row, column, value);
		}
	}
	
	
}
