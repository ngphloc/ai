package net.ea.ann.mane.layers;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.MatrixLayerImpl;
import net.ea.ann.mane.weight.NullWeight;

/**
 * This class represents residual layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ResidualLayer extends MatrixLayerImpl  {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for residual mode.
	 */
	public final static String RESIDUAL_MODE_FIELD = "mane_layer_residual";
	
	
	/**
	 * Default value for residual mode.
	 */
	public final static boolean RESIDUAL_MODE_DEFAULT = true;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ResidualLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ResidualLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ResidualLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ResidualLayer(int neuronChannel) {this(neuronChannel, null, null, null);}
	
	
	@Override
	protected Matrix evaluateByFilter() {
		if (this.filter == null || this.prevLayer == null) return super.evaluateByFilter();
		if (!this.filter.doesApplyActivate() || this.filter.isIndexMode()) return super.evaluateByFilter();
		if (getStartLayer() == null) return super.evaluateByFilter();
//		assert (this.filter instanceof KernelFilterProduct);
		
		Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(this.prevLayer.queryOutput());
		Matrix thisPrevInputConv = matrixToConvLayer(this.prevInput);
		Matrix thisPrevOutputConv = matrixToConvLayer(this.prevOutput);
		
		this.filter.forward(prevLayerOutputConv, thisPrevInputConv, thisPrevOutputConv, Kernel.GLOBAL_BIAS ? this.filterBias : null, null/*this.getFilterActivateRef()*/); //Please pay attention to this code line.
		this.prevInput = convLayerToMatrix(thisPrevInputConv);
		this.prevInput = this.prevInput.add(getStartLayer().queryInput()); //This code line is important for residual network.
		return this.prevOutput = this.prevInput.evaluate0(this.getFilterActivateRef());
	}

	
	@Override
	public Matrix evaluate(Object...params) {
		if (getStartLayer() == null) return super.evaluate(params);
		if ((getNetwork() == null) || !(getNetwork() instanceof ResidualNetwork)) return super.evaluate(params);
		
		assert (this.prevLayer != null); //Do not evaluate the input layer.
		if (this.prevLayer == null) return null;
		Matrix prevOutput0 = this.filter != null ? evaluateByFilter() : null;
		if (this.weight == null) {
			Error.addLayerOInput(this, params);
			assert (prevOutput0 != null && prevOutput0 == this.prevOutput);
			return prevOutput0;
		}
		
		this.input = prevOutput0 != null ? prevOutput0 : this.prevLayer.queryOutput();
		this.input = this.weight.evaluate(this.input, Kernel.GLOBAL_BIAS ? this.bias : null);
		this.input = this.input.add(this.getStartLayer().queryOutput()); //This code line is important for residual network.
		this.output = (this.getWeightActivateRef() != null) && !(this.weight instanceof NullWeight) ?
			this.input.evaluate0(this.getWeightActivateRef()) : this.input;
		
		Error.addLayerOInput(this, params);
		return this.output;
	}

	
}

