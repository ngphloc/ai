package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.weight.NullWeight;
import net.hudup.core.logistic.NextUpdate;

/**
 * This class represents residual layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ResidualLayer extends DropoutLayer {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This class represents residual function.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class ResidualFunction implements Function {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Internal function.
		 */
		protected Function function = null;
		
		/**
		 * Constructor with function.
		 * @param function function.
		 */
		public ResidualFunction(Function function) {
			this.function = function;
		}

		@Override
		public NeuronValue evaluate(NeuronValue x) {
			return function.evaluate(x);
		}

		/*
		 * Checking this method.
		 */
		@NextUpdate
		@Override
		public NeuronValue derivative(NeuronValue x) {
			return function.derivative(x).add(x.unit()); //This code line is very important.
		}
		
	}
	
	
	/**
	 * Input layer.
	 */
	protected MatrixLayerAbstract inputLayer = null;

	
	/**
	 * Residual function.
	 */
	protected ResidualFunction residualRef = null;
	
	
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
	
	
	/**
	 * Getting input layer.
	 * @return input layer.
	 */
	public MatrixLayerAbstract getInputLayer() {return this.inputLayer;}
	
	
	/**
	 * Getting residual function.
	 * @return residual function.
	 */
	public ResidualFunction getResidualRef() {return this.residualRef;}
	
	
	/**
	 * Setting residual function.
	 * @param residualRef function.
	 */
	public void setResidualRef(ResidualFunction residualRef) {this.residualRef = residualRef;}
	
	
	/**
	 * Evaluating layer.
	 * @param params additional parameters.
	 * @return evaluated matrix as output.
	 */
	private Matrix evaluate0(Object...params) {
		if (this.getResidualRef() == null || this.getInputLayer() == null || this.getPrevLayer() == null) return super.evaluate(params);
		if (this.getWeight() == null) return super.evaluate(params);
		
		Matrix prevOutput = this.filter != null ? evaluateByFilter() : null;
		this.input = prevOutput != null ? prevOutput : this.prevLayer.queryOutput();
		this.input = this.weight.evaluate(this.input, this.bias);
		this.input = this.input.add(this.getInputLayer().queryInput()); //This code line is very important.
		this.output = !(this.weight instanceof NullWeight) ? this.input.evaluate0(this.getResidualRef()) : this.input;
		
		Error.addLayerOInput(this, params);
		return this.output;
	}

	
	@Override
	public Matrix evaluate(Object...params) {
		if (!isDropoutMode()) return this.evaluate0(params);
		if (!extractTrainingFlag(params)) {
			this.dropoutMask = null;
			return this.evaluate0(params);
		}
		setupMask();
		if (this.dropoutMask == null) return this.evaluate0(params);
        
		Matrix thisOutput = this.evaluate0(params);
		Matrix maskedOutput = this.dropoutMask.multiplyWise(thisOutput);
		if (thisOutput == this.output) this.output = maskedOutput;
        return maskedOutput;
	}

	
	@Override
	public Function getOutputActivateRef() {
		return this.residualRef == null ? super.getOutputActivateRef() : this.residualRef;
	}

	
}

