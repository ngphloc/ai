/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;

/**
 * This class represents residual network (residual connection).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ResidualNetwork extends MatrixNetworkImpl {


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
	static class ResidualFunction implements Function {
		
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

		@Override
		public NeuronValue derivative(NeuronValue x) {
			return function.derivative(x).add(x.unit()); //This code line is very important.
		}
		
	}
	
	
	/**
	 * This class represents residual layer.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class ResidualLayer extends MatrixLayerImpl {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

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
		 * Getting residual function.
		 * @return residual function.
		 */
		public ResidualFunction getResidualRef() {return this.residualRef;}
		
		/**
		 * Setting residual function.
		 * @param residualRef function.
		 */
		public void setResidualRef(ResidualFunction residualRef) {this.residualRef = residualRef;}
		
		@Override
		public Matrix evaluate(Object...params) {
			if (this.getResidualRef() == null || this.getNetwork() == null || this.prevLayer == null) return super.evaluate(params);

			this.input = this.filter != null ? evaluateByFilterCall() : null;
			if (this.weight != null) {
				this.input = this.input != null ? this.input : this.prevLayer.queryOutput();
				this.input = this.weight.evaluate(this.input, this.bias);
			}

			this.input = this.input.add(this.getNetwork().getInput()); //This code line is very important.
			this.output = this.input.evaluate0(this.getResidualRef());
			return this.output;
		}

		@Override
		public Function getOutputActivateRef() {
			return this.residualRef == null ? super.getOutputActivateRef() : this.residualRef;
		}

	}
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ResidualNetwork(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ResidualNetwork(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ResidualNetwork(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}


	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ResidualNetwork(int neuronChannel) {this(neuronChannel, null, null, null);}


	@Override
	protected MatrixLayerAbstract newLayer() {
		ResidualLayer layer = new ResidualLayer(neuronChannel, getActivateRef(), getConvActivateRef(), idRef);
		layer.setNetwork(this);
		return layer;
	}


	@Override
	protected boolean initialize(LayerSpec[] layerSpecs, boolean dual) {
		if (!super.initialize(layerSpecs, dual)) return false;
		if (!checkInoutputSameSize()) return true;
		if (!(getOutputLayer() instanceof ResidualLayer)) return true;
		ResidualLayer outputLayer = (ResidualLayer)getOutputLayer();
		if (outputLayer.residualRef != null) return true;
		
		assert (outputLayer.getOutputActivateRef() != null);
		
		outputLayer.residualRef = new ResidualFunction(outputLayer.getOutputActivateRef());
		return true;
	}

	
	/**
	 * Checking whether input and output of the network have the same size.
	 * @return whether input and output of the network have the same size.
	 */
	private boolean checkInoutputSameSize() {
		if (!validate()) return false;
		Matrix input = getInput(), output = getOutput();
		
		if ((input instanceof MatrixStack) && !(output instanceof MatrixStack)) return false;
		if (!(input instanceof MatrixStack) && (output instanceof MatrixStack)) return false;
		if (input.rows() != output.rows() || input.columns() != output.columns() || MatrixUtil.depth(input) != MatrixUtil.depth(output))
			return false;
		
		return true;
	}
	
	
}
