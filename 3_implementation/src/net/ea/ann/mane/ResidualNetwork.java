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
import net.ea.ann.mane.MatrixLayerAbstract.LayerSpec;
import net.ea.ann.mane.ResidualLayer.ResidualFunction;

/**
 * This class represents residual network (residual connection).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ResidualNetwork extends DropoutNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
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
		if (getOutputLayer().getWeight() == null) return true;
		if (!(getOutputLayer() instanceof ResidualLayer)) return true;
		ResidualLayer outputLayer = (ResidualLayer)getOutputLayer();
		if (outputLayer.inputLayer != null && outputLayer.residualRef != null) return true;
		
		if (outputLayer.inputLayer == null && outputLayer.residualRef != null) throw new IllegalArgumentException();
		if (outputLayer.inputLayer != null && outputLayer.residualRef == null) throw new IllegalArgumentException();
		assert (outputLayer.getOutputActivateRef() != null);
		
		MatrixLayerAbstract inputLayer = findInputLayer(outputLayer);
		if (inputLayer == null) return true;
		outputLayer.inputLayer = inputLayer;
		outputLayer.residualRef = new ResidualFunction(outputLayer.getOutputActivateRef());
		return true;
	}

	
	/**
	 * Checking whether input and output of the network have the same size.
	 * @param input input matrix.
	 * @param output output matrix.
	 * @return whether input and output of the network have the same size.
	 */
	private static boolean checkInoutputSameSize(Matrix input, Matrix output) {
		if ((input instanceof MatrixStack) && !(output instanceof MatrixStack)) return false;
		if (!(input instanceof MatrixStack) && (output instanceof MatrixStack)) return false;
		if (input.rows() != output.rows() || input.columns() != output.columns() || MatrixUtil.depth(input) != MatrixUtil.depth(output))
			return false;
		
		return true;
	}
	
	
	/**
	 * Finding input layer of specified output layer.
	 * @param outputLayer specified output layer.
	 * @return input layer of specified output layer.
	 */
	private MatrixLayerAbstract findInputLayer(MatrixLayerAbstract outputLayer) {
		if (outputLayer == null) return null;
		MatrixLayerAbstract prevLayer = null, found = null;
		while ((prevLayer = outputLayer.getPrevLayer()) != null) {
			Matrix input = prevLayer.queryInput(), output = outputLayer.queryOutput();
			if (checkInoutputSameSize(input, output) && prevLayer.getWeight() != null) found = prevLayer;
			outputLayer = prevLayer;
		}
		return found;
	}
	
	
}
