/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;

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
		config.put(ResidualLayer.RESIDUAL_MODE_FIELD, ResidualLayer.RESIDUAL_MODE_DEFAULT);
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
	 * Finding start layer of specified residual layer.
	 * @param residualLayer specified residual layer.
	 * @return start layer of specified residual layer.
	 */
	private MatrixLayerAbstract findStartLayer(MatrixLayerAbstract residualLayer) {
		if (residualLayer == null || residualLayer == getInputLayer()) return null;
		MatrixLayerAbstract layer = residualLayer, prevLayer = null, found = null;
		Matrix output = residualLayer.queryOutput();
		while ((prevLayer = layer.getPrevLayer()) != null) {
			if (prevLayer.getEndLayer() != null) break;
			Matrix input = prevLayer.queryInput();
			if (checkInoutputSameSize(input, output) &&
				/*(prevLayer.getWeight() != null || prevLayer.getFilter() != null) &&*/
				!(prevLayer instanceof ResidualLayer) && //It is possible to remove this condition.
				!(prevLayer instanceof DropoutLayer)) found = prevLayer;
			layer = prevLayer;
		}
		return found;
	}
	
	
//	/**
//	 * Getting residual layer of starting layer.
//	 * @param startLayer starting layer.
//	 * @return residual layer of starting layer.
//	 */
//	private ResidualLayer residualLayerOf(MatrixLayerAbstract startLayer) {
//		if (startLayer == null) return null;
//		MatrixLayerAbstract layer = startLayer, nextLayer = null;
//		while ((nextLayer = layer.getNextLayer()) != null) {
//			if (nextLayer instanceof ResidualLayer && ((ResidualLayer)nextLayer).getStartLayer() == startLayer)
//				return (ResidualLayer)nextLayer;
//			layer = nextLayer;
//		}
//		return null;
//	}
	
	
	/**
	 * Getting residual layers.
	 * @return residual layers.
	 */
	protected List<ResidualLayer> getResidualLayers() {
		List<ResidualLayer> residualLayers = Util.newList(0);
		if (this.layers == null) return residualLayers;
		for (int l = 0; l < this.layers.length; l++) {
			if (!(this.layers[l] instanceof ResidualLayer)) continue;
			ResidualLayer residualLayer = (ResidualLayer)this.layers[l];
			if (residualLayer.getStartLayer() != null) residualLayers.add(residualLayer);
		}
		return residualLayers;
	}
	
	
	/**
	 * Setting residual layer.
	 * @param residualIndex index of residual layer.
	 * @return true if setting is successful.
	 */
	protected boolean setResidualLayer(int residualIndex) {
		if (this.layers == null || residualIndex < 0 || residualIndex >= this.layers.length) return false;

		MatrixLayerAbstract layer = this.layers[residualIndex];
		if (!(layer instanceof ResidualLayer)) return false;
		ResidualLayer residualLayer = (ResidualLayer)layer;
		if (residualLayer.getStartLayer() != null) return false;
		
		MatrixLayerAbstract startLayer = findStartLayer(residualLayer);
		if (startLayer == null) return false;
		residualLayer.setStartLayer(startLayer);
		return true;
	}
	
	
	/**
	 * Unsetting residual layer.
	 * @param residualLayer residual layer.
	 * @return true if unsetting is successful.
	 */
	protected boolean unsetResidualLayer(ResidualLayer residualLayer) {
		if (residualLayer == null) return false;
		residualLayer.setStartLayer(null);
		return true;
	}
	
	
	/**
	 * Validating all residual layers.
	 * @return true if all residual layers are valid.
	 */
	protected boolean validateResidualLayers() {
		List<ResidualLayer> residualLayers = getResidualLayers();
		if (residualLayers.size() < 2) return true;
		for (int i = 1; i < residualLayers.size(); i++) {
			if (residualLayers.get(i-1) == residualLayers.get(i)) return false;
			int previousIndex = indexOf(residualLayers.get(i-1).getStartLayer());
			if (previousIndex < 0) return false;
			int index = indexOf(residualLayers.get(i).getStartLayer());
			if (index < 0) return false;
			
			if (previousIndex >= index) return false;
		}
		return true;
	}
	
	
//	@Override
//	protected Error[] backward(Error[] outputErrors, boolean learning, double learningRate) {
//		if (getResidualLayers().size() == 0) return super.backward(outputErrors, learning, learningRate);
//		assert (validate() && outputErrors != null && outputErrors.length > 0);
//		if (!validate() || outputErrors == null || outputErrors.length == 0) return null;
//		
//		MatrixLayerAbstract startResidualLayer = null;
//		Error[] residualErrors = null, startResidualErrors = null;
//		for (int l = layers.length-1; l >= 0; l--) {
//			assert (layers[l] instanceof MatrixLayerImpl); //Improving later.
//			assert (outputErrors != null && outputErrors.length > 0);
//			
//			if ( (!learning) || (!(layers[l] instanceof MatrixLayerImpl)) )
//				outputErrors = layers[l].backward(outputErrors, layers[l], learning, learningRate);
//			else
//				outputErrors = ((MatrixLayerImpl)layers[l]).backwardWithoutLearning(outputErrors, learningRate);
//			
//			//Back-warding residual errors.
//			if (startResidualErrors != null)
//				startResidualErrors = null;
//			if (layers[l] instanceof ResidualLayer && startResidualErrors == null) {
//				residualErrors = outputErrors;
//				startResidualLayer = ((ResidualLayer)layers[l]).getStartLayer();
//				assert (startResidualLayer != null);
//			}
//			if (l > 0 && startResidualLayer != null && layers[l-1] == startResidualLayer && residualErrors != null)
//				startResidualErrors = outputErrors;
//			if (startResidualErrors != null) {
//				//Accumulating starting residual error. Only filter and null weight are supported because error from previous normal weight layer must be transformed again.
//				for (int i = 0; i < startResidualErrors.length; i++) {
//					Matrix error = startResidualErrors[i].error().add(residualErrors[i].error());
//					startResidualErrors[i].errorSet(error);
//				}
//				outputErrors = startResidualErrors;
//				startResidualLayer = null;
//				startResidualErrors = null;
//				residualErrors = null;
//			}
//		}
//		
//		for (int i = layers.length-1; i >= 0; i--) {
//			if ( (!learning) || (!(layers[i] instanceof MatrixLayerImpl)) ) continue;
//			((MatrixLayerImpl)layers[i]).updateParametersFromBackwardInfo(outputErrors.length, learningRate);
//		}
//		
//		return outputErrors;
//	}


	/**
	 * Checking residual mode.
	 * @return residual mode.
	 */
	protected boolean paramIsResidualMode() {
		if (config.containsKey(ResidualLayer.RESIDUAL_MODE_FIELD))
			return config.getAsBoolean(ResidualLayer.RESIDUAL_MODE_FIELD);
		else
			return ResidualLayer.RESIDUAL_MODE_DEFAULT;
	}
	
	
	/**
	 * Setting residual mode.
	 * @param residual residual mode.
	 * @return this network.
	 */
	protected DropoutNetwork paramSetResidualMode(boolean residual) {
		config.put(ResidualLayer.RESIDUAL_MODE_FIELD, residual);
		return this;
	}


}
