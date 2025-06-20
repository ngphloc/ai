/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.rnn;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.LayerStandardAbstract;
import net.ea.ann.core.LayerStandardAssoc;
import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Point;
import net.ea.ann.raster.Size;

/**
 * This class is default implementation of recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RecurrentNetworkImpl extends RecurrentNetworkAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Recurrent network layout.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static enum Layout {
		
		/**
		 * Output-input layout.
		 * In output-input layout, outputs of previous state become inputs of current state.
		 * In other words, output layer of previous state connects to the first hidden layer of current state.
		 * This implies the current state has two input layers, one input layer from itself and one input layer from previous state.
		 */
		outin,
		
		/**
		 * Parallel layout.
		 * In parallel layout, all layers of previous network connect parallel with all layers of previous network.
		 * Note, all connections are parallel because any pair of neurons (between two networks) have only one connection.
		 * It is implied that there is only one input layer for entire parallel layout recurrent network. This unique input layer is attached to the first layer. 
		 */
		parallel,
		
	}
	
	
	/**
	 * Constructor with neuron channel, activation functions, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param auxActivateRef auxiliary activation function.
	 * @param idRef identifier reference.
	 */
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef, Function auxActivateRef, Id idRef) {
		super(neuronChannel, activateRef, auxActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param auxActivateRef auxiliary activation function.
	 */
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef, Function auxActivateRef) {
		this(neuronChannel, activateRef, auxActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public RecurrentNetworkImpl(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	/**
	 * Initialize according to specified layout.
	 * @param nTotalNeuron number of total neurons.
	 * @param size size of states.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nTotalNeuron, Size size) {
		if (nTotalNeuron < 2) return false;
		int[] nHiddenNeuron = null;
		if (nTotalNeuron > 2) {
			nHiddenNeuron = new int[nTotalNeuron - 2];
			for (int i = 0; i < nHiddenNeuron.length; i++) nHiddenNeuron[i] = 1; 
		}
		return initialize(1, 1, nHiddenNeuron, size);
	}
	
	
	/**
	 * Initialize according to specified layout.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons as well as number of hidden layers.
	 * @param size size of states.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, Size size) {
		size = size != null ? size : Size.unit();
		states.clear();
		for (int t = 0; t < size.time; t++) {
			for (int z = 0; z < size.depth; z++) {
				for (int y = 0; y < size.height; y++) {
					for (int x = 0; x < size.width; x++) {
						State state = newState();
						state.initialize(nInputNeuron, nOutputNeuron, nHiddenNeuron, 0);
						states.add(state);
						
						Point[] neighbors = getNeighbors(new Point(x, y, z, t));
						if (neighbors == null || neighbors.length == 0) continue;
						for (Point neighbor : neighbors) {
							State nextState = get(neighbor);
							connect(state, nextState);
						} //End neighbors.
					} //End x.
				} //End y.
			} //End z.
		} //End t.
		
		return true;
	}


	/**
	 * Connect two states.
	 * @param state current state.
	 * @param nextState next state.
	 * @return true if connection is successful.
	 */
	protected boolean connect(State state, State nextState) {
		return connect(state, nextState, Layout.outin);
	}
	
	
	/**
	 * Connect two states.
	 * @param state current state.
	 * @param nextState next state.
	 * @param layout layout.
	 * @return true if connection is successful.
	 */
	private static boolean connect(State state, State nextState, Layout layout) {
		if (state == null || nextState == null)
			return false;
		else if (layout == Layout.parallel) {
			List<LayerStandard> backbone = state.getBackbone();
			List<LayerStandard> nextBackbone = nextState.getBackbone();
			int n = Math.min(backbone.size(), nextBackbone.size());
			if (n < 2) return false;
			for (int i = 1; i < n; i++) connectRibout(backbone.get(i), nextBackbone.get(i));
			return true;
		}
		else {
			LayerStandard out = state.getOutputLayer();
			LayerStandard[] hiddens = nextState.getHiddenLayers();
			LayerStandard in = hiddens != null && hiddens.length > 0 ? hiddens[0] : nextState.getOutputLayer();
			return connectOutside(out, in);
		}
	}
	
	
	/**
	 * Connecting two layers by outside connection.
	 * @param layer current layer.
	 * @param nextLayer next layer.
	 * @return true if connecting is successful.
	 */
	private static boolean connectOutside(LayerStandard layer, LayerStandard nextLayer) {
		if (layer == null || nextLayer == null) return false;
		layer.addOutsideNextVirtualLayer(nextLayer);
		return true;
	}
	
	
	/**
	 * Connecting two layers by rib-out connection.
	 * @param layer current layer.
	 * @param nextLayer next layer.
	 * @return true if connecting is successful.
	 */
	private static boolean connectRibout(LayerStandard layer, LayerStandard nextLayer) {
		if (layer == null || nextLayer == null) return false;
		new LayerStandardAssoc((LayerStandardAbstract)layer).setRiboutLayer(nextLayer);
		return true;
	}


}
