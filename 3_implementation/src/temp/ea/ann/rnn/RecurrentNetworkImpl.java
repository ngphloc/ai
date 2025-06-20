/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn;

import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.LayerStandardAbstract;
import net.ea.ann.core.LayerStandardAssoc;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This class is default implementation of recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RecurrentNetworkImpl extends NetworkAbstract implements RecurrentNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Activation function.
	 */
	protected Function activateRef = null;
	
	
	/**
	 * List of states which are neural networks.
	 */
	protected List<NetworkStandardImpl> states = Util.newList(0);
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		
		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, true)) : activateRef;
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public RecurrentNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public RecurrentNetworkImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	/**
	 * Resetting network.
	 */
	public void reset() {
		states.clear();
	}
	
	
	/**
	 * Initialize according to specified layout.
	 * @param nTotalNeuron number of total neurons.
	 * @param T number of states.
	 * @param layout specified layout.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nTotalNeuron, int T, Layout layout) {
		if (nTotalNeuron < 2) return false;
		int[] nHiddenNeuron = null;
		if (nTotalNeuron > 2) {
			nHiddenNeuron = new int[nTotalNeuron - 2];
			for (int i = 0; i < nHiddenNeuron.length; i++) nHiddenNeuron[i] = 1; 
		}
		
		return initialize(1, 1, nHiddenNeuron, T, layout);
	}
	
	
	/**
	 * Initialize according to specified layout.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons as well as number of hidden layers.
	 * @param T number of states.
	 * @param layout specified layout.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, int T, Layout layout) {
		boolean result = false;
		switch (layout) {
		case outin:
			result = initializeOutin(nInputNeuron, nOutputNeuron, nHiddenNeuron, T);
			break;
		case parallel:
			result = initializeParallel(nInputNeuron, nOutputNeuron, nHiddenNeuron, T);
			break;
		default:
			result = initializeOutin(nInputNeuron, nOutputNeuron, nHiddenNeuron, T);
			break;
		}
		
		return result;
	}
	
	
	/**
	 * Initialize with output-input layout.
	 * In output-input layout, outputs of previous state become inputs of current state.
	 * In other words, output layer of previous state connects to the first hidden layer of current state.
	 * This implies the current state has two input layers, one input layer from itself and one input layer from previous state.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons as well as number of hidden layers.
	 * @param T number of states.
	 * @return true if initialization is successful.
	 */
	private boolean initializeOutin(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, int T) {
		if (nHiddenNeuron == null || nHiddenNeuron.length == 0) return false;
		T = T < 1 ? 1 : T;
		states.clear();
		NetworkStandardImpl prevState = null;
		for (int t = 0; t < T; t++) {
			NetworkStandardImpl state = newState();
			state.initialize(nInputNeuron, nOutputNeuron, nHiddenNeuron, 0);
			states.add(state);
			
			if (prevState != null) {
				LayerStandard output = prevState.getOutputLayer();
				LayerStandard input = state.getHiddenLayers()[0];
				new LayerStandardAssoc((LayerStandardAbstract)output).setRiboutLayer(input);
			}
			
			prevState = state;
		}
		
		return true;
	}


	/**
	 * Initialize with parallel layout.
	 * In parallel layout, all layers of previous state connect parallel with all layers of previous state.
	 * Note, all connections are parallel because any pair of neurons (between two state) have only one connection.
	 * It is implied that there is only one input layer for entire parallel layout recurrent network. This unique input layer is attached to the first layer.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons as well as number of hidden layers.
	 * @param T number of states.
	 * @return true if initialization is successful.
	 */
	private boolean initializeParallel(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, int T) {
		T = T < 1 ? 1 : T;
		states.clear();
		NetworkStandardImpl prevState = null;
		for (int t = 0; t < T; t++) {
			NetworkStandardImpl state = newState();
			state.initialize(nInputNeuron, nOutputNeuron, nHiddenNeuron, 0);
			states.add(state);
			
			if (prevState != null) {
				List<LayerStandard> prevBackbone = prevState.getBackbone();
				List<LayerStandard> backbone = state.getBackbone();
				int n = Math.min(prevBackbone.size(), backbone.size());
				for (int i = 0; i < n; i++) {
					LayerStandard prevLayer = prevBackbone.get(i);
					LayerStandard layer = backbone.get(i);
					new LayerStandardAssoc((LayerStandardAbstract)prevLayer).setRiboutLayer(layer);
				}
			}
			
			prevState = state;
		}
		
		return true;
	}

	
	/**
	 * Creating state.
	 * @return standard neural network as state.
	 */
	protected NetworkStandardImpl newState() {
		return new NetworkStandardImpl(neuronChannel, activateRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public List<List<LayerStandard>> getRiboutbones() {
				return super.getShortRiboutbones();
			}

		};
	}
	
	
	/**
	 * Getting number of states.
	 * @return number of states.
	 */
	public int size() {
		return states.size();
	}
	
	
	/**
	 * Getting state at specified index.
	 * @param index specified index.
	 * @return state at specified index.
	 */
	public NetworkStandardImpl get(int index) {
		return states.get(index);
	}


	@Override
	public void evaluate(NeuronValue...input) throws RemoteException {
		if (states.size() == 0) return;
		states.get(0).evaluate(new Record(input != null ? input : new NeuronValue[] {}));
		for (int i = 1; i < states.size(); i++) {
			states.get(i).evaluate(new Record(new NeuronValue[] {}));
		}
	}
	
	
	@Override
	public void evaluate(List<NeuronValue[]> inputs) throws RemoteException {
		if (inputs == null || inputs.size() == 0) {
			evaluate();
			return;
		}
		if (states.size() == 0) return;
		
		int n = Math.min(states.size(), inputs.size());
		for (int i = 0; i < n; i++) {
			states.get(i).evaluate(new Record(inputs.get(i)));
		}
		
		for (int i = n; i < states.size(); i++) {
			states.get(i).evaluate(new Record(new NeuronValue[] {}));
		}
	}
	
	
	/**
	 * Evaluating recurrent neural network from starting state.
	 * @param startState starting state.
	 */
	public void evaluate(int startState) {
		if (startState < 0 || startState >= states.size()) return;
		for (int i = startState; i < states.size(); i++) {
			try {
				states.get(i).evaluate(new Record(new NeuronValue[] {}));
			} catch (Throwable e) {Util.trace(e);}
		}
	}
	
	
}
