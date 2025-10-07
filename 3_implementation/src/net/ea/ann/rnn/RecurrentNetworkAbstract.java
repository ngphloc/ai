/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.rnn;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.Generator;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Point;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is abstract implementation of recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class RecurrentNetworkAbstract extends NetworkAbstract implements RecurrentNetwork, Generator {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of Markov steps field.
	 */
	public final static String MARKOV_STEPS_FIELD = "rn_markov_steps";

	
	/**
	 * Default value of Markov steps field.
	 */
	public final static int MARKOV_STEPS_DEFAULT = 1;
	
	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Size.
	 */
	protected Size size = Size.unit();
	
	
	/**
	 * Activation function.
	 */
	protected Function activateRef = null;
	
	
	/**
	 * Auxiliary activation function which is often ramp function like ReLU.
	 */
	protected Function auxActivateRef = null;

	
	/**
	 * List of states which are neural networks.
	 */
	protected List<State> states = Util.newList(0);
	
	
	/**
	 * State trainer.
	 */
	protected Trainer stateTrainer = null;

	
	/**
	 * Constructor with neuron channel, activation functions, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param auxActivateRef auxiliary activation function which is often ramp function like ReLU.
	 * @param idRef identifier reference.
	 */
	protected RecurrentNetworkAbstract(int neuronChannel, Function activateRef, Function auxActivateRef, Id idRef) {
		super(idRef);
		
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		fillConfig(this.config);

		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		
		if (activateRef == null && auxActivateRef == null)
			this.auxActivateRef = this.activateRef = auxActivateRef = activateRef = Raster.toConvActivationRef(this.neuronChannel, isNorm());
		else if (activateRef != null && auxActivateRef != null) {
			this.activateRef = activateRef;
			this.auxActivateRef = auxActivateRef;
		}
		else if (activateRef != null)
			this.activateRef = activateRef; 
		else
			this.auxActivateRef = this.activateRef = activateRef = auxActivateRef;
	}

	
	/**
	 * Constructor with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param auxActivateRef auxiliary activation function which is often ramp function like ReLU.
	 */
	protected RecurrentNetworkAbstract(int neuronChannel, Function activateRef, Function auxActivateRef) {
		this(neuronChannel, activateRef, auxActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	protected RecurrentNetworkAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}


	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected RecurrentNetworkAbstract(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	/**
	 * Resetting network.
	 * @return this recurrent neural network.
	 */
	public RecurrentNetworkAbstract reset() {
		states.clear();
		return this;
	}
	

	/**
	 * Setting state trainer.
	 * @param stateTrainer specified state trainer. It can be null.
	 * @return this generator.
	 */
	protected RecurrentNetworkAbstract setStateTrainer(Trainer stateTrainer) {
		this.stateTrainer = stateTrainer;
		return this;
	}

	
	/**
	 * Creating an empty neuron value.
	 * @param state specified state.
	 * @param layer specified layer.
	 * @return empty neuron value.
	 */
	protected NeuronValue newNeuronValue(State state, LayerStandard layer) {
		return state.newNeuronValueCaller(layer);
	}

	
	/**
	 * Create a new weight.
	 * @param state specified state.
	 * @param layer specified layer.
	 * @return new weight.
	 */
	protected Weight newWeight(State state, LayerStandard layer) {
		return state.newWeightCaller(layer);
	}

	
	/**
	 * Create a new bias.
	 * @param state specified state.
	 * @param layer specified layer.
	 * @return created bias.
	 */
	protected NeuronValue newBias(State state, LayerStandard layer) {
		return state.newBiasCaller(layer);
	}

	
	/**
	 * Create neuron.
	 * @param state specified state.
	 * @param layer specified layer.
	 * @return created neuron
	 */
	protected NeuronStandard newNeuron(State state, LayerStandard layer) {
		return state.newNeuronCaller(layer);
	}

	
	/**
	 * Create layer.
	 * @param state calling state.
	 * @return created layer.
	 */
	protected LayerStandard newLayer(State state) {
		return state.newLayerCaller();
	}

	
	/**
	 * Adjusting backbone. This method is necessary to out-in layout.
	 * @param state state.
	 * @param bone specified bone.
	 */
	private void adjustBone(State state, List<LayerStandard> bone) {
		if (bone.size() < 2 || bone.get(0) != state.getInputLayer() || bone.get(bone.size()-1) != state.getOutputLayer())
			return;
		LayerStandard ribinLayer = bone.get(0).getRibinLayer();
		if (ribinLayer != null && ribinLayer.size() > 0)
			return;
		LayerStandard outsidePrevLayer = bone.get(0).getOutsidePrevVirtualLayer();
		if (outsidePrevLayer.size() > 0)
			bone.add(0, outsidePrevLayer);
	}
	
	
	/**
	 * Updating weights and biases. Derived class can override this method.
	 * @param state calling state.
	 * @param bp backpropagation agorithm.
	 * @param bone list of layers including input layer.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the neuron output too. 
	 * @param lastError last error which is optional parameter for batch learning.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	protected NeuronValue[] updateWeightsBiases(State state, GeneratorStandard.Backpropagator bp, List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		adjustBone(state, bone);
		return state.updateWeightsBiasesCaller(bp, bone, outputBatch, lastError, learningRate);
	}
	
	
	/**
	 * Updating weights and biases for every layer. Derived class can override this method.
	 * @param state calling state.
	 * @param bp backpropagation agorithm.
	 * @param bone list of layers including input layer.
	 * @param boneInput bone input. Note, index of layer is ID.
	 * @param boneOutput bone output. Note, index of layer is ID.
	 * @param learningRate learning rate.
	 * @return errors of output errors.
	 */
	protected Map<Integer, NeuronValue[]> updateWeightsBiases(State state, GeneratorStandard.Backpropagator bp, List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
		adjustBone(state, bone);
		return state.updateWeightsBiasesCaller(bp, bone, boneInput, boneOutput, learningRate);
	}
	
	
	/**
	 * Calculate error of output neuron for the second activation function. Derived classes should implement this method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output.  
	 * @param state calling state.
	 * @param outputNeuron output neuron. Output value of this neuron is retrieved by method {@link NeuronStandard#getOutput()}.
	 * @param realOutput real output. It can be null because this method is flexible.
	 * @param outputLayer output layer. It can be null because this method is flexible.
	 * @param outputNeuronIndex index of output neuron. It can be -1 because this method is flexible. This is optional parameter.
	 * @param realOutputs real outputs. It can be null because this method is flexible. This is optional parameter.
	 * @param params optional parameter array which can be null. 
	 * @return error or loss of the output neuron.
	 */
	protected NeuronValue calcOutputError2(State state, NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		return state.calcOutputError2Caller(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
	}

	
	/**
	 * Creating state.
	 * @return standard neural network as state.
	 */
	protected State newState() {
		State state = new State(neuronChannel, activateRef);
		state.setParent(this);
		state.setTrainer(stateTrainer);
		state.setAuxActivateRef(auxActivateRef);
		return state;
	}
	
	
	/**
	 * Getting number of states.
	 * @return number of states.
	 */
	public int length() {
		return states.size();
	}
	
	
	/**
	 * Getting state at specified index.
	 * @param index specified index.
	 * @return state at specified index.
	 */
	public State get(int index) {
		return states.get(index);
	}

	
	/**
	 * Converting location to index.
	 * @param loc location.
	 * @return index.
	 */
	private int convertLocToIndex(Point loc) {
		int wh = size.width*size.height;
		int whd = wh*size.depth;
		return loc.t*whd + loc.z*wh + loc.y*size.width + loc.x;
	}
	
	
	/**
	 * Getting state at specified location.
	 * @param loc specified location.
	 * @return state at specified location.
	 */
	public State get(Point loc) {
		int index = convertLocToIndex(loc);
		return get(index);
	}
	
	
	/**
	 * Getting dimension.
	 * @return dimension of this network.
	 */
	private int getDim() {
		if (size.time > 1)
			return 4;
		else if (size.depth > 1)
			return 3;
		else if (size.height > 1)
			return 2;
		else if (size.width >= 1)
			return 1;
		else
			return 0;
	}

	
	/**
	 * Getting Markov steps.
	 * @return Markov steps.
	 */
	private int getMarkovSteps() {
		int k = config.getAsInt(MARKOV_STEPS_FIELD);
		return k < MARKOV_STEPS_DEFAULT ? MARKOV_STEPS_DEFAULT : k;
	}
	
	
	/**
	 * Getting neighbors of current location, not including itself.
	 * @param loc current location.
	 * @return neighbors of current location.
	 */
	protected Point[] getNeighbors(Point loc) {
		int dim = getDim();
		List<Point> neighbors = Util.newList(0);
		
		int k = getMarkovSteps();
		if (dim == 1) {
			for (int x = loc.x-1; x >= loc.x-k; x--) neighbors.add(new Point(x));
		}
		else if (dim == 2) {
			for (int y = loc.y; y >= loc.y-k; y--) {
				for (int x = loc.x+k; x >= loc.x-k; x--) {
					neighbors.add(new Point(x));
				}
			}
		}
		else if (dim == 3) {
			for (int z = loc.z; z >= loc.z-k; z--) {
				for (int y = loc.y+k; y >= loc.y-k; y--) {
					for (int x = loc.x+k; x >= loc.x-k; x--) {
						neighbors.add(new Point(x, y, z));
					}
				}
			}
		}
		else if (dim == 4) {
			for (int t = loc.t; t >= loc.t-k; t--) {
				for (int z = loc.z+k; z >= loc.z-k; z--) {
					for (int y = loc.y+k; y >= loc.y-k; y--) {
						for (int x = loc.x+k; x >= loc.x-k; x--) {
							neighbors.add(new Point(x, y, z, t));
						}
					}
				}
			}
		}
		
		if (neighbors.size() == 0) return neighbors.toArray(new Point[] {});
		
		List<Point> validNeighbors = Util.newList(0);
		Cube container = new Cube(new Point(0), size);
		int currentIndex = convertLocToIndex(loc);
		for (Point neighbor : neighbors) {
			if (!container.contains(neighbor)) continue;
			int index = convertLocToIndex(neighbor);
			if (index < currentIndex) validNeighbors.add(neighbor);
		}
		return validNeighbors.toArray(new Point[] {});
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
		
		//Evaluating remaining states. 
		for (int i = n; i < states.size(); i++) {
			states.get(i).evaluate(new Record(new NeuronValue[] {}));
		}
	}
	
	
	/**
	 * Evaluating by input list.
	 * @param inputs input list.
	 */
	public void evaluate(NeuronValue[][] inputs) {
		if (inputs == null || inputs.length == 0) {
			try {
				evaluate();
			} catch (Throwable e) {Util.trace(e);}
			return;
		}
		
		List<NeuronValue[]> inputList = Util.newList(inputs.length);
		for (NeuronValue[] input : inputs) inputList.add(input);
		try {
			evaluate(inputList);
		} catch (Throwable e) {Util.trace(e);}
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
	
	
	@Override
	public NeuronValue[] learnOneByOne(Iterable<List<Record>> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learnOneByOne(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	@Override
	public NeuronValue[] learn(Iterable<List<Record>> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learn(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning recurrent neural network one-by-one record over sample.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public NeuronValue[] learnOneByOne(Iterable<List<Record>> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			double lr = calcLearningRate(learningRate, iteration+1);
			Iterable<List<Record>> subsample = resample(sample, iteration, maxIteration);

			for (List<Record> records : subsample) {
				if (records == null) continue;
				for (int i = 0; i < states.size() && i < records.size(); i++) {
					State state = states.get(i);
					error = state.learnOneByOne(Arrays.asList(records.get(i)), lr, terminatedThreshold, 1);
				}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "rnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "rnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}
	
	
	/**
	 * Learning neural network.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public NeuronValue[] learn(Iterable<List<Record>> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			double lr = calcLearningRate(learningRate, iteration+1);
			Iterable<List<Record>> subsample = resample(sample, iteration, maxIteration);

			for (int i = 0; i < states.size(); i++) {
				List<Record> samplei = Util.newList(0);
				for (List<Record> records : subsample) {
					if (records != null && i < records.size()) samplei.add(records.get(i));
				}
				if (samplei.size() == 0) continue;
				
				State state = states.get(i);
				error = state.learn(samplei, lr, terminatedThreshold, 1);
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "rnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}//End while
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "rnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	protected boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Filling default configuration values.
	 * @param config network configuration.
	 */
	public static void fillConfig(NetworkConfig config) {
		config.put(MARKOV_STEPS_FIELD, MARKOV_STEPS_DEFAULT);
	}


}
