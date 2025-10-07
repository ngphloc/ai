/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.bp.Backpropagator;
import net.ea.ann.core.bp.BackpropagatorAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This class is default implementation of standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NetworkStandardImpl extends NetworkStandardAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;


	/**
	 * Backpropagation algorithm.
	 */
	protected Backpropagator bp = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public NetworkStandardImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		
		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, true)) : activateRef;

		this.bp = createBackpropagator();
	}


	/**
	 * Creating backpropagation algorithm. Derived class can override this method.
	 * @return backpropagation algorithm.
	 */
	protected Backpropagator createBackpropagator() {
		return new BackpropagatorAbstract() {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
				return thisNetwork().calcOutputError(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
			}

		};
	}
	
	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public NetworkStandardImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public NetworkStandardImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}
	

	/**
	 * Getting this network.
	 * @return this network.
	 */
	protected NetworkStandardImpl thisNetwork() {return this;}
	
	
	@Override
	protected LayerStandard newLayer() {
		return new LayerStandardImpl(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public NetworkStandardAbstract getNetwork() {
				return thisNetwork();
			}
			
		};
	}
	

	/**
	 * Getting activation function.
	 * @return activation function.
	 */
	public Function getActivateRef() {
		return activateRef;
	}
	
	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	public int getNeuronChannel() {
		return neuronChannel;
	}
	

	@Override
	public NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			for (Record record : subsample) {
				if (record == null) continue;
				NeuronValue[] output = record.output != null? NeuronValue.adjustArray(record.output, backbone.get(backbone.size()-1).size(), backbone.get(backbone.size()-1)) : null;
				
				//Evaluating network.
				try {
					evaluate(record);
				} catch (Throwable e) {Util.trace(e);}
				
				//Learning backbone.
				error = bp.updateWeightsBiases(backbone, output, lr);
				
				//Learning rib-bone and memory.
				learnRibMem(record, lr);
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ann_backpropogate",
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "ann_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}


	@Override
	public NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			//Learning backbone.
			error = bp.updateWeightsBiases(subsample, backbone, lr, this);			
			
			//Learning rib-bone and memory.
			learnRibMem(subsample, lr);
		
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ann_backpropogate",
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "ann_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Learning by back propagate algorithm.
	 * @param input input values.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public synchronized NeuronValue[] learn(NeuronValue[] input, double learningRate, double terminatedThreshold, int maxIteration) {
		return learn(getBackbone(), input, true, learningRate, terminatedThreshold, maxIteration);
	}
	
	
	/**
	 * Learning by back propagate algorithm.
	 * @param input input values.
	 * @param realOutput realistic output. It can be null;
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public synchronized NeuronValue[] learn(NeuronValue[] input, NeuronValue[] realOutput, double learningRate, double terminatedThreshold, int maxIteration) {
		return learn(getBackbone(), input, realOutput, true, learningRate, terminatedThreshold, maxIteration);
	}
	
	
	/**
	 * Learning bone by back propagate algorithm.
	 * @param bone list of layers including input layer.
	 * @param input input values.
	 * @param learningRibMem flag to indicate whether to learn rib-bones and memory.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learn(List<LayerStandard> bone, NeuronValue[] input, boolean learningRibMem, double learningRate, double terminatedThreshold, int maxIteration) {
		return learn(bone, input, (NeuronValue[]) null, learningRibMem, learningRate, terminatedThreshold, maxIteration);
	}
	
	
	/**
	 * Learning bone by back propagate algorithm.
	 * @param bone list of layers including input layer.
	 * @param input input values.
	 * @param realOutput realistic output. It can be null;
	 * @param learningRibMem flag to indicate whether to learn rib-bones and memory.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learn(List<LayerStandard> bone, NeuronValue[] input, NeuronValue[] realOutput, boolean learningRibMem, double learningRate, double terminatedThreshold, int maxIteration) {
		if (bone == null || bone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		while (maxIteration <= 0 || iteration < maxIteration) {
			double lr = calcLearningRate(learningRate, iteration+1);

			//Evaluating layers.
			evaluate(bone, input);
			
			//Learning main bone.
			error = bp.updateWeightsBiases(bone, realOutput, lr);
			
			//Learning rib-bone and memory.
			if (learningRibMem) learnRibMem((Record)null, lr);
			
			iteration ++;
			
			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				break;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
		}
		
		return error;
	}
	

	/**
	 * Learning bone by back propagate algorithm.
	 * @param input input. Note, index of layer is ID.
	 * @param output output. Note, index of layer is ID.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return errors of output errors. Return null if errors occur.
	 */
	public synchronized Map<Integer, NeuronValue[]> learn(
			Map<Integer, NeuronValue[]> input, Map<Integer, NeuronValue[]> output,
			double learningRate, double terminatedThreshold, int maxIteration) {
		return learn(input, output, null, learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning bone by back propagate algorithm.
	 * @param input input. Note, index of layer is ID.
	 * @param output output. Note, index of layer is ID.
	 * @param ribMemOutput rib bone and memory outputs.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return errors of output errors. Return null if errors occur.
	 */
	private Map<Integer, NeuronValue[]> learn(
			Map<Integer, NeuronValue[]> input, Map<Integer, NeuronValue[]> output,
			Record ribMemOutput,
			double learningRate, double terminatedThreshold, int maxIteration) {
		Map<Integer, NeuronValue[]> ribinOutput = ribMemOutput != null ? ribMemOutput.ribinOutput : null;
		Map<Integer, NeuronValue[]> riboutOutput = ribMemOutput != null ? ribMemOutput.riboutOutput : null;
		NeuronValue[] memOutput = ribMemOutput != null ? ribMemOutput.memOutput : null;

		return learn(getBackbone(), input, output, ribinOutput, riboutOutput, memOutput, learningRate, terminatedThreshold, maxIteration);
	}
	
	
	/**
	 * Learning bone by back propagate algorithm.
	 * @param bone list of layers including input layer.
	 * @param boneInput bone input. Note, index of layer is ID.
	 * @param boneOutput bone output. Note, index of layer is ID.
	 * @param ribinOutput rib-in bone output. Note, index of layer is ID.
	 * @param riboutOutput rib-out bone output. Note, index of layer is ID.
	 * @param memOutput memory output.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return errors of output errors. Return null if errors occur.
	 */
	private Map<Integer, NeuronValue[]> learn(List<LayerStandard> bone,
			Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput,
			Map<Integer, NeuronValue[]> ribinOutput, Map<Integer, NeuronValue[]> riboutOutput,
			NeuronValue[] memOutput,
			double learningRate, double terminatedThreshold, int maxIteration) {
		if (bone == null || bone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Map<Integer, NeuronValue[]> error = null;
		int iteration = 0;
		while (maxIteration <= 0 || iteration < maxIteration) {
			double lr = calcLearningRate(learningRate, iteration+1);

			//Learning main bone.
			error = bp.updateWeightsBiases(bone, boneInput, boneOutput, lr);
			
			//Learning rib-bone and memory.
			if (ribinOutput != null || riboutOutput != null || memOutput != null) {
				Record record = new Record();
				record.ribinOutput = ribinOutput;
				record.riboutOutput = riboutOutput;
				record.memOutput = memOutput;
				learnRibMem(record, lr);
			}
			
			iteration ++;
			
			if (error == null || error.size() == 0 || (iteration >= maxIteration && maxIteration == 1))
				break;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				NeuronValue[][] errors = error.values().toArray(new NeuronValue[][] {});
				double errorMean = NeuronValue.normMean(errors);
				if (errorMean < terminatedThreshold) break;
			}
		}
		
		return error;
	}
	
	
	/**
	 * Learning rib-bones and memory.
	 * @param record specified record.
	 * @param learningRate learning rate.
	 */
	protected void learnRibMem(Record record, double learningRate) {
		if (record == null) {
			//System.out.println("Rib bones and memory are not learned yet because of null record but this is not a error.");
			return;
		}
		List<LayerStandard> backbone = getBackbone();
		
		//Updating weights and biases of rib-in bone.
		List<List<LayerStandard>> ribinbones = getRibinbones();
		for (List<LayerStandard> ribinbone : ribinbones) {
			if (record.ribinOutput == null) break;
			if (ribinbone.size() < 2) continue;
			LayerStandard layer = ribinbone.get(ribinbone.size() - 1);
			int index = backbone.indexOf(layer);
			if (index < 0 || !record.ribinOutput.containsKey(index)) continue;
			
			if (record.ribinInput.containsKey(index)) evaluate(ribinbone, record.ribinInput.get(index));
			
			NeuronValue[] output = record.ribinOutput.get(index);
			if (output != null) bp.updateWeightsBiases(ribinbone, output, learningRate);
		}
		
		//Updating weights and biases of rib-out bone.
		List<List<LayerStandard>> riboutbones = getRiboutbones();
		for (List<LayerStandard> riboutbone : riboutbones) {
			if (record.riboutOutput == null) break;
			if (riboutbone.size() < 2) continue;
			LayerStandard layer = riboutbone.get(0);
			int index = backbone.indexOf(layer);
			if (index < 0 || !record.riboutOutput.containsKey(index)) continue;
			
			if (record.riboutInput.containsKey(index)) evaluate(riboutbone, record.riboutInput.get(index));

			NeuronValue[] output = record.riboutOutput.get(index);
			bp.updateWeightsBiases(riboutbone, output, learningRate);
		}
		
		//Updating weights and biases of memory layer.
		if (memoryLayer != null && memoryLayer.size() > 0 && record.memOutput != null) {
			List<LayerStandard> memoryBone = Arrays.asList(memoryLayer.getPrevLayer(), memoryLayer, memoryLayer.getNextLayer());
			bp.updateWeightsBiases(memoryBone, record.memOutput, learningRate);
		}
	}
	
	
	/**
	 * Learning rib-bones and memory.
	 * @param sample specified sample.
	 * @param learningRate learning rate.
	 */
	protected void learnRibMem(Iterable<Record> sample, double learningRate) {
		if (sample == null) {
			//System.out.println("Rib bones and memory are not learned yet because of null sample but this is not a error.");
			return;
		}
		List<LayerStandard> backbone = getBackbone();
		
		//Updating weights and biases of rib-in bone.
		List<List<LayerStandard>> ribinbones = getRibinbones();
		for (List<LayerStandard> ribinbone : ribinbones) {
			if (ribinbone.size() < 2) continue;
			LayerStandard attachedLayer = ribinbone.get(ribinbone.size()-1);
			int index = backbone.indexOf(attachedLayer);
			if (index < 0) continue;
			
			List<Record> newSample = Util.newList(0);
			for (Record record : sample) {
				Record newRecord = new Record();
				if (record.ribinInput != null && record.ribinInput.containsKey(index))
					newRecord.input = record.ribinInput.get(index);
				if (record.ribinOutput != null && record.ribinOutput.containsKey(index))
					newRecord.output = record.ribinOutput.get(index);
				if (newRecord.input != null || newRecord.output != null)
					newSample.add(newRecord);
			}
			
			if (newSample.size() > 0) {
				bp.updateWeightsBiases(newSample, ribinbone, learningRate, new Evaluator() {
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public NeuronValue[] evaluate(Record inputRecord) throws RemoteException {
						return inputRecord.input != null ? NetworkStandardAbstract.evaluate(ribinbone, inputRecord.input) : null;
					}
				});
			}
		}
		
		//Updating weights and biases of rib-out bone.
		List<List<LayerStandard>> riboutbones = getRiboutbones();
		for (List<LayerStandard> riboutbone : riboutbones) {
			if (riboutbone.size() < 2) continue;
			LayerStandard attachedLayer = riboutbone.get(0);
			int index = backbone.indexOf(attachedLayer);
			if (index < 0) continue;
			
			List<Record> newSample = Util.newList(0);
			for (Record record : sample) {
				Record newRecord = new Record();
				if (record.riboutInput != null && record.riboutInput.containsKey(index))
					newRecord.input = record.riboutInput.get(index);
				if (record.riboutOutput != null && record.riboutOutput.containsKey(index))
					newRecord.output = record.riboutOutput.get(index);
				if (newRecord.input != null || newRecord.output != null)
					newSample.add(newRecord);
			}
			
			if (newSample.size() > 0) {
				bp.updateWeightsBiases(newSample, riboutbone, learningRate, new Evaluator() {
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public NeuronValue[] evaluate(Record inputRecord) throws RemoteException {
						return inputRecord.input != null ? NetworkStandardAbstract.evaluate(riboutbone, inputRecord.input) : null;
					}
				});
			}
		}
		
		//Updating weights and biases of memory layer. The following code need to be improved.
		if (memoryLayer != null && memoryLayer.size() > 0) {
			List<Record> newSample = Util.newList(0);
			for (Record record : sample) {
				if (record.memInput != null || record.memOutput != null)
					newSample.add(new Record(record.memInput, record.memOutput));
			}
			
			if (newSample.size() > 0) {
				List<LayerStandard> memoryBone = Arrays.asList(memoryLayer.getPrevLayer(), memoryLayer, memoryLayer.getNextLayer());
				bp.updateWeightsBiases(newSample, memoryBone, learningRate, new Evaluator() {
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public NeuronValue[] evaluate(Record inputRecord) throws RemoteException {
						return inputRecord.input != null ? NetworkStandardAbstract.evaluate(memoryBone, inputRecord.input) : null;
					}
				});
			}
		}
	}

	
	@Override
	protected NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		return BackpropagatorAbstract.calcOutputErrorDefault(outputNeuron, realOutput, outputLayer);
	}
	
	
}
