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

import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.IdentityFunction1;
import net.ea.ann.core.function.LogisticFunction1;

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
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public NetworkStandardImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		
		if (neuronChannel <= 1 && activateRef == null) {
			this.neuronChannel = 1;
			this.activateRef = new IdentityFunction1();
		}
		else {
			this.neuronChannel = neuronChannel;
			this.activateRef = activateRef;
		}
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
	 * Default constructor.
	 */
	public NetworkStandardImpl() {
		this(1, null, null);
	}
	

//	@Override
//	protected void reset() {
//		super.reset();
//		neuronChannel = 1;
//		activateRef = null;
//	}

	
	@Override
	protected LayerStandard newLayer() {
		return new LayerStandardImpl(neuronChannel, activateRef, idRef);
	}
	

	@Override
	public synchronized NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return bpLearn(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning neural network by back propagate algorithm.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public synchronized NeuronValue[] bpLearn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			for (Record record : sample) {
				if (record == null || record.input == null || record.output == null) continue;
				NeuronValue[] output = record.output != null? NeuronValue.adjustArray(record.output, backbone.get(backbone.size()-1).size(), backbone.get(backbone.size()-1)) : null;
				
				//Evaluating layers.
				try {
					evaluate(record, true);
				} catch (Throwable e) {Util.trace(e);}
				
				//Updating weights and biases of backbone.
				error = bpUpdateWeightsBiases(backbone, output, learningRate);
				
				List<List<LayerStandard>> ribinbones = getRibinbones();
				for (List<LayerStandard> ribinbone : ribinbones) {
					if (ribinbone.size() < 2) continue;
					
					//Updating weights and biases of rib-in bone.
					try {
						bpUpdateWeightsBiases(ribinbone, ribinbone.get(ribinbone.size()-1).getOutput(), learningRate);
					} catch (Throwable e) {Util.trace(e);}
				}
				
				List<List<LayerStandard>> riboutbones = getRiboutbones();
				for (List<LayerStandard> riboutbone : riboutbones) {
					if (riboutbone.size() < 2) continue;
					
					//Updating weights and biases of rib-out bone.
					try {
						bpUpdateWeightsBiases(riboutbone, riboutbone.get(riboutbone.size()-1).getOutput(), learningRate);
					} catch (Throwable e) {Util.trace(e);}
				}
				
				//Updating weights and biases of memory layer.
				if (memoryLayer != null && memoryLayer.size() > 0) {
					try {
						List<LayerStandard> memoryBone = Arrays.asList(memoryLayer.getPrevLayer(), memoryLayer, memoryLayer.getNextLayer());
						bpUpdateWeightsBiases(memoryBone, memoryLayer.getNextLayer().getOutput(), learningRate);
					} catch (Throwable e) {Util.trace(e);}
				}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ann_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0)
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
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

	
	/**
	 * Learning bone by back propagate algorithm.
	 * @param bone list of layers including input layer.
	 * @param input input values.
	 * @param output output values. It can be null;
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public NeuronValue[] bpLearn(List<LayerStandard> bone, NeuronValue[] input, NeuronValue[] output, double learningRate, double terminatedThreshold, int maxIteration) {
		if (bone == null || bone.size() < 2) return null;
		if (output != null)
			output = NeuronValue.adjustArray(output, bone.get(bone.size()-1).size(), bone.get(bone.size()-1));
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		while (maxIteration <= 0 || iteration < maxIteration) {
			//Evaluating layers.
			evaluate(bone, input);
			
			//Updating weights and biases of backbone.
			error = bpUpdateWeightsBiases(bone, output, learningRate);
			
			iteration ++;
			
			if (error == null || error.length == 0)
				break;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) break; 
			}
			
		}
		
		return error;
	}
	
	
	/**
	 * Updating weights and biases.
	 * @param bone list of layers including input layer.
	 * @param output output of output layer. 
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	protected NeuronValue[] bpUpdateWeightsBiases(List<LayerStandard> bone, NeuronValue[] output, double learningRate) {
		if (bone.size() < 2) return null;
		NeuronValue[] outputError = null;
		
		NeuronValue[] nextError = null;
		for (int i = bone.size() - 1; i >= 1; i--) {
			LayerStandard layer = bone.get(i);
			NeuronValue[] error = NeuronValue.makeArray(layer.size(), layer);
			
			for (int j = 0; j < layer.size(); j++) {
				NeuronStandard neuron = layer.get(j);
				
				//Calculate error of current layer.
				if (i == bone.size() - 1) {
					error[j] = calcError(neuron, output != null? output[j] : null);
				}
				else {
					LayerStandard nextLayer = bone.get(i + 1);
					NeuronValue rsum = layer.newNeuronValue();
					WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
					for (WeightedNeuron target : targets) {
						int index = nextLayer.indexOf(target.neuron);
						rsum = rsum.add(nextError[index].multiply(target.weight.value));
					}
					
					NeuronValue out = neuron.getOutput();
					NeuronValue derivative = neuron.getActivateRef().derivative(out);
					error[j] = rsum.multiplyDerivative(derivative);
				}
				
				//Update biases of current layer.
				NeuronValue delta = error[j].multiply(learningRate);
				neuron.setBias(neuron.getBias().add(delta));
			}
			
			//Update weights of previous layer.
			LayerStandard prevLayer = bone.get(i - 1);
			for (int j = 0; j < prevLayer.size(); j++) {
				NeuronStandard prevNeuron = prevLayer.get(j);
				NeuronValue prevOut = prevNeuron.getOutput();
				
				WeightedNeuron[] targets = prevNeuron.getNextNeurons(layer);
				for (WeightedNeuron target : targets) {
					Weight nw = target.weight;
					int index = layer.indexOf(target.neuron);
					NeuronValue delta = error[index].multiply(prevOut).multiply(learningRate);
					nw.value = nw.value.add(delta);
				}
			}

			nextError = error;
			if (i == bone.size() - 1) outputError = error;
		}
		
		return outputError;
	}
	
	
	/**
	 * Calculate error of an neuron.
	 * @param neuron specific neuron.
	 * @param output real output.
	 * @return error or loss of the neuron.
	 */
	protected NeuronValue calcError(NeuronStandard neuron, NeuronValue output) {
		NeuronValue out = neuron.getOutput();
		NeuronValue derivative = neuron.getActivateRef().derivative(out);
		return output.subtract(out).multiplyDerivative(derivative);
	}
	
	
	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (NetworkStandardImpl network = new NetworkStandardImpl(1, new LogisticFunction1())) {
			network.initialize(3, 3, new int[] {3, 3, 3});
			//System.out.println(network.toString());
			
			Record record1 = new Record();
			record1.input = new NeuronValue[] {new NeuronValue1(1), new NeuronValue1(2), new NeuronValue1(3)};
			record1.output = new NeuronValue[] {new NeuronValue1(4), new NeuronValue1(5), new NeuronValue1(6)};

			Record record2 = new Record();
			record2.input = new NeuronValue[] {new NeuronValue1(99), new NeuronValue1(88), new NeuronValue1(77)};
			record2.output = new NeuronValue[] {new NeuronValue1(66), new NeuronValue1(55), new NeuronValue1(44)};

			network.learn(Arrays.asList(record1, record2));
			
			System.out.println(network.toString());
		}
		catch (Exception e) {
			
		}
	}
	
	
}
