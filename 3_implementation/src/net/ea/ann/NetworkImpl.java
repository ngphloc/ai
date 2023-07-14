/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import net.ea.ann.NetworkDoEvent.Type;

/**
 * This class is default implementation of neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NetworkImpl extends NetworkAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Constructor with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenLayer number of hidden layers.
	 * @param nHiddenNeuron number of hidden neurons.
	 * @param nMemoryNeuron number of memory neurons.
	 */
	public NetworkImpl(int nInputNeuron, int nOutputNeuron, int nHiddenLayer, int nHiddenNeuron, int nMemoryNeuron) {
		super(nInputNeuron, nOutputNeuron, nHiddenLayer, nHiddenNeuron, nMemoryNeuron);
	}

	
	/**
	 * Constructor with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenLayer number of hidden layers.
	 * @param nHiddenNeuron number of hidden neurons.
	 */
	public NetworkImpl(int nInputNeuron, int nOutputNeuron, int nHiddenLayer, int nHiddenNeuron) {
		super(nInputNeuron, nOutputNeuron, nHiddenLayer, nHiddenNeuron);
	}

	
	/**
	 * Constructor with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 */
	public NetworkImpl(int nInputNeuron, int nOutputNeuron) {
		super(nInputNeuron, nOutputNeuron);
	}


	@Override
	protected Function newFunction() {
		return new LogisticFunctionScalar();
	}
	

	@Override
	protected Layer newLayer() {
		activateRef = activateRef != null? activateRef : newFunction();
		return new LayerImpl(activateRef, idRef);
	}
	
	
	@Override
	public synchronized Value[] learn(Iterable<Record> sample) throws RemoteException {
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
	protected Value[] bpLearn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		List<Layer> backbone = getBackbone();
		if (backbone.size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Value[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			List<Value[]> errors = Util.newList(0);
			for (Record record : sample) {
				if (record == null || record.input == null || record.output == null) continue;
				Value[] output = Value.adjustArray(record.output, backbone.get(backbone.size()-1).size(), backbone.get(backbone.size()-1));
				
				List<List<Layer>> ribinbones = getRibinbones();
				List<List<Layer>> riboutbones = getRiboutbones();
				Map<Integer, List<Value[]>> ribinErrorMap = Util.newMap(0);
				Map<Integer, List<Value[]>> riboutErrorMap = Util.newMap(0);

				//Evaluating layers.
				try {
					eval(record, true);
				} catch (Throwable e) {Util.trace(e);}
				
				//Calculating errors.
				errors = bpCalcErrors(backbone, output);
				
				for (List<Layer> ribinbone : ribinbones) {
					Layer layer = ribinbone.get(ribinbone.size() - 1);
					Value[] ribOutput = Value.makeArray(layer.size(), layer);
					for (int j = 0; j < ribOutput.length; j++)
						ribOutput[j] = layer.get(j).getOutput();
					
					List<Value[]> ribinErrors = bpCalcErrors(ribinbone, ribOutput);
					if (ribinErrors != null && ribinErrors.size() > 0)
						ribinErrorMap.put(layer.id(), ribinErrors);
				}
				
				for (List<Layer> riboutbone : riboutbones) {
					if (riboutbone.size() < 2) continue;
					int id = riboutbone.get(riboutbone.size() - 1).id();
					if (!record.ribOutput.containsKey(id)) continue;
					
					List<Value[]> riboutErrors = bpCalcErrors(riboutbone, record.ribOutput.get(id));
					if (riboutErrors != null && riboutErrors.size() > 0)
						riboutErrorMap.put(id, riboutErrors);
				}
				
				
				//Updating weights and biases.
				bpUpdateWeightsBiases(backbone, errors, learningRate);
				
				for (List<Layer> ribinbone : ribinbones) {
					int id = ribinbone.get(ribinbone.size() - 1).id();
					if (ribinErrorMap.containsKey(id))
						bpUpdateWeightsBiases(ribinbone, ribinErrorMap.get(id), learningRate);
				}

				for (List<Layer> riboutbone : riboutbones) {
					int id = riboutbone.get(0).id();
					if (riboutErrorMap.containsKey(id))
						bpUpdateWeightsBiases(riboutbone, riboutErrorMap.get(id), learningRate);
				}

				
				//Updating weights and biases related to memory layer.
				if (memoryLayer != null && memoryLayer.size() > 0)
					bpUpdateWeightsBiasesAttachedTriple(memoryLayer, backbone, errors, learningRate);
				
				
				error = (errors != null && errors.size() > 0) ? errors.get(errors.size()-1) : null;
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ann_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			
			if (error == null || error.length == 0)
				doStarted = false;
			else {
				double errorMean = 0;
				for (Value r : error) errorMean += Math.abs(r.norm());
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
	 * @param output output values.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected static Value[] bpLearn(List<Layer> bone, Value[] input, Value[] output, double learningRate, double terminatedThreshold, int maxIteration) {
		if (bone == null || bone.size() < 2) return null;
		output = Value.adjustArray(output, bone.get(bone.size()-1).size(), bone.get(bone.size()-1));
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		Value[] error = null;
		int iteration = 0;
		while (maxIteration <= 0 || iteration < maxIteration) {
			List<Value[]> errors = Util.newList(0);

			//Evaluating layers.
			eval(bone, input);
			
			//Calculating errors.
			errors = bpCalcErrors(bone, output);
			
			//Updating weights and biases.
			bpUpdateWeightsBiases(bone, errors, learningRate);
			
			error = (errors != null && errors.size() > 0) ? errors.get(errors.size()-1) : null;
			
			iteration ++;
			
			if (error == null || error.length == 0)
				break;
			else {
				double errorMean = 0;
				for (Value r : error) errorMean += Math.abs(r.norm());
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) break; 
			}
			
		}
		
		return error;
	}
	
	
	/**
	 * Calculating errors.
	 * @param bone list of layers including input layer.
	 * @param errors list of errors. Error list excludes input error and so it is 1 less than backbone. 
	 * @return list of errors.
	 */
	private static List<Value[]> bpCalcErrors(List<Layer> bone, Value[] output) {
		List<Value[]> errors = Util.newList(0);
		if (bone.size() < 2) return errors;
		
		for (int i = bone.size() - 1; i >= 1; i--) {
			Layer layer = bone.get(i);
			Layer nextLayer = i < bone.size() - 1 ? bone.get(i + 1) : null;
			Value[] error = Value.makeArray(layer.size(), layer);
			errors.add(0, error);
			
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				Value out = neuron.getOutput();
				Value derivative = neuron.getActivateRef().derivative(out);
				
				if (i == bone.size() - 1)
					error[j] = output[j].subtract(out).multiply(derivative);
				else {
					Value rsum = layer.newValue();
					Value[] nextError = errors.get(1);
					WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
					for (WeightedNeuron target : targets) {
						int index = nextLayer.indexOf(target.neuron);
						rsum = rsum.add(nextError[index].multiply(target.weight.value));
					}
					error[j] = rsum.multiply(derivative);
				}
			}
		}
		
		return errors;
	}
	
	
	/**
	 * Updating weights and biases.
	 * @param bone list of layers including input layer.
	 * @param errors list of errors. Error list excludes input error and so it is 1 less than backbone. 
	 * @param learningRate learning rate.
	 */
	private static void bpUpdateWeightsBiases(List<Layer> bone, List<Value[]> errors, double learningRate) {
		if (bone.size() < 2) return;
		
		for (int i = 0; i < bone.size() - 1; i++) {
			Layer layer = bone.get(i);
			Layer nextLayer = bone.get(i + 1);
			Value[] error = i > 0 ? errors.get(i - 1) : null;
			Value[] nextError = errors.get(i);
			
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				Value out = neuron.getOutput();
				
				WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
				for (WeightedNeuron target : targets) {
					Weight nw = target.weight;
					int index = nextLayer.indexOf(target.neuron);
					Value delta = nextError[index].multiply(out).multiply(learningRate);
					nw.value = nw.value.add(delta);
				}
				
				if (i > 0) {
					Value delta = error[j].multiply(learningRate);
					neuron.setBias(neuron.getBias().add(delta));
				}
			}
			
			if (i == bone.size() - 1) {
				for (int j = 0; j < nextLayer.size(); j++) {
					Neuron neuron = nextLayer.get(j);
					Value delta = nextError[j].multiply(learningRate);
					neuron.setBias(neuron.getBias().add(delta));
				}
			}
		}
	}
	
	
	/**
	 * Update weights and biases of the triple including center layer.
	 * @param centerLayer specified center layer.
	 * @param bone list of layers including input layer.
	 * @param errors list of errors. Error list excludes input error and so it is 1 less than bone. 
	 * @param learningRate learning rate.
	 */
	private static void bpUpdateWeightsBiasesAttachedTriple(Layer centerLayer, List<Layer> bone, List<Value[]> errors, double learningRate) {
		if (bone.size() < 2) return;

		Layer prevLayer = centerLayer.getPrevLayer();
		if (prevLayer == null) return;
		Layer nextLayer = centerLayer.getNextLayer();
		if (nextLayer == null) return;
		
		int nextErrorIndex = findLayer(bone, nextLayer) - 1;
		if (nextErrorIndex < 0) return;
		
		//Evaluating center neurons.
		for (int j = 0; j < centerLayer.size(); j++) centerLayer.get(j).eval();
		
		//Updating errors of center layer.
		Value[] centerError = Value.makeArray(centerLayer.size(), centerLayer);
		Value[] nextError = errors.get(nextErrorIndex);
		for (int j = 0; j < centerLayer.size(); j++) {
			Neuron centerNeuron = centerLayer.get(j);
			Value out = centerNeuron.getOutput();
			Value derivative = centerNeuron.getActivateRef().derivative(out);

			Value rsum = centerLayer.newValue();;
			WeightedNeuron[] targets = centerNeuron.getNextNeurons(nextLayer);
			for (WeightedNeuron target : targets) {
				int index = nextLayer.indexOf(target.neuron);
				rsum = rsum.add(nextError[index].multiply(target.weight.value));
			}
			centerError[j] = rsum.multiply(derivative);
		}
		
		List<Layer> newBackbone = Util.newList(3);
		newBackbone.add(prevLayer);
		newBackbone.add(centerLayer);
		newBackbone.add(nextLayer);
		List<Value[]> newErrors = Util.newList(2);
		newErrors.add(centerError);
		newErrors.add(nextError);
		
		bpUpdateWeightsBiases(newBackbone, newErrors, learningRate);
	}


	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		try (NetworkImpl network = new NetworkImpl(3, 3, 1, 3)) {
			System.out.println(network.toString());
			
			Record record = new Record();
			record.input = new Value[] {new ValueScalar(1), new ValueScalar(2), new ValueScalar(3)};
			record.output = new Value[] {new ValueScalar(4), new ValueScalar(5), new ValueScalar(6)};
			network.learn(Arrays.asList(record));
			
			System.out.println(network.toString());
		}
		catch (Exception e) {
			
		}
	}
	
	
}
