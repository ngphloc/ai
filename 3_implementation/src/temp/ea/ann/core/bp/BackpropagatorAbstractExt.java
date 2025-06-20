/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.core.bp;

import java.util.List;
import java.util.Set;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.Network;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.WeightedNeuron;
import net.ea.ann.core.bp.BackpropagatorAbstract;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;
import net.ea.ann.core.value.WeightValue;
import net.ea.ann.core.value.vector.NeuronValueVector;
import net.ea.ann.core.value.vector.NeuronValueVectorImpl;
import net.ea.ann.core.value.vector.WeightValueVector;
import net.ea.ann.core.value.vector.WeightValueVectorImpl;

/**
 * This class is extensive abstract implementation of backpropagation algorithm.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class BackpropagatorAbstractExt extends BackpropagatorAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public BackpropagatorAbstractExt() {
		super();
	}

	
	/**
	 * This enum represents operator.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	private enum Operator {
		add,
		multiply,
	}
	
	
	/**
	 * Taking operator between neuron value and neuron value.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	private NeuronValue operator0(NeuronValue v1, NeuronValue v2, Operator op) {
		NeuronValue result = null;
		switch (op) {
		case add:
			result = v1.add(v2);
			break;
		case multiply:
			result = v1.multiply(v2);
			break;
		}
		
		return result;
	}
	
	
	/**
	 * Taking operator between neuron value and weight value.
	 * @param v neuron value.
	 * @param w weight vector.
	 * @return neuron value.
	 */
	private NeuronValue operator0(NeuronValue v, WeightValue w, Operator op) {
		NeuronValue result = null;
		switch (op) {
		case multiply:
			result = v.multiply(w);
			break;
		default:
			break;
		}
		
		return result;
	}

	
	/**
	 * Taking operator between neuron value and neuron value with regard to vector.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	private NeuronValue operator(NeuronValue v1, NeuronValue v2, Operator op) {
		if ((v1 instanceof NeuronValueVector) && (v2 instanceof NeuronValueVector))
			return operator0(v1, v2, op);
		if (!(v1 instanceof NeuronValueVector) && !(v2 instanceof NeuronValueVector))
			return operator0(v1, v2, op);
		
		NeuronValueVector vector1 = null, vector2 = null;
		if (v1 instanceof NeuronValueVector) {
			vector1 = (NeuronValueVector)v1;
			vector2 = new NeuronValueVectorImpl(vector1.length(), v2);
		}
		else {
			vector1 = (NeuronValueVector)v2;
			vector2 = new NeuronValueVectorImpl(vector1.length(), v1);
		}
		
		NeuronValue result = null;
		switch (op) {
		case add:
			result = vector1.add(vector2);
			break;
		case multiply:
			result = vector1.multiply(vector2);
			break;
		default:
			break;
		}
		return result;
	}

	
	/**
	 * Taking operator between neuron value and weight value with regard to vector.
	 * @param v neuron value.
	 * @param w weight value.
	 * @return neuron value.
	 */
	private NeuronValue operator(NeuronValue v, WeightValue w, Operator op) {
		if ((v instanceof NeuronValueVector) && (w instanceof WeightValueVector))
			return operator0(v, w, op);
		if (!(v instanceof NeuronValueVector) && !(w instanceof WeightValueVector))
			return operator0(v, w, op);
		
		NeuronValueVector valueVector = v instanceof NeuronValueVector ? (NeuronValueVector)v : null;
		WeightValueVector weightVector = w instanceof WeightValueVector ? (WeightValueVector)w : null;
		int dim = valueVector != null ? valueVector.length() : weightVector.length();
		if (valueVector == null) valueVector = new NeuronValueVectorImpl(dim, v);
		if (weightVector == null) weightVector = new WeightValueVectorImpl(dim, w);
		
		NeuronValue result = null;
		switch (op) {
		case multiply:
			result = valueVector.multiply(weightVector);
			break;
		default:
			break;
		}
		return result;
	}

	
	/**
	 * Adding neuron value and neuron value.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	protected NeuronValue add(NeuronValue v1, NeuronValue v2) {
		return operator(v1, v2, Operator.add);
	}


	/**
	 * Multiplying neuron value and neuron value.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	protected NeuronValue multiply(NeuronValue v1, NeuronValue v2) {
		return operator(v1, v2, Operator.multiply);
	}

	
	/**
	 * Multiplying neuron value and weight value.
	 * @param v neuron value.
	 * @param w weight value.
	 * @return neuron value.
	 */
	protected NeuronValue multiply(NeuronValue v, WeightValue w) {
		return operator(v, w, Operator.multiply);
	}

	
	@Override
	public NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		if (bone.size() < 2) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
		NeuronValue[] outputError = null;
		
		NeuronValue[] nextError = lastError;
		for (int i = bone.size()-1; i >= 1; i--) { //Browsing layers reversely from output layer down to first hidden layer.
			LayerStandard layer = bone.get(i);
			NeuronValue[] error = NeuronValue.makeArray(layer.size(), layer);
			
			for (int j = 0; j < layer.size(); j++) { //Browsing neurons of current layer.
				NeuronStandard neuron = layer.get(j);
				
				//Calculate error of current neuron at current layer.
				if (i == bone.size() - 1) {//Calculate error of last layer. This is most important for backpropagation algorithm.
					error[j] = nextError == null ? calcOutputError(layer, j, outputBatch) : nextError[j];
				}
				else {//Calculate error of hidden layers.
					LayerStandard nextLayer = bone.get(i + 1);
					NeuronValue rsum = layer.newBias().zero(); //Fixing for vector.
					WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
					for (WeightedNeuron target : targets) {
						int index = nextLayer.indexOf(target.neuron);
						rsum = rsum.add(multiply(nextError[index], target.weight.value)); //Fixing for vector.
					}
					
					NeuronValue out = neuron.getOutput();
					if (neuron.getActivateRef() != null) {
						NeuronValue derivative = neuron.getActivateRef().derivative(out); //This is the trick to squash the derivative at output as at input.
						error[j] = multiply(rsum, derivative); //Fixing for vector.
					}
					else
						error[j] = rsum;
				}
				
//				error[j] = add(error[j], neuron.getBias().zero()); //Trick to convert into vector.

				//Update biases of current layer.
				if (isLearningBias()) {
					NeuronValue delta = error[j].multiply(learningRate);
					neuron.setBias(add(neuron.getBias(), delta)); //Fixing for vector.
				}
			}
			
			//Update weights stored in previous layers.
			Set<LayerStandard> prevLayers = layer.getAllPrevLayers();
			if (!prevLayers.contains(bone.get(i-1))) prevLayers.add(bone.get(i-1));
			for (LayerStandard prevLayer : prevLayers) {
				if (prevLayer == null) continue;
				for (int j = 0; j < prevLayer.size(); j++) {
					NeuronStandard prevNeuron = prevLayer.get(j);
					NeuronValue prevOut = prevNeuron.getOutput();
					
					WeightedNeuron[] targets = prevNeuron.getNextNeurons(layer);
					for (WeightedNeuron target : targets) {
						Weight nw = target.weight;
						int index = layer.indexOf(target.neuron);
						NeuronValue delta = multiply(error[index], prevOut).multiply(learningRate); //Fixing for vector.
						nw.value = nw.value.add(delta);
					}
				}
			}
			
			nextError = error;
			if (i == bone.size() - 1) outputError = error;
		}
		
		return outputError;
	}


}
