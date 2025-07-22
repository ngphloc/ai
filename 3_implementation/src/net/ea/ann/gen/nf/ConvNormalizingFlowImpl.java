/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.nf;

import java.util.List;
import java.util.Map;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkStandardAssoc;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.NeuronStandardAssoc;
import net.ea.ann.core.WeightedNeuron;
import net.ea.ann.core.bp.BackpropagatorAbstract;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.Mean;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.gen.gan.ConvGANImpl;
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of convolutional normalizing flow (NF) network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvNormalizingFlowImpl extends ConvGANImpl implements ConvNormalizingFlow {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of inverse learning field.
	 */
	public final static String INVERSE_LEARING_FIELD = "nf_inverse_learning";
			
	
	/**
	 * Default value of inverse learning field.
	 */
	public final static boolean INVERSE_LEARING_DEFAULT = true;

	
	/**
	 * Name of bidirectional learning field.
	 */
	public final static String BIDIRECTION_LEARING_FIELD = "nf_bidirection_learning";
			
	
	/**
	 * Default value of bidirectional learning field.
	 */
	public final static boolean BIDIRECTION_LEARING_DEFAULT = false;

	
	/**
	 * This class represents backpropagation algorithm for normalizing flow network.
	 * 
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	protected class NFBackpropagator extends GeneratorStandard.Backpropagator {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Inverse learning mode. 
		 */
		protected boolean inverseLearningMode = true;
		
		/**
		 * Default constructor.
		 */
		public NFBackpropagator() {
			super();
		}

		@Override
		protected boolean isLearningBias() {
			return super.isLearningBias();
		}

		/**
		 * Setting inverse learning mode.
		 * @param inverseLearningMode inverse learning mode.
		 */
		protected void setInverseLearningMode(boolean inverseLearningMode) {
			this.inverseLearningMode = inverseLearningMode;
		}
		
		/**
		 * Checking whether in inverse learning mode.
		 * @return whether in inverse learning mode.
		 */
		public boolean isInverseLearningMode() {
			return inverseLearningMode;
		}
		
		@Override
		public NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
			if (!isInverseLearningMode()) return super.updateWeightsBiases(bone, outputBatch, lastError, learningRate);
			
			if (bone.size() < 2) return null;
			learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
			NeuronValue[] outputError = null;
			
			for (int i = bone.size()-1; i >= 1; i--) { //Browsing layers reversely from output layer down to first hidden layer.
				LayerStandard layer = bone.get(i);
				NeuronValue[] error = NeuronValue.makeArray(layer.size(), layer);
				
				for (int j = 0; j < layer.size(); j++) { //Browsing neurons of current layer.
					NeuronStandard neuron = layer.get(j);
					
					//Calculate error of current neuron at current layer.
					if (i == bone.size() - 1) {//Calculate error of last layer. This is most important for backpropagation algorithm.
						error[j] = lastError == null ? calcOutputError(layer, j, outputBatch) : lastError[j];
					}
					else {//Calculate error of of hidden layers.
						error[j] = calcOutputError(neuron, null, layer, -1, null);
					}
					
					WeightedNeuron[] prevNeurons = new NeuronStandardAssoc(neuron).getPrevNeuronsIncludeOutside();
					if (prevNeurons.length == 0) continue;
					Mean biasDeltaMean = null;
					for (WeightedNeuron prevNeuron : prevNeurons) {
						NeuronValue w = prevNeuron.weight.value.toValue();
						if (!w.canInvert()) continue;
						
						//Updating bias delta.
						NeuronValue w2 = w.multiply(w);
						NeuronValue errorw2 = error[j].divide(w2);
						if (isLearningBias()) {
							if (biasDeltaMean == null)
								biasDeltaMean = new Mean(errorw2);
							else
								biasDeltaMean.accum(errorw2);
						}
						
						//Updating weight.
						NeuronValue weightDelta = errorw2.subtract(w.inverse()).multiply(learningRate);
						prevNeuron.weight.value = prevNeuron.weight.value.addValue(weightDelta);
					}
					
					//Updating bias.
					if (isLearningBias() && biasDeltaMean != null) {
						NeuronValue biasDelta = biasDeltaMean.getMean().multiply(learningRate);
						neuron.setBias(neuron.getBias().add(biasDelta));
					}
				}
				

				if (i == bone.size() - 1) outputError = error;
			}
			
			return outputError;
		}
		
		@Override
		protected NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
			if (!isInverseLearningMode()) return BackpropagatorAbstract.calcOutputErrorDefault(outputNeuron, realOutput, outputLayer);
			
			NeuronValue output = realOutput;
			if (realOutput == null && outputNeuron != null) output = outputNeuron.getOutput();
			if (output == null) return null;
			if ((activateRef == null) || !(activateRef instanceof FunctionInvertible)) return output.zero();
			
			FunctionInvertible af = (FunctionInvertible)activateRef;
			NeuronValue inverseValue = af.evaluateInverse(output);
			NeuronValue inverseDerivative = af.derivativeInverse(output);
			if (inverseValue == null || inverseDerivative == null)
				return output.zero();
			else
				return inverseValue.multiply(inverseDerivative).negative(); //Please pay attention here.
		}

		@Override
		public Map<Integer, NeuronValue[]> updateWeightsBiases(List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
			throw new RuntimeException("NFBackpropagator.updateWeightsBiases(List<LayerStandard>, Map<Integer, NeuronValue[]>, Map<Integer, NeuronValue[]>, double learningRate) not implemented yet");
		}

	}
	

	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel which is often larger than or equal to neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected ConvNormalizingFlowImpl(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, rasterChannel, size, idRef);
		this.config.put(INVERSE_LEARING_FIELD, INVERSE_LEARING_DEFAULT);
		this.config.put(BIDIRECTION_LEARING_FIELD, BIDIRECTION_LEARING_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected ConvNormalizingFlowImpl(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	protected ConvNormalizingFlowImpl(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected ConvNormalizingFlowImpl(int neuronChannel) {
		this(neuronChannel, neuronChannel, Size.unit(), null);
	}


	@Override
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronDecode, int[] nHiddenNeuronAdversarial) {
		if (xDim <= 0 || zDim <= 0) return false;
		
		this.decoder = createDecoder();
		if(!this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode)) return false;
		
		//This code line will be improved in the next version.
		if(isInverseLearning() && !config.getAsBoolean(BIDIRECTION_LEARING_FIELD))
			new NetworkStandardAssoc(this.decoder).setWeights(1); //Preventing divide-by-zero when inverse learning.
		return true;
	}

	
	@Override
	protected NetworkStandardImpl createDecoder() {
		GeneratorStandard<?> generator = new GeneratorStandard<Trainer>(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected Backpropagator createBackpropagator() {
				NFBackpropagator bp = new NFBackpropagator();
				bp.setNetwork(this);
				return bp;
			}

			/*
			 * This method will be improved in the next version.
			 */
			@Override
			public synchronized NeuronValue[] learn(NeuronValue[] input, NeuronValue[] realOutput, double learningRate, double terminatedThreshold, int maxIteration) {
				NFBackpropagator nfbp = (NFBackpropagator)bp;
				boolean inverseLearning = nfbp.isInverseLearningMode();
				
				NeuronValue[] error = null;
				if (!isInverseLearning()) {
					nfbp.setInverseLearningMode(false);
					error = super.learn(input, realOutput, learningRate, terminatedThreshold, maxIteration);
				}
				else {
					if (config.getAsBoolean(BIDIRECTION_LEARING_FIELD)) {
						nfbp.setInverseLearningMode(false);
						error = super.learn(input, realOutput, learningRate, terminatedThreshold, maxIteration);
					}
					nfbp.setInverseLearningMode(true);
					error = super.learn(input, realOutput, learningRate, terminatedThreshold, maxIteration);
				}
				
				nfbp.setInverseLearningMode(inverseLearning);
				return error;
			}

		};
		generator.setParent(this);
		return generator;
	}


	/**
	 * Checking whether inverse learning.
	 * @return whether inverse learning.
	 */
	private boolean isInverseLearning() {
		return config.getAsBoolean(INVERSE_LEARING_FIELD);
	}
	
	
	/**
	 * Creating with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return convolutional normalizing flow network network (convolutional NF network).
	 */
	public static ConvNormalizingFlowImpl create(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		size.width = size.width < 1 ? 1 : size.width;
		size.height = size.height < 1 ? 1 : size.height;
		size.depth = size.depth < 1 ? 1 : size.depth;
		size.time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		rasterChannel = rasterChannel < neuronChannel ? neuronChannel : rasterChannel;
		return new ConvNormalizingFlowImpl(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Creating NF with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return convolutional normalizing flow network network (convolutional NF network).
	 */
	public static ConvNormalizingFlowImpl create(int neuronChannel, Size size, Id idRef) {
		return create(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Creating NF with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size raster size.
	 * @return convolutional normalizing flow network network (convolutional NF network).
	 */
	public static ConvNormalizingFlowImpl create(int neuronChannel, Size size) {
		return create(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Creating NF with neuron channel and raster channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return convolutional normalizing flow network network (convolutional NF network).
	 */
	public static ConvNormalizingFlowImpl create(int neuronChannel, int rasterChannel) {
		return create(neuronChannel, rasterChannel, new Size(1, 1, 1, 1), null);
	}

	
	/**
	 * Creating NF with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return convolutional normalizing flow network network (convolutional NF network).
	 */
	public static ConvNormalizingFlowImpl create(int neuronChannel) {
		return create(neuronChannel, neuronChannel, new Size(1, 1, 1, 1), null);
	}

	
}
