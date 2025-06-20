/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.generator;

import java.util.List;
import java.util.Map;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.LayerStandardImpl;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.NetworkStandardAbstract;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.NeuronStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.bp.BackpropagatorAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Mean;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.core.value.Weight;

/**
 * This class represents standard generator as standard neural network.
 * 
 * @author Loc Nguyen
 * @param <T> type of trainer.
 * @version 1.0
 *
 */
public class GeneratorStandard<T extends Trainer> extends NetworkStandardImpl implements Generator {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of updating mean field.
	 */
	public static final String ERROR_DYNAMIC_MEAN_FIELD = "gs_error_dynamic_mean";
	
	
	/**
	 * Default value of updating mean field.
	 */
	public static final boolean ERROR_DYNAMIC_MEAN_DEFAULT = false;

	
	/**
	 * Name of updating variance field.
	 */
	public static final String ERROR_DYNAMIC_VARIANCE_FIELD = "gs_error_dynamic_var";
	
	
	/**
	 * Default value of updating variance field.
	 */
	public static final boolean ERROR_DYNAMIC_VARIANCE_DEFAULT = false;

	
	/**
	 * Name of prior mean-variance field.
	 */
	public static final String ERROR_PRIOR_MEAN_VARIANCE_FIELD = "gs_error_prior_meanvar";
	
	
	/**
	 * Default value of prior mean-variance field.
	 */
	public static final boolean ERROR_PRIOR_MEAN_VARIANCE_DEFAULT = false;

	
	/**
	 * This class represents neuron for generator.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class Neuron extends NeuronStandardImpl {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Accumulated error mean.
		 */
		protected Mean accumErrorMean = null;
		
		/**
		 * Accumulated error variance.
		 */
		protected Mean accumErrorVariance = null;
		
		/**
		 * Previous input.
		 */
		protected NeuronValue prevInput = null;
		
		/**
		 * Constructor with standard layer.
		 * @param layer layer.
		 */
		public Neuron(LayerStandard layer) {
			super(layer);
			NeuronValue zero = layer.newNeuronValue().zero();
			setAccumErrorMean(new Mean(zero));
			setAccumErrorVariance(new Mean(zero.unit()));
			this.prevInput = null;
		}
		
		/**
		 * Getting generator layer.
		 * @return generator layer.
		 */
		private Layer getGeneratorLayer() {
			LayerStandard layer = getLayer();
			return layer != null && layer instanceof Layer ? (Layer)layer : null;
		}
		
		/**
		 * Getting the second activation function of neuron.
		 * @return the second activation function of neuron.
		 */
		public Function getActivateRef2() {
			return getActivateRef2(this);
		}
		
		/**
		 * Getting the second activation function of neuron.
		 * @param neuron specified neuron.
		 * @return the second activation function of neuron.
		 */
		public static Function getActivateRef2(NeuronStandard neuron) {
			if (neuron == null) return null;
			LayerStandard layer = neuron.getLayer();
			return Layer.getActivateRef2(layer);
		}
		
		/**
		 * Getting auxiliary activation function which is often ramp function like ReLU.
		 * @return auxiliary activation function which is often ramp function like ReLU.
		 */
		public Function getAuxActivateRef() {
			Layer layer = getGeneratorLayer();
			return layer != null ? layer.getAuxActivateRef() : null;
		}

		/**
		 * Getting error mean.
		 * @return error mean.
		 */
		public NeuronValue getErrorMean() {
			return accumErrorMean.getMean();
		}

		/**
		 * Getting error variance.
		 * @return error variance.
		 */
		public NeuronValue getErrorVariance() {
			return accumErrorVariance.getMean();
		}


		/**
		 * Getting accumulated error mean.
		 * @return accumulated error mean.
		 */
		protected Mean getAccumErrorMean() {
			return accumErrorMean;
		}
		
		/**
		 * Setting accumulated error mean.
		 * @param accumErrorMean accumulated error mean.
		 */
		protected void setAccumErrorMean(Mean accumErrorMean) {
			this.accumErrorMean = accumErrorMean;
		}
		
		/**
		 * Resetting accumulated error mean.
		 * @param value specified value.
		 */
		public void resetAccumErrorMean(NeuronValue value) {
			accumErrorMean.reset(value);
		}
		
		/**
		 * Getting accumulated error variance.
		 * @return accumulated error variance.
		 */
		protected Mean getAccumErrorVariance() {
			return accumErrorVariance;
		}
		
		/**
		 * Setting accumulated error variance.
		 * @param accumErrorVariance accumulated error variance.
		 */
		protected void setAccumErrorVariance(Mean accumErrorVariance) {
			this.accumErrorVariance = accumErrorVariance;
		}

		/**
		 * Resetting accumulated error variance.
		 * @param value specified value.
		 */
		public void resetAccumErrorVariance(NeuronValue value) {
			accumErrorVariance.reset(value);
		}

		@Override
		public NeuronValue evaluate() {
			prevInput = null;
			return super.evaluate();
		}

		/**
		 * Calculating derivative of neuron.
		 * @param neuron specified neuron.
		 * @return derivative of neuron.
		 */
		public static NeuronValue derivative(NeuronStandard neuron) {
			if (neuron == null) return null;
			Function activateRef2 = getActivateRef2(neuron);
			if (activateRef2 == null || !(neuron instanceof Neuron)) return neuron.derivative();
			
			NeuronValue input = neuron.getInput();
			NeuronValue prevInput = ((Neuron)neuron).prevInput;
			if (input == null || prevInput == null) return neuron.derivative();

			NeuronValue derivative = input.derivative(activateRef2);
			Function activateRef = neuron.getActivateRef();
			if (activateRef == null) return derivative;
			NeuronValue d = prevInput.derivative(activateRef);
			return derivative.multiply(d);
		}

	}
	
	
	/**
	 * This class represents generator layer.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class Layer extends LayerStandardImpl {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Parent network.
		 */
		protected NetworkStandardAbstract network = null;
		
		/**
		 * Auxiliary activation reference which is often ramp function like ReLU.
		 */
		private Function auxActivateRef = null;
		
		/**
		 * Second activation reference.
		 */
		protected Function activateRef2 = null;
		
		/**
		 * Constructor with neuron channel, activation function, and identifier reference.
		 * @param neuronChannel neuron channel.
		 * @param activateRef activation function.
		 * @param idRef identifier reference.
		 */
		public Layer(int neuronChannel, Function activateRef, Id idRef) {
			super(neuronChannel, activateRef, idRef);
		}

		/**
		 * Constructor with neuron channel and activation function.
		 * @param neuronChannel neuron channel.
		 * @param activateRef activation function.
		 */
		public Layer(int neuronChannel, Function activateRef) {
			this(neuronChannel, activateRef, null);
		}

		/**
		 * Constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		public Layer(int neuronChannel) {
			this(neuronChannel, null, null);
		}
		
		@Override
		public NetworkStandardAbstract getNetwork() {
			return network;
		}

		/**
		 * Setting network.
		 * @param network network.
		 */
		protected void setNetwork(NetworkStandardAbstract network) {
			this.network = network;
		}
		
		/**
		 * Getting generator.
		 * @return generator.
		 */
		private GeneratorStandard<?> getGenerator() {
			return network != null && network instanceof GeneratorStandard ? (GeneratorStandard<?>)network : null;
		}
		
		/**
		 * Getting second activation reference.
		 * @return second activation reference.
		 */
		public Function getActivateRef2() {
			return activateRef2;
		}
		
		/**
		 * Getting second activation reference of specified layer.
		 * @param layer specified layer.
		 * @return second activation reference of specified layer.
		 */
		public static Function getActivateRef2(LayerStandard layer) {
			return (layer != null && layer instanceof Layer) ? ((Layer)layer).getActivateRef2() : null;
		}
		
		/**
		 * Setting second activation reference.
		 * @param activateRef2 activation reference.
		 */
		public void setActivateRef2(Function activateRef2) {
			this.activateRef2 = activateRef2;
		}
		
		/**
		 * Getting auxiliary activation function which is often ramp function like ReLU.
		 * @return auxiliary activation function which is often ramp function like ReLU.
		 */
		public Function getAuxActivateRef() {
			if (auxActivateRef != null) return auxActivateRef;
			GeneratorStandard<?> generator = getGenerator();
			return generator != null ? generator.getAuxActivateRef() : null;
		}
		
		/**
		 * Resetting error means and error variances of all neurons.
		 * @param mean mean.
		 * @param variance variance.
		 */
		public void resetErrorMeansVariances(double mean, double variance) {
			NeuronValue nmean = newNeuronValue().valueOf(mean);
			NeuronValue nvariance = nmean.valueOf(variance);
			for (int i = 0; i < size(); i++) {
				NeuronStandard neuron = get(i);
				if (!(neuron instanceof Neuron)) continue;
				Neuron gn = (Neuron)neuron;
				gn.resetAccumErrorMean(nmean);
				gn.resetAccumErrorVariance(nvariance);
			}
		}

		/**
		 * Resetting error means and error variances of all neurons.
		 */
		public void resetErrorMeansVariances( ) {
			resetErrorMeansVariances(0, 1);
		}

		@Override
		public NeuronValue newNeuronValue() {
			GeneratorStandard<?> generator = getGenerator();
			return generator != null ? generator.newNeuronValue(this) : super.newNeuronValue();
		}
		
		/**
		 * Creating an empty neuron value.
		 * @return empty neuron value.
		 */
		protected NeuronValue newNeuronValueCaller() {
			return super.newNeuronValue();
		}

		@Override
		public Weight newWeight() {
			GeneratorStandard<?> generator = getGenerator();
			return generator != null ? generator.newWeight(this) : super.newWeight();
		}

		/**
		 * Create a new weight.
		 * @return new weight.
		 */
		protected Weight newWeightCaller() {
			return super.newWeight();
		}

		@Override
		public NeuronValue newBias() {
			GeneratorStandard<?> generator = getGenerator();
			return generator != null ? generator.newBias(this) : super.newBias();
		}
		
		/**
		 * Create bias.
		 * @return created bias.
		 */
		protected NeuronValue newBiasCaller() {
			return super.newBias();
		}

		@Override
		public NeuronStandard newNeuron() {
			GeneratorStandard<?> generator = getGenerator();
			return generator != null ? generator.newNeuron(this) : new Neuron(this);
		}
		
		/**
		 * Create neuron.
		 * @return created neuron.
		 */
		protected NeuronStandard newNeuronCaller() {
			return new Neuron(this);
		}

		@Override
		protected void postEvaluate() {
			GeneratorStandard<?> generator = getGenerator();
			if(generator != null) 
				generator.postEvaluate(this);
			else
				postEvaluateCaller();
		}

		/**
		 * Post evaluation.
		 */
		protected void postEvaluateCaller() {
			super.postEvaluate();
			Function activateRef2 = getActivateRef2();
			if (activateRef2 == null) return;
			
			for (int i = 0; i < size(); i++) {
				NeuronStandard neuron = get(i);
				if (neuron instanceof Neuron) ((Neuron)neuron).prevInput = neuron.getInput();
			}
			postEvaluate(this, activateRef2);
		}

	}
	
	
	/**
	 * This class represents backpropagation algorithm for recurrent neural network.
	 * 
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class Backpropagator extends BackpropagatorAbstract {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Internal network.
		 */
		protected NetworkStandardAbstract network = null;
		
		/**
		 * Default constructor.
		 */
		public Backpropagator() {
			super();
		}

		/**
		 * Getting network.
		 * @return network.
		 */
		public NetworkStandardAbstract getNetwork() {
			return network;
		}
		
		/**
		 * Setting network.
		 * @param network network.
		 */
		public void setNetwork(NetworkStandardAbstract network) {
			this.network = network;
		}
		
		/**
		 * Getting generator.
		 * @return generator.
		 */
		private GeneratorStandard<?> getGenerator() {
			return network != null && network instanceof GeneratorStandard ? (GeneratorStandard<?>)network : null;
		}
		
		/**
		 * Updating weights and biases. Derived class can override this method.
		 * @param bone list of layers including input layer.
		 * @param outputBatch output batch of output layer.
		 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
		 * because the method {@link NeuronStandard#getOutput()} returns the neuron output too. 
		 * @param lastError last error which is optional parameter for batch learning.
		 * @param learningRate learning rate.
		 * @return errors of output errors. Return null if errors occur.
		 */
		private NeuronValue[] updateWeightsBiasesCaller(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
			return super.updateWeightsBiases(bone, outputBatch, lastError, learningRate);
		}

		@Override
		public NeuronValue[] updateWeightsBiases(List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
			GeneratorStandard<?> generator = getGenerator();
			if (generator != null)
				return generator.updateWeightsBiases(this, bone, outputBatch, lastError, learningRate);
			else
				return super.updateWeightsBiases(bone, outputBatch, lastError, learningRate);
		}

		/**
		 * Updating weights and biases for every layer. Derived class can override this method.
		 * @param bone list of layers including input layer.
		 * @param boneInput bone input. Note, index of layer is ID.
		 * @param boneOutput bone output. Note, index of layer is ID.
		 * @param learningRate learning rate.
		 * @return errors of output errors.
		 */
		private Map<Integer, NeuronValue[]> updateWeightsBiasesCaller(List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
			return super.updateWeightsBiases(bone, boneInput, boneOutput, learningRate);
		}
		
		@Override
		public Map<Integer, NeuronValue[]> updateWeightsBiases(List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
			GeneratorStandard<?> generator = getGenerator();
			if (generator != null)
				return generator.updateWeightsBiases(this, bone, boneInput, boneOutput, learningRate);
			else
				return super.updateWeightsBiases(bone, boneInput, boneOutput, learningRate);
		}

		@Override
		protected NeuronValue calcDerivative(NeuronStandard neuron) {
			return Neuron.derivative(neuron);
		}

		@Override
		protected NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
			GeneratorStandard<?> generator = getGenerator();
			return generator.calcOutputError(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		}

	}


	/**
	 * Internal trainer.
	 */
	protected T trainer = null;
	
	
	/**
	 * Parent network.
	 */
	protected Network parent = null;
	
	
	/**
	 * Auxiliary activation function which is often ramp function like ReLU.
	 */
	protected Function auxActivateRef = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public GeneratorStandard(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
		fillConfig(config);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public GeneratorStandard(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public GeneratorStandard(int neuronChannel) {
		this(neuronChannel, null, null);
	}


	/**
	 * Getting auxiliary activation function which is often ramp function like ReLU.
	 * @return auxiliary activation function which is often ramp function like ReLU.
	 */
	public Function getAuxActivateRef() {
		return auxActivateRef;
	}
	
	
	/**
	 * Setting auxiliary activation function which is often ramp function like ReLU.
	 * @param auxActivateRef auxiliary activation function which is often ramp function like ReLU.
	 */
	public void setAuxActivateRef(Function auxActivateRef) {
		this.auxActivateRef = auxActivateRef;
	}
	
	
	/**
	 * Getting activation function of output layer.
	 * @return activation function of output layer.
	 */
	public Function getOutputLayerActivateRef() {
		LayerStandard outputLayer = getOutputLayer();
		return outputLayer != null ? outputLayer.getActivateRef() : null;
	}
	
	
	/**
	 * Setting  activation function of output layer.
	 * @param activateRef activation function.
	 */
	public void changeOutputLayerActivateRef(Function activateRef) {
		LayerStandard outputLayer = getOutputLayer();
		if (outputLayer == null) return;
		outputLayer.setActivateRef(activateRef);
		for (int i = 0; i < outputLayer.size(); i++) outputLayer.get(i).setActivateRef(activateRef);
	}

	
	/**
	 * Getting the second activation function of output layer.
	 * @return the second activation function of output layer.
	 */
	public Function getOutputLayerActivateRef2() {
		LayerStandard outputLayer = getOutputLayer();
		return Layer.getActivateRef2(outputLayer);
	}
	
	
	/**
	 * Setting the second activation function of output layer.
	 * @param activateRef2 the second activation function.
	 * @return true if setting is successful.
	 */
	public boolean setOutputLayerActivateRef2(Function activateRef2) {
		LayerStandard outputLayer = getOutputLayer();
		if ((outputLayer != null) && (outputLayer instanceof Layer)) {
			((Layer)outputLayer).setActivateRef2(activateRef2);
			return true;
		}
		else
			return false;
	}

	
	/**
	 * Getting activation reference of output layer.
	 * @return activation reference of output layer.
	 */
	public Function getOutputLayerActivateRefOutermost() {
		LayerStandard outputLayer = getOutputLayer();
		Function activateRef = getActivateRefOutermost(null, outputLayer);
		if (activateRef != null) return activateRef;
		
		if (outputLayer.size() > 0) activateRef = getActivateRefOutermost(outputLayer.get(0), outputLayer);
		return activateRef != null ? activateRef : this.activateRef;
	}

	
	/**
	 * Creating an empty neuron value.
	 * @param layer specified layer.
	 * @return empty neuron value.
	 */
	protected NeuronValue newNeuronValue(LayerStandard layer) {
		return (layer != null && layer instanceof Layer) ? ((Layer)layer).newNeuronValueCaller() : NeuronValueCreator.newNeuronValue(neuronChannel);
	}

	
	/**
	 * Create a new weight.
	 * @param layer specified layer.
	 * @return new weight.
	 */
	protected Weight newWeight(LayerStandard layer) {
		return (layer != null && layer instanceof Layer) ? ((Layer)layer).newWeightCaller() : new Weight(newNeuronValue(layer).newWeightValue().zero());
	}

	
	/**
	 * Create a new bias.
	 * @param layer specified layer.
	 * @return new bias.
	 */
	protected NeuronValue newBias(LayerStandard layer) {
		return (layer != null && layer instanceof Layer) ? ((Layer)layer).newBiasCaller() : newNeuronValue(layer).zero();
	}

	
	/**
	 * Create neuron.
	 * @param layer specified layer.
	 * @return created neuron
	 */
	protected NeuronStandard newNeuron(LayerStandard layer) {
		return (layer != null && layer instanceof Layer) ? ((Layer)layer).newNeuronCaller() : new Neuron(layer);
	}

	
	/**
	 * Post evaluation.
	 * @param layer specified layer.
	 */
	protected void postEvaluate(LayerStandard layer) {
		if (layer != null && layer instanceof Layer) ((Layer)layer).postEvaluateCaller();
	}

	
	@Override
	protected LayerStandard newLayer() {
		Layer layer = new Layer(neuronChannel, activateRef, idRef);
		layer.setNetwork(this);
		return layer;
	}


	/**
	 * Setting trainer.
	 * @param trainer specified trainer. It can be null to remove trainer from generator.
	 * @return this generator.
	 */
	public GeneratorStandard<T> setTrainer(T trainer) {
		this.trainer = trainer;
		if (trainer != null) trainer.setGenerator(this);
		return this;
	}


	@Override
	protected Backpropagator createBackpropagator() {
		Backpropagator bp = new Backpropagator();
		bp.setNetwork(this);
		return bp;
	}


	@Override
	public NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		if (trainer != null)
			return trainer.learnOne(sample, learningRate, terminatedThreshold, maxIteration);
		else
			return super.learnOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		if (trainer != null)
			return trainer.learn(sample, learningRate, terminatedThreshold, maxIteration);
		else
			return super.learn(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Updating weights and biases. Derived class can override this method.
	 * @param bp backpropagator agorithm.
	 * @param bone list of layers including input layer.
	 * @param outputBatch output batch of output layer.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is neuron output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the neuron output too. 
	 * @param lastError last error which is optional parameter for batch learning.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	protected NeuronValue[] updateWeightsBiases(Backpropagator bp, List<LayerStandard> bone, Iterable<NeuronValue[][]> outputBatch, NeuronValue[] lastError, double learningRate) {
		return bp.updateWeightsBiasesCaller(bone, outputBatch, lastError, learningRate);
	}

	
	/**
	 * Updating weights and biases for every layer. Derived class can override this method.
	 * @param bp backpropagator agorithm.
	 * @param bone list of layers including input layer.
	 * @param boneInput bone input. Note, index of layer is ID.
	 * @param boneOutput bone output. Note, index of layer is ID.
	 * @param learningRate learning rate.
	 * @return errors of output errors.
	 */
	protected Map<Integer, NeuronValue[]> updateWeightsBiases(Backpropagator bp, List<LayerStandard> bone, Map<Integer, NeuronValue[]> boneInput, Map<Integer, NeuronValue[]> boneOutput, double learningRate) {
		return bp.updateWeightsBiasesCaller(bone, boneInput, boneOutput, learningRate);
	}
	
	
	/**
	 * Adjusting error.
	 * @param error error.
	 * @param outputNeuron output neuron.
	 * @param outputLayer output layer.
	 * @return adjusted error.
	 */
	protected NeuronValue adjustError(NeuronValue error, NeuronStandard outputNeuron, LayerStandard outputLayer) {
		if (error == null || outputNeuron == null) return error;
		if (outputLayer == null && Neuron.getActivateRef2(outputNeuron) == null) return error;
		if (Layer.getActivateRef2(outputLayer) == null && Neuron.getActivateRef2(outputNeuron) == null) return error;
		Function activateRef = outputNeuron.getActivateRef();
		if (activateRef == null) return error;

		NeuronValue input = outputNeuron instanceof Neuron ? ((Neuron)outputNeuron).prevInput : null;
		if (input == null) return error;
		NeuronValue d = input.derivative(activateRef);
		return d != null ? error.multiply(d) : error;
	}
	
	
	@Override
	protected NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		NeuronValue error = calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
		return adjustError(error, outputNeuron, outputLayer);
	}

	
	/**
	 * Calculate error of output neuron for the second activation function. Derived classes should implement this method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output.  
	 * @param outputNeuron output neuron. Output value of this neuron is retrieved by method {@link NeuronStandard#getOutput()}.
	 * @param realOutput real output. It can be null because this method is flexible.
	 * @param outputLayer output layer. It can be null because this method is flexible.
	 * @param outputNeuronIndex index of output neuron. It can be -1 because this method is flexible. This is optional parameter.
	 * @param realOutputs real outputs. It can be null because this method is flexible. This is optional parameter. 
	 * @param params option parameter array which can be null. 
	 * @return error or loss of the output neuron.
	 */
	protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
		if (outputLayer == null || outputLayer != getOutputLayer())
			return calcOutputErrorDefault(outputNeuron, realOutput, outputLayer);
		if (!isErrorPriorMeanVariance() && !isErrorDynamicMean() && !isErrorDynamicVariance())
			return calcOutputErrorDefault(outputNeuron, realOutput, outputLayer);
		
		Function activateRef = getActivateRefOutermost(outputNeuron, outputLayer);
		if (activateRef == null && outputLayer != null) activateRef = outputLayer.getActivateRef();
		NeuronValue neuronOutput = outputNeuron != null ? outputNeuron.getOutput() : null;
		if (neuronOutput == null) return null;
		NeuronValue derivativeInput = NeuronStandard.getDerivativeInput(outputNeuron);
		if (derivativeInput == null) derivativeInput = neuronOutput;
		
		NeuronValue error = realOutput.subtract(neuronOutput);
		if (!(outputNeuron instanceof Neuron)) {
			if (activateRef == null) return error;
			NeuronValue derivative = derivativeInput.derivative(activateRef);
			return error.multiplyDerivative(derivative);
		}
		NeuronValue errorValue = error;

		//Calculating error.
		Neuron neuron = (Neuron)outputNeuron;
		NeuronValue errorMean = neuron.getErrorMean();
		if (isErrorPriorMeanVariance() || isErrorDynamicMean())
			error = error.add(errorMean); //Add error mean.
		NeuronValue errorVariance = neuron.getErrorVariance();
		if ((isErrorPriorMeanVariance() || isErrorDynamicVariance()) && errorVariance.canInvert())
			error = error.divide(errorVariance); //Divided by error variance.
		if (activateRef != null) {
			NeuronValue derivative = derivativeInput.derivative(activateRef);
			error = error.multiplyDerivative(derivative);
		}
		
		if (!isErrorDynamicMean() && !isErrorDynamicVariance()) return error;

		//Updating error mean and error variance.
		errorValue = errorValue.negative(); //A value contributes to error mean. Please pay attention to this code line.
		if (isErrorDynamicMean()) {
			Mean accumErrorMean = neuron.getAccumErrorMean().accum(errorValue);
			neuron.setAccumErrorMean(accumErrorMean);
		}
		if (isErrorDynamicVariance()) {
			NeuronValue bias = errorValue.subtract(errorMean);
			Mean accumErrorVariance = neuron.getAccumErrorVariance().duplicateShallow();
			accumErrorVariance = accumErrorVariance.accum(bias.multiply(bias));
			if (accumErrorVariance.getMean().canInvert()) neuron.setAccumErrorVariance(accumErrorVariance);
		}
		
		return error;
	}

	
	/**
	 * Calculate error of output neuron.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * @param outputNeuron output neuron.
	 * @param realOutput real output. It can be null.
	 * @param outputLayer output layer. It can be null.
	 * @return error or loss of the output neuron.
	 */
	private static NeuronValue calcOutputErrorDefault(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer) {
		Function activateRef = getActivateRefOutermost(outputNeuron, outputLayer);
		NeuronValue neuronOutput = outputNeuron != null ? outputNeuron.getOutput() : null;
		NeuronValue neuronInput = NeuronStandard.getDerivativeInput(outputNeuron);
		return BackpropagatorAbstract.calcOutputErrorDefault(activateRef, realOutput, neuronOutput, neuronInput);
	}

	
	/**
	 * Getting activation reference.
	 * @param neuron neuron. It can be null.
	 * @param layer layer. It can be null.
	 * @return activation reference.
	 */
	private static Function getActivateRefOutermost(NeuronStandard neuron, LayerStandard layer) {
		if (layer == null && neuron != null && neuron.getLayer() != null) layer = neuron.getLayer();
		if (layer != null && layer instanceof Layer) {
			Function activateRef2 = ((Layer)layer).getActivateRef2();
			if (activateRef2 != null) return activateRef2;
		}
		return BackpropagatorAbstract.getActivateRef(neuron, layer);
	}

	
	/**
	 * Getting parent network.
	 * @return parent network.
	 */
	public Network getParent() {
		return parent;
	}
	
	
	/**
	 * Setting parent network.
	 * @param parent parent network.
	 * @return this generator.
	 */
	public GeneratorStandard<T> setParent(Network parent) {
		this.parent = parent;
		return this;
	}
	
	
	/**
	 * Checking whether to update field.
	 * @param field specified field.
	 * @param defaultValue default value.
	 * @return whether to update field.
	 */
	private boolean getBooleanField(String field, boolean defaultValue) {
		Network parent = getParent();
		if (parent == null) return this.config.getAsBoolean(field);
		try {
			NetworkConfig parentConfig = parent.getConfig();
			boolean update = parentConfig.getAsBoolean(field);
			this.config.put(field, update);
			return update;
		} catch (Throwable e) {Util.trace(e);}
		return defaultValue;
	}

	
	/**
	 * Checking whether error mean is dynamic.
	 * @return whether error mean is dynamic.
	 */
	private boolean isErrorPriorMeanVariance() {
		return getBooleanField(ERROR_PRIOR_MEAN_VARIANCE_FIELD, ERROR_PRIOR_MEAN_VARIANCE_DEFAULT);
	}

	
	/**
	 * Checking whether error mean is dynamic.
	 * @return whether error mean is dynamic.
	 */
	private boolean isErrorDynamicMean() {
		return getBooleanField(ERROR_DYNAMIC_MEAN_FIELD, ERROR_DYNAMIC_MEAN_DEFAULT);
	}

	
	/**
	 * Checking whether error variance is dynamic.
	 * @return whether error variance is dynamic.
	 */
	private boolean isErrorDynamicVariance() {
		return getBooleanField(ERROR_DYNAMIC_VARIANCE_FIELD, ERROR_DYNAMIC_VARIANCE_DEFAULT);
	}
	
	
	/**
	 * Filling default configuration values.
	 * @param config network configuration.
	 */
	public static void fillConfig(NetworkConfig config) {
		config.put(ERROR_PRIOR_MEAN_VARIANCE_FIELD, ERROR_PRIOR_MEAN_VARIANCE_DEFAULT);
		config.put(ERROR_DYNAMIC_MEAN_FIELD, ERROR_DYNAMIC_MEAN_DEFAULT);
		config.put(ERROR_DYNAMIC_VARIANCE_FIELD, ERROR_DYNAMIC_VARIANCE_DEFAULT);
	}


}
