/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.generator;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is an associator of standard neural network.
 * 
 * @author Loc Nguyen
 * @param <T> type of trainer.
 * @version 1.0
 *
 */
public class GeneratorStandardAssoc<T extends Trainer> implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal generator.
	 */
	protected GeneratorStandard<T> generator = null;
	
	
	/**
	 * Constructor with specified generator.
	 * @param generator specified generator.
	 */
	public GeneratorStandardAssoc(GeneratorStandard<T> generator) {
		this.generator = generator;
	}

	
	/**
	 * Resetting error means and error variances of all neurons.
	 * @param mean mean.
	 * @param variance variance.
	 * @return this association.
	 */
	public GeneratorStandardAssoc<T> resetErrorMeansVariances(double mean, double variance) {
		List<LayerStandard> layers = generator.getAllLayers();
		for (LayerStandard layer : layers) {
			NeuronValue nmean = layer.newNeuronValue().valueOf(mean);
			NeuronValue nvariance = nmean.valueOf(variance);
			for (int i = 0; i < layer.size(); i++) {
				NeuronStandard neuron = layer.get(i);
				if (!(neuron instanceof net.ea.ann.core.generator.GeneratorStandard.Neuron)) continue;
				net.ea.ann.core.generator.GeneratorStandard.Neuron gn = (net.ea.ann.core.generator.GeneratorStandard.Neuron)neuron;
				gn.resetAccumErrorMean(nmean);
				gn.resetAccumErrorVariance(nvariance);
			}
		}
		
		return this;
	}

	
	/**
	 * Resetting zero error means and unit error variances of all neurons.
	 * @return this association.
	 */
	public GeneratorStandardAssoc<T> resetErrorMeansVariances( ) {
		resetErrorMeansVariances(0, 1);
		return this;
	}


}
