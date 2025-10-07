/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.generator;

import java.io.Serializable;

import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents learning algorithm to train generator.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Trainer extends Serializable, Cloneable {

	
	/**
	 * Setting generator.
	 * @param generator specified generator.
	 */
	void setGenerator(Generator generator);
	
	
	/**
	 * Learning generator by back propagate algorithm, one-by-one record over sample.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration);
	
	
	/**
	 * Learning generator by back propagate algorithm.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration);


}
