/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack.bp;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.stack.Stack;
import net.ea.ann.core.Evaluator;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;

/**
 * This interface represents backpropagation algorithm for stack network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Backpropagator extends Serializable, Cloneable {


	/**
	 * Updating weights and biases.
	 * @param sample sample.
	 * @param stacks list of stacks.
	 * @param learningRate learning rate.
	 * @param evaluator specified evaluator.
	 * @return errors of output errors. Return null if errors occur.
	 */
	Content[] updateWeightsBiases(Iterable<Record> sample, List<Stack> stacks, double learningRate, Evaluator evaluator);

	
	/**
	 * Updating weights and biases.
	 * @param stacks list of stacks including input stack.
	 * @param realOutput realistic output of output stack. 
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	Content[] updateWeightsBiases(List<Stack> stacks, Content[] realOutput, double learningRate);

	
	/**
	 * Updating weights and biases. Derived class can override this method.
	 * @param stacks list of stacks including input stack.
	 * @param outputBatch output batch of output stack.
	 * For each element (e = NeuronValue[2][]) of this batch, the first part (e[0]) is real output and the second part (e[1]) is layer output. The second part may be null or removed
	 * because the method {@link NeuronStandard#getOutput()} returns the layer output too. 
	 * @param lastError last error which is optional parameter.
	 * @param learningRate learning rate.
	 * @return errors of output errors. Return null if errors occur.
	 */
	Content[] updateWeightsBiases(List<Stack> stacks, Iterable<Content[][]> outputBatch, Content[] lastError, double learningRate);

	
}
