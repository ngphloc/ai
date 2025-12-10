/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.function.Softmax;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This interface represents gradient of likelihood function to train matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@FunctionalInterface
public interface LikelihoodGradient {

	
	/**
	 * Calculating the optimal derivative given computed output and real output.
	 * @param output computed or predicted output.
	 * @param realOutput real output from environment. It can be null.
	 * @param params additional parameters.
	 * @return optimal derivative.
	 */
	Matrix gradient(Matrix output, Matrix realOutput, Object...params);

	
	/**
	 * Calculating error of computed output and environmental output, which is often the negative of likelood.
	 * @param output computed or predicted output. It cannot be null.
	 * @param realOutput real output from environment. It cannot be null.
	 * @param params additional parameters.
	 * @return the last bias.
	 */
	static Matrix error(Matrix output, Matrix realOutput, Object...params) {
		return realOutput.subtract(output);
	}
	
	
	/**
	 * Calculating gradient of loss entropy by column.
	 * @param output computed or predicted output.
	 * @param realOutputProb real probabilistic distribution from environment.
	 * @param params additional parameters.
	 * @return the last bias.
	 */
	static Matrix lossEntropyGradientByRow(Matrix output, Matrix realOutputProb, Object...params) {
		//Normalizing real probabilities.
		for (int row = 0; row < realOutputProb.rows(); row++) {
			NeuronValue sum = realOutputProb.get(0, 0).zero();
			for (int column = 0; column < realOutputProb.columns(); column++) {
				sum = sum.add(realOutputProb.get(row, column));
			}
			if (!sum.canInvert()) continue;

			for (int column = 0; column < realOutputProb.columns(); column++) {
				NeuronValue p = realOutputProb.get(row, column);
				realOutputProb.set(row, column, p.divide(sum));
			}
		}

		//Weight matrix.
		//Sample weight.
		Matrix sampleWeight = null;
		if (params != null && params.length > 0 && params[0] != null && (params[0] instanceof Matrix)) {
			sampleWeight = (Matrix)params[0];
			if (sampleWeight.rows() != realOutputProb.rows() || sampleWeight.columns() != realOutputProb.columns())
				sampleWeight = null;
		}
		
		//Calculating gradient of loss entropy by row.
		int rows = output.rows(), columns = output.columns();
		Matrix grad = output.create(new Size(columns, rows));
		Matrix softmax = Softmax.softmaxByRow(output);
		NeuronValue zero = output.get(0, 0).zero();
		NeuronValue unit = zero.unit();
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue sum = zero;
				NeuronValue p = softmax.get(row, column);
				for (int i = 0; i < columns; i++) {
					NeuronValue realp = realOutputProb.get(row, i);
					NeuronValue value = i==column ? realp.multiply(unit.subtract(p)) : realp.multiply(p.negative());
					//Optional coding line for class balancing by weighted sample.
					if (sampleWeight != null) value = value.multiply(sampleWeight.get(row, column));
					sum = sum.add(value);
				}
				grad.set(row, column, sum);
			}
		}
		
		return grad;
	}

	
	/**
	 * Calculating gradient of loss entropy by column.
	 * @param output computed or predicted output.
	 * @param realOutputProb real probabilistic distribution from environment.
	 * @param params additional parameters.
	 * @return the last bias.
	 */
	static Matrix lossEntropyGradientByColumn(Matrix output, Matrix realOutputProb, Object...params) {
		//Normalizing real probabilities.
		for (int column = 0; column < realOutputProb.columns(); column++) {
			NeuronValue sum = realOutputProb.get(0, 0).zero();
			for (int row = 0; row < realOutputProb.rows(); row++) {
				sum = sum.add(realOutputProb.get(row, column));
			}
			if (!sum.canInvert()) continue;

			for (int row = 0; row < realOutputProb.rows(); row++) {
				NeuronValue p = realOutputProb.get(row, column);
				realOutputProb.set(row, column, p.divide(sum));
			}
		}
		
		//Sample weight.
		Matrix sampleWeight = null;
		if (params != null && params.length > 0 && params[0] != null && (params[0] instanceof Matrix)) {
			sampleWeight = (Matrix)params[0];
			if (sampleWeight.rows() != realOutputProb.rows() || sampleWeight.columns() != realOutputProb.columns())
				sampleWeight = null;
		}

		//Calculating gradient of loss entropy by column.
		int rows = output.rows(), columns = output.columns();
		Matrix grad = output.create(new Size(columns, rows));
		Matrix softmax = Softmax.softmaxByColumn(output);
		NeuronValue zero = output.get(0, 0).zero();
		NeuronValue unit = zero.unit();
		for (int column = 0; column < columns; column++) {
			for (int row = 0; row < rows; row++) {
				NeuronValue sum = zero;
				NeuronValue p = softmax.get(row, column);
				for (int i = 0; i < rows; i++) {
					NeuronValue realp = realOutputProb.get(i, column);
					NeuronValue value = i==row ? realp.multiply(unit.subtract(p)) : realp.multiply(p.negative());
					//Optional coding line for class balancing by weighted sample.
					if (sampleWeight != null) value = value.multiply(sampleWeight.get(row, column));
					sum = sum.add(value);
				}
				grad.set(row, column, sum);
			}
		}
		
		return grad;
	}
	
	
}
