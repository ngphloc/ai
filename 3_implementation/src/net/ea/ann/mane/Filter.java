/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter extends Serializable, Cloneable {

	
	/**
	 * Flag to calculate error mean.
	 */
	static boolean CALC_ERROR_MEAN = net.ea.ann.conv.filter.Filter.CALC_ERROR_MEAN;
	
	
	/**
	 * Moving stride mode.
	 */
	static boolean MOVE_STRIDE = false;
	
	
	/**
	 * Getting filter width.
	 * @return filter width.
	 */
	default int width() {return 1;}

	
	/**
	 * Getting stride width.
	 * @return stride width.
	 */
	default int getStrideWidth() {
		return isMoveStride() ? width() : 1;
	}
	
	
	/**
	 * Getting filter height.
	 * @return filter height.
	 */
	default int height() {return 1;}
	
	
	/**
	 * Getting stride height.
	 * @return stride height.
	 */
	default int getStrideHeight() {
		return isMoveStride() ? height() : 1;
	}
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @return whether to move according to stride when filtering.
	 */
	boolean isMoveStride();
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @param moveStride flag to whether to move according to stride when filtering.
	 */
	void setMoveStride(boolean moveStride);
	
	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	default boolean isPadZero() {return true;}

	
	/**
	 * Getting applying activation function mode.
	 * If activation function mode is false, the filter does not apply activation function like ReLU into squashing the output.
	 * @return applying activation function mode.
	 */
	default boolean doesApplyActivate() {return true;}
	
	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	default void initParams(double v) {}
	
	
	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	default void initParams(Random rnd) {}
	
	
	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	default int sizeOfParams() {return 0;}

	
	/**
	 * Accumulating kernel.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 * @return this filter.
	 */
	Filter accumKernel(Kernel dKernel, double factor);

	
	/**
	 * Forwarding evaluation from previous layer to current layer.
	 * @param time time.
	 * @param prevLayer current layer.
	 * @param thisInputLayer current input layer.
	 * @param thisOutputLayer current output layer.
	 * @param bias bias.
	 * @param thisActivateRef activation function.
	 */
	void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef);

	
	/**
	 * Calculating derivative of previous layers given current layers as bias layers.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of previous layers given current layers as bias layers.
	 */
	Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef);

	
	/**
	 * Calculating derivative of kernel of previous layers given current layers as bias layers.
	 * @param time time.
	 * @param prevInputLayer previous input layer.
	 * @param prevOutputLayer previous output layer.
	 * @param thisErrorLayer current layer as bias layer.
	 * @param thisActivateRef activation function of current layer.
	 * @return derivative of kernel of previous layers given current layers as bias layers.
	 */
	Kernel dKernel(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef);

	
}



