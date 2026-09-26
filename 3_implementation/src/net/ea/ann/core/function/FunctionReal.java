package net.ea.ann.core.function;

import java.io.Serializable;

/**
 * This interface represents real function.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface FunctionReal extends Cloneable, Serializable {


	/**
	 * Evaluating specified value.
	 * @param x specified value.
	 * @return evaluated value.
	 */
	double evaluate(double x);
	
	
	/**
	 * Calculate gradient (the first order derivative) at specified value.
	 * @param x specified value.
	 * @return gradient (the first order derivative) at specified value.
	 */
	double derivative(double x);


}
