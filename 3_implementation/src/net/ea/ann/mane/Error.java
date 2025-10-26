/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.value.Matrix;

/**
 * This class represents error in training.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Error implements Cloneable, Serializable {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Error.
	 */
	public Matrix error = null;
	
	
	/**
	 * Input of current layer.
	 */
	public Matrix input = null;

	
	/**
	 * Output of current layer.
	 */
	public Matrix output = null;
	
	
	/**
	 * Constructor with error, output, and input.
	 * @param error error.
	 * @param output output of current layer.
	 * @param input input of current layer.
	 */
	public Error(Matrix error, Matrix output, Matrix input) {
		this.error = error;
		this.output = output;
		this.input = input;
	}

	
	/**
	 * Constructor with error and output.
	 * @param error error.
	 * @param output output of current layer.
	 */
	public Error(Matrix error, Matrix output) {
		this.error = error;
		this.output = output;
	}

	
	/**
	 * Constructor with error.
	 * @param error error.
	 */
	public Error(Matrix error) {
		this.error = error;
	}

	
	/**
	 * Getting error.
	 * @return error.
	 */
	public Matrix error() {return this.error;}
	
	
	/**
	 * Getting output.
	 * @return output of current layer.
	 */
	public Matrix output() {return this.output;}
	
	
	/**
	 * Getting input.
	 * @return input of current layer.
	 */
	public Matrix input() {return this.input;}

	
	/**
	 * Copying source errors to target errors.
	 * @param sourceErrors source errors.
	 * @param targetErrors target errors.
	 */
	public static void copyErrors(Matrix[] sourceErrors, Error[] targetErrors) {
		if (sourceErrors == null || targetErrors == null) return;
		int n = Math.min(sourceErrors.length, targetErrors.length);
		for (int i = 0; i < n; i++) targetErrors[i].error = sourceErrors[i];
	}
	
	
	/**
	 * Creating errors.
	 * @param errors source errors.
	 * @return created errors.
	 */
	public static Error[] create(Matrix...errors) {
		if (errors == null || errors.length == 0) return null;
		Error[] targetErrors = new Error[errors.length];
		for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = new Error(errors[i]);
		return targetErrors;
	}
	
	
	/**
	 * Creating errors.
	 * @param errors source errors.
	 * @return created errors.
	 */
	public static Error[] create(List<Matrix> errors) {
		if (errors == null || errors.size() == 0) return null;
		Error[] targetErrors = new Error[errors.size()];
		for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = new Error(errors.get(i));
		return targetErrors;
	}

	
	/**
	 * Extracting errors.
	 * @param errors source errors.
	 * @return extracted errors.
	 */
	public static Matrix[] errors(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		Matrix[] targetErrors = new Matrix[errors.length];
		for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = errors[i].error;
		return targetErrors;
	}

	
}
