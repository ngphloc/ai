package net.ea.ann.transformer;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
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
	 * Y error.
	 */
	public Matrix errorY = null;
	
	
	/**
	 * X error.
	 */
	public Matrix errorX = null;
	
	
	/**
	 * Default constructor.
	 */
	public Error() {
		
	}
	
	
	/**
	 * Constructor with Y error and X error.
	 * @param errorY Y error.
	 * @param errorX X error.
	 */
	public Error(Matrix errorY, Matrix errorX) {
		this.errorY = errorY;
		this.errorX = errorX;
	}
	
	
	/**
	 * Constructor with Y error.
	 * @param errorY Y error.
	 */
	public Error(Matrix errorY) {
		this.errorY = errorY;
	}
	
	
	/**
	 * Getting default error.
	 * @return default error.
	 */
	public Matrix error() {return errorY;}
	
	
	/**
	 * Creating error array.
	 * @param errs array of matrix errors.
	 * @return error array.
	 */
	public static Error[] create(Matrix...errs) {
		if (errs == null || errs.length == 0) return null;
		Error[] errors = new Error[errs.length];
		for (int i = 0; i < errs.length; i++) errors[i] = new Error(errs[i]);
		return errors;
	}
	
	
	/**
	 * Extracting matrix errors.
	 * @param errors errors.
	 * @return matrix errors.
	 */
	public static Matrix[] create(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		Matrix[] errs = new Matrix[errors.length];
		for (int i = 0; i < errors.length; i++) errs[i] = errors[i].error();
		return errs;
	}
	
	
	/**
	 * Extracting X errors.
	 * @param errors errors.
	 * @return X errors.
	 */
	public static Error[] extractX(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		List<Error> errXList = Util.newList(0);
		for (Error error : errors) {
			if (error.errorX != null) errXList.add(new Error(error.errorX));
		}
		return errXList.size() > 0 ? errXList.toArray(new Error[] {}) : null;
	}
	
	
}



