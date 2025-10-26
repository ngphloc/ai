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
	 * Default constructor.
	 */
	public Error() {
		
	}
	
	
	/**
	 * Getting default error.
	 * @return default error.
	 */
	public Matrix error() {return errorY;}
	
	
	/**
	 * Getting main error.
	 * @return main error.
	 */
	public Matrix mainError() {return error();}
	
	
	/**
	 * Getting attach error.
	 * @return attach error.
	 */
	public Matrix attachError() {return errorX;}
	
	
	/**
	 * Accumulating with other error.
	 * @param error other error.
	 * @return accumulative error.
	 */
	public Error accum(Error error) {
		if (error == null) return this;
		Error result = new Error();
		
		if (this.errorY != null && error.errorY != null)
			result.errorY = Matrix.sum(this.errorY, error.errorY);
		else if (this.errorY != null && error.errorY == null)
			result.errorY = this.errorY;
		else if (this.errorY == null && error.errorY != null)
			result.errorY = error.errorY;
		
		if (this.errorX != null && error.errorX != null)
			result.errorX = Matrix.sum(this.errorX, error.errorX);
		else if (this.errorX != null && error.errorX == null)
			result.errorX = this.errorX;
		else if (this.errorX == null && error.errorX != null)
			result.errorX = error.errorX;
		
		return result;
	}
	
	
	/**
	 * Accumulating two arrays of errors
	 * @param errors1 errors 1
	 * @param errors2 errors 2
	 * @return array of accumulated errors.
	 */
	public static Error[] accum(Error[] errors1, Error[] errors2) {
		if (errors1 == null) return errors2;
		if (errors2 == null) return errors1;
		int N = Math.max(errors1.length, errors2.length);
		Error[] result = new Error[N];
		for (int i = 0; i < N; i++) {
			if (i < errors1.length && i < errors2.length)
				result[i] = errors1[i].accum(errors2[i]);
			else if (i >= errors1.length)
				result[i] = errors2[i];
			else if (i >= errors2.length)
				result[i] = errors1[i];
		}
		return result;
	}
	
	
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
	 * Creating error array.
	 * @param errs array of matrix errors.
	 * @return error array.
	 */
	public static Error[] create(net.ea.ann.mane.Error...errs) {
		if (errs == null || errs.length == 0) return null;
		Error[] errors = new Error[errs.length];
		for (int i = 0; i < errs.length; i++) errors[i] = new Error(errs[i].error);
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
	 * Extracting matrix errors.
	 * @param errors errors.
	 * @return matrix errors.
	 */
	public static net.ea.ann.mane.Error[] create2(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		net.ea.ann.mane.Error[] errs = new net.ea.ann.mane.Error[errors.length];
		for (int i = 0; i < errors.length; i++) errs[i] = new net.ea.ann.mane.Error(errors[i].error());
		return errs;
	}

	
	/**
	 * Extracting X errors.
	 * @param errors errors.
	 * @return X errors.
	 */
	public static Error[] createByX(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		List<Error> errXList = Util.newList(0);
		for (Error error : errors) {
			if (error.errorX != null) errXList.add(new Error(error.errorX));
		}
		return errXList.size() > 0 ? errXList.toArray(new Error[] {}) : null;
	}
	
	
	
}



