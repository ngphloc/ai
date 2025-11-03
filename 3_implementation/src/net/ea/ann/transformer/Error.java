package net.ea.ann.transformer;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.MatrixLayerExt;

/**
 * This class represents core error in training.
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
	 * List of core errors.
	 */
	public List<Error0> errors = Util.newList(0);
	
	
	/**
	 * Constructor of array of core errors.
	 * @param inouts array of core errors.
	 */
	public Error(Error0...errors) {
		if (errors == null || errors.length == 0) return;
		for (int i = 0; i < errors.length; i++) {
			if (errors[i] != null) this.errors.add(errors[i]);
		}
	}


	/**
	 * Constructor with Y error and X error.
	 * @param errorY Y error.
	 * @param errorX X error.
	 */
	public Error(Matrix errorY, Matrix errorX) {
		this (new Error0(errorY, errorX));
	}
	
	
	/**
	 * Constructor with Y error.
	 * @param errorY Y error.
	 */
	public Error(Matrix errorY) {
		this (new Error0(errorY));
	}
	
	
	/**
	 * Default constructor.
	 */
	public Error() {
		
	}

	
	/**
	 * Getting number of core errors.
	 * @return number of core errors.
	 */
	public int size() {return errors.size();}
	
	
	/**
	 * Getting core error at specified index.
	 * @param index specified index.
	 * @return core error at specified index.
	 */
	public Error0 get(int index) {return errors.get(index);}
	
	
	/**
	 * Getting first (default) core error.
	 * @return first (default) core error.
	 */
	public Error0 get() {return size() > 0 ? get(0) : null;}
	
	
	/**
	 * Getting second core error.
	 * @return second core error.
	 */
	public Error0 get2() {return size() > 1 ? get(1) : null;}

	
	/**
	 * Adding core error.
	 * @param error0 core error.
	 * @return true if adding is successful.
	 */
	public boolean add(Error0 error0) {
		return errors.add(error0);
	}
	
	
	/**
	 * Adding core error.
	 * @param errorY Y error.
	 * @param errorX X error.
	 */
	public boolean add(Matrix errorY, Matrix errorX) {
		return add(new Error0(errorY, errorX));
	}
	
	
	/**
	 * Adding core error.
	 * @param errorY Y error.
	 */
	public boolean add(Matrix errorY) {
		return add(new Error0(errorY));
	}

	
	/**
	 * Removing core error at specified index.
	 * @param index specified index.
	 * @return previous core error.
	 */
	public Error0 remove(int index) {
		return errors.remove(index);
	}
	
	
	/**
	 * Clearing core errors.
	 */
	public void clear() {
		errors.clear();
	}


	/**
	 * Getting error at specified index.
	 * @param index specified index.
	 * @return error.
	 */
	public Matrix error(int index) {
		Error0 error0 = get(index);
		return error0 != null ? error0.errorY : null;
	}
	
	
	/**
	 * Getting main error at specified index.
	 * @param index specified index 
	 * @return main error.
	 */
	public Matrix mainError(int index) {return error(index);}
	
	
	/**
	 * Getting attached error at specified index.
	 * @param index specified index
	 * @return attached error.
	 */
	public Matrix attachError(int index) {
		Error0 error0 = get(index);
		return error0 != null ? error0.errorX : null;
	}
	
	
	/**
	 * Getting default error.
	 * @return default error.
	 */
	public Matrix error() {return size() > 0 ? error(0) : null;}
	
	
	/**
	 * Setting default core error.
	 * @param error default core error.
	 */
	public void errorSet(Matrix error) {
		Error0 error0 = get();
		if (error0 != null) error0.errorY = error;
	}

	
	/**
	 * Getting default main error.
	 * @return default main error.
	 */
	public Matrix mainError() {return size() > 0 ? mainError(0) : null;}
	
	
	/**
	 * Getting default attached error.
	 * @param index specified index
	 * @return default attached error.
	 */
	public Matrix attachError() {return size() > 0 ? attachError(0) : null;}


	/**
	 * Getting layer inputs.
	 * @return layer inputs.
	 */
	public List<LayerInput> layerOInputs() {
		Error0 error0 = get();
		return error0 != null ? error0.layerOInputs : null;
	}
	
	
	/**
	 * Adding layer input.
	 * @param layerOInput layer input
	 * @return true if adding is successful.
	 */
	boolean addLayerOInput(LayerInput layerOInput) {
		Error0 error0 = get();
		return error0 != null ? error0.addLayerOInput(layerOInput) : false;
	}
	
	
	/**
	 * Adding layer input.
	 * @param layer layer.
	 * @param oinput input.
	 * @return true if adding is successful.
	 */
	public boolean addLayerOInput(Object layer, Matrix oinput) {
		Error0 error0 = get();
		return error0 != null ? error0.addLayerOInput(layer, oinput) : false;
	}

	
	/**
	 * Adding layer input.
	 * @param layer layer.
	 * @return true if adding is successful.
	 */
	public boolean addLayerOInput(MatrixLayerExt layer) {
		Error0 error0 = get();
		return error0 != null ? error0.addLayerOInput(layer) : false;
	}

	
	/**
	 * Getting input of specified layer.
	 * @param layer specified layer.
	 * @return input of specified layer.
	 */
	public Matrix oinputOfLayer(Object layer) {
		Error0 error0 = get();
		return error0 != null ? error0.oinputOfLayer(layer) : null;
	}
	
	
	/**
	 * Calculating derivative of input given layer.
	 * @param layer layer.
	 * @return derivative of input given layer.
	 */
	public Matrix oinputDerivative(Object layer) {
		Error0 error0 = get();
		return error0 != null ? error0.oinputDerivative(layer) : null;
	}

	
	/**
	 * Accumulating with other error.
	 * @param error other error.
	 * @return accumulative error.
	 */
	public Error accum(Error error) {
		if (error == null) return this;
		int N = Math.max(this.size(), error.size());
		if (N == 0) return null;
		Error0[] accum0s = new Error0[N];
		for (int i = 0; i < N; i++) {
			Error0 accum0 = null;
			if (i < this.size() && i < error.size())
				accum0 = this.get(i).accum(error.get(i));
			if (i < this.size())
				accum0 = this.get(i); 
			else if (i < error.size())
				accum0 = error.get(i);
			accum0s[i] = accum0; 
		}
		return new Error(accum0s);
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
			else if (i < errors1.length)
				result[i] = errors1[i];
			else if (i < errors2.length)
				result[i] = errors2[i];
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
		for (int i = 0; i < errs.length; i++) {
			Error0[] error0s = Error0.create(errs[i].errors());
			errors[i] = new Error(error0s);
		}
		return errors;
	}


	/**
	 * Extracting matrix errors.
	 * @param errors errors.
	 * @return matrix errors.
	 */
	public static net.ea.ann.mane.Error[] extract(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		net.ea.ann.mane.Error[] errs = new net.ea.ann.mane.Error[errors.length];
		for (int i = 0; i < errors.length; i++) {
			net.ea.ann.mane.Error.Error0[] error0s = Error0.extract(errors[i].errors.toArray(new Error0[] {}));
			errs[i] = new net.ea.ann.mane.Error(error0s);
		}
		return errs;
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
	 * Extracting attached errors.
	 * @param errors errors.
	 * @return attached errors.
	 */
	public static Error[] createByAttach(Error...errors) {
		if (errors == null || errors.length == 0) return null;
		List<Error> errAttachList = Util.newList(0);
		for (Error error : errors) {
			if (error.attachError() != null) {
				Error err = new Error(error.attachError());
				err.get().layerOInputs.addAll(error.get().layerOInputs);
				errAttachList.add(err);
			}
		}
		return errAttachList.size() > 0 ? errAttachList.toArray(new Error[] {}) : null;
	}


	/**
	 * Adjusting errors
	 * @param layer layer.
	 * @param errors errors.
	 */
	public static void adjustErrors(Object layer, Error...errors) {
		if (errors == null || errors.length == 0) return;
		for (Error error : errors) {
			if (error == null) continue;
			Matrix derivative = error.oinputDerivative(layer);
			if (derivative != null) error.errorSet(derivative.multiplyWise(error.error()));
		}
	}


}


/**
 * This class represents core error in training.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class Error0 implements Cloneable, Serializable {
	
	
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
	 * List of layer inputs.
	 */
	public List<LayerInput> layerOInputs = Util.newList(0);
	
	
	/**
	 * Constructor with Y error and X error.
	 * @param errorY Y error.
	 * @param errorX X error.
	 */
	public Error0(Matrix errorY, Matrix errorX) {
		this.errorY = errorY;
		this.errorX = errorX;
	}
	
	
	/**
	 * Constructor with Y error.
	 * @param errorY Y error.
	 */
	public Error0(Matrix errorY) {
		this.errorY = errorY;
	}
	
	
	/**
	 * Default constructor.
	 */
	public Error0() {
		
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
	 * Getting layer input.
	 * @param layer layer.
	 * @return input.
	 */
	LayerInput layerInput(Object layer) {
		for (LayerInput layerInput : layerOInputs) {
			if (layerInput.layer == layer) return layerInput;
		}
		return null;
	}

	
	/**
	 * Adding layer input.
	 * @param layerOInput layer input
	 * @return true if adding is successful.
	 */
	boolean addLayerOInput(LayerInput layerOInput) {
		return layerOInputs.add(layerOInput);
	}
	
	
	/**
	 * Adding layer input.
	 * @param layer layer.
	 * @param oinput input.
	 * @return true if adding is successful.
	 */
	public boolean addLayerOInput(Object layer, Matrix oinput) {
		return addLayerOInput(new LayerInput(layer, oinput));
	}
	
	
	/**
	 * Adding layer input.
	 * @param layer layer.
	 * @return true if adding is successful.
	 */
	public boolean addLayerOInput(MatrixLayerExt layer) {
		if (layer == null) return false;
		MatrixLayerAbstract outputLayer = LayerInput.getOutputLayer(layer);
		if (outputLayer == null) return false;
		Matrix oinput = outputLayer.queryActualInput();
		return oinput != null ? addLayerOInput(new LayerInput(layer, oinput)) : false;
	}

	
	/**
	 * Getting input of specified layer.
	 * @param layer specified layer.
	 * @return input of specified layer.
	 */
	public Matrix oinputOfLayer(Object layer) {
		LayerInput layerInput = layerInput(layer);
		return layerInput != null ? layerInput.oinput : null;
	}
	

	/**
	 * Calculating derivative of input given layer.
	 * @param layer layer.
	 * @return derivative of input given layer.
	 */
	public Matrix oinputDerivative(Object layer) {
		LayerInput layerInput = layerInput(layer);
		return layerInput != null ? layerInput.derivative() : null;
	}

	
	/**
	 * Accumulating with other error.
	 * @param error other error.
	 * @return accumulative error.
	 */
	public Error0 accum(Error0 error) {
		if (error == null) return this;
		Error0 result = new Error0();
		
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
	public static Error0[] accum(Error0[] errors1, Error0[] errors2) {
		if (errors1 == null) return errors2;
		if (errors2 == null) return errors1;
		int N = Math.max(errors1.length, errors2.length);
		Error0[] result = new Error0[N];
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
	public static Error0[] create(Matrix...errs) {
		if (errs == null || errs.length == 0) return null;
		Error0[] errors = new Error0[errs.length];
		for (int i = 0; i < errs.length; i++) errors[i] = new Error0(errs[i]);
		return errors;
	}
	
	
	/**
	 * Creating error array.
	 * @param errs array of matrix errors.
	 * @return error array.
	 */
	public static Error0[] create(net.ea.ann.mane.Error.Error0...errs) {
		if (errs == null || errs.length == 0) return null;
		Error0[] errors = new Error0[errs.length];
		for (int i = 0; i < errs.length; i++) {
			errors[i] = new Error0(errs[i].error);
			for (net.ea.ann.mane.Error.LayerInput layerOInput : errs[i].layerOInputs) {
				errors[i].layerOInputs.add(new LayerInput(layerOInput));
			}
		}
		return errors;
	}

	
	/**
	 * Extracting matrix errors.
	 * @param errors errors.
	 * @return matrix errors.
	 */
	public static Matrix[] create(Error0...errors) {
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
	public static Error0[] createByAttach(Error0...errors) {
		if (errors == null || errors.length == 0) return null;
		List<Error0> errAttachList = Util.newList(0);
		for (Error0 error : errors) {
			if (error.attachError() != null) {
				Error0 error0 = new Error0(error.attachError());
				error0.layerOInputs.addAll(error.layerOInputs);
				errAttachList.add(error0);
			}
		}
		return errAttachList.size() > 0 ? errAttachList.toArray(new Error0[] {}) : null;
	}
	
	
	/**
	 * Extracting matrix errors.
	 * @param errors errors.
	 * @return matrix errors.
	 */
	public static net.ea.ann.mane.Error.Error0[] extract(Error0...errors) {
		if (errors == null || errors.length == 0) return null;
		net.ea.ann.mane.Error.Error0[] errs = new net.ea.ann.mane.Error.Error0[errors.length];
		for (int i = 0; i < errors.length; i++) {
			errs[i] = new net.ea.ann.mane.Error.Error0(errors[i].error());
			errs[i].layerOInputs.addAll(errors[i].layerOInputs);
		}
		return errs;
	}

	
}



/**
 * This class represents an association of layer and input.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class LayerInput extends net.ea.ann.mane.Error.LayerInput {
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor with layer and input.
	 * @param layer layer.
	 * @param oinput input.
	 */
	public LayerInput(Object layer, Matrix oinput) {
		super(layer, oinput);
	}
	
	/**
	 * Constructor with layer input.
	 * @param layerOInput layer input.
	 */
	public LayerInput(net.ea.ann.mane.Error.LayerInput layerOInput) {
		super(layerOInput.layer, layerOInput.oinput);
	}
	
}


