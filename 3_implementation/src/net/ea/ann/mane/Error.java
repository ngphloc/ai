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

import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
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
	 * This class represents an association of layer and input.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class LayerInput implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Layer.
		 */
		public Object layer = null;
		
		/**
		 * Input.
		 */
		public Matrix oinput = null;
		
		/**
		 * Constructor with layer and input.
		 * @param layer layer.
		 * @param oinput input.
		 */
		public LayerInput(Object layer, Matrix oinput) {
			this.layer = layer;
			this.oinput = oinput;
		}
		
		/**
		 * Getting output layer.
		 * @return output layer.
		 */
		public static MatrixLayerAbstract getOutputLayer(MatrixLayerExt layer) {
			if (layer == null) return null;
			MatrixLayer outputLayer = layer.getOutputLayer();
			return (outputLayer != null) && (outputLayer instanceof MatrixLayerAbstract) ? (MatrixLayerAbstract)outputLayer : null;
		}

		/**
		 * Getting output layer.
		 * @return output layer.
		 */
		public static MatrixLayerAbstract getOutputLayer(Object layer) {
			if ((layer == null) || !(layer instanceof MatrixLayerExt))
				return null;
			else
				return getOutputLayer((MatrixLayerExt)layer);
		}

		/**
		 * Getting output layer.
		 * @return output layer.
		 */
		public MatrixLayerAbstract getOutputLayer() {
			return getOutputLayer(layer);
		}
		
		/**
		 * Getting activation function.
		 * @return activation function.
		 */
		public static Function getOutputActivateRef(Object layer) {
			MatrixLayerAbstract outputLayer = getOutputLayer(layer);
			if (outputLayer == null)
				return null;
			else if (outputLayer.containsWeights())
				return outputLayer.getActivateRef();
			else if (outputLayer.getFilter() != null)
				return outputLayer.getConvActivateRef();
			else
				return null;
		}

		/**
		 * Getting activation function.
		 * @return activation function.
		 */
		public Function getOutputActivateRef() {
			return getOutputActivateRef(layer);
		}
		
		/**
		 * Calculating derivative of input.
		 * @return derivative of input.
		 */
		public Matrix derivative() {
			if (oinput == null) return null;
			Function activateRef = getOutputActivateRef();
			return activateRef != null ? oinput.derivativeWise(activateRef) : null;
		}
		
	}
	
	
	/**
	 * This class represents core error in training.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class Error0 implements Cloneable, Serializable {
		
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
		 * Real output of current layer.
		 */
		public Matrix realOutput = null;
		
		/**
		 * List of layer inputs.
		 */
		public List<LayerInput> layerOInputs = Util.newList(0);
		
		/**
		 * Constructor with error, input, output, and real output.
		 * @param error error.
		 * @param input input of current layer.
		 * @param output output of current layer.
		 * @param realOutput real output.
		 */
		public Error0(Matrix error, Matrix input, Matrix output, Matrix realOutput) {
			this.error = error;
			this.input = input;
			this.output = output;
			this.realOutput = realOutput;
		}

		/**
		 * Constructor with error, input, and output.
		 * @param error error.
		 * @param input input of current layer.
		 * @param output output of current layer.
		 */
		public Error0(Matrix error, Matrix input, Matrix output) {
			this(error, input, output, null);
		}

		/**
		 * Constructor with error and input.
		 * @param error error.
		 * @param input input of current layer.
		 */
		public Error0(Matrix error, Matrix input) {
			this(error, input, null);
		}

		/**
		 * Constructor with error.
		 * @param error error.
		 */
		public Error0(Matrix error) {
			this(error, null);
		}

		/**
		 * Getting output.
		 * @return output of current layer.
		 */
		public Matrix realOutput() {return this.realOutput;}
		
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
		 * Copying source errors to target errors.
		 * @param sourceErrors source errors.
		 * @param targetErrors target errors.
		 */
		public static void copyErrors(Matrix[] sourceErrors, Error0[] targetErrors) {
			if (sourceErrors == null || targetErrors == null) return;
			int n = Math.min(sourceErrors.length, targetErrors.length);
			for (int i = 0; i < n; i++) targetErrors[i].error = sourceErrors[i];
		}
		
		/**
		 * Creating errors.
		 * @param errors source errors.
		 * @return created errors.
		 */
		public static Error0[] create(Matrix...errors) {
			if (errors == null || errors.length == 0) return null;
			Error0[] targetErrors = new Error0[errors.length];
			for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = new Error0(errors[i]);
			return targetErrors;
		}
		
		/**
		 * Creating errors.
		 * @param errors source errors.
		 * @return created errors.
		 */
		public static Error0[] create(List<Matrix> errors) {
			if (errors == null || errors.size() == 0) return null;
			Error0[] targetErrors = new Error0[errors.size()];
			for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = new Error0(errors.get(i));
			return targetErrors;
		}

		/**
		 * Extracting errors.
		 * @param errors source errors.
		 * @return extracted errors.
		 */
		public static Matrix[] errors(Error0...errors) {
			if (errors == null || errors.length == 0) return null;
			Matrix[] targetErrors = new Matrix[errors.length];
			for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = errors[i].error;
			return targetErrors;
		}
		
	}


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
	 * Constructor with error, input, output, and real output.
	 * @param error error.
	 * @param input input of current layer.
	 * @param output output of current layer.
	 * @param realOutput real output.
	 */
	public Error(Matrix error, Matrix input, Matrix output, Matrix realOutput) {
		this(new Error0(error, input, output, realOutput));
	}

	
	/**
	 * Constructor with error, input, and output.
	 * @param error error.
	 * @param input input of current layer.
	 * @param output output of current layer.
	 */
	public Error(Matrix error, Matrix input, Matrix output) {
		this(new Error0(error, input, output));
	}

	
	/**
	 * Constructor with error and input.
	 * @param error error.
	 * @param input input of current layer.
	 */
	public Error(Matrix error, Matrix input) {
		this(new Error0(error, input));
	}

	
	/**
	 * Constructor with error.
	 * @param error error.
	 */
	public Error(Matrix error) {
		this(new Error0(error));
	}
	
	
	/**
	 * Getting array of core errors.
	 * @return array of core errors.
	 */
	public Error0[] errors() {
		return errors.toArray(new Error0[] {});
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
	 * @param error error.
	 * @param input input of current layer.
	 * @param output output of current layer.
	 * @param realOutput real output.
	 */
	public boolean add(Matrix error, Matrix input, Matrix output, Matrix realOutput) {
		return add(new Error0(error, input, output, realOutput));
	}

	
	/**
	 * Adding core error.
	 * @param error error.
	 * @param input input of current layer.
	 * @param output output of current layer.
	 */
	public boolean add(Matrix error, Matrix input, Matrix output) {
		return add(new Error0(error, input, output));
	}

	
	/**
	 * Adding core error.
	 * @param error error.
	 * @param input input of current layer.
	 */
	public boolean add(Matrix error, Matrix input) {
		return add(new Error0(error, input));
	}

	
	/**
	 * Adding core error.
	 * @param error error.
	 */
	public boolean add(Matrix error) {
		return add(new Error0(error));
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
	 * Getting default core error at specified index.
	 * @param index specified index.
	 * @return default core error at specified index.
	 */
	public Matrix error(int index) {
		Error0 error0 = get(index);
		return error0 != null ? error0.error : null;
	}

	
	/**
	 * Setting default core error.
	 * @param index specified index.
	 * @param error default core error.
	 */
	public void errorSet(int index, Matrix error) {
		Error0 error0 = get(index);
		if (error0 != null) error0.error = error;
	}

	
	/**
	 * Getting default core error.
	 * @return default core error.
	 */
	public Matrix error() {
		Error0 error0 = get();
		return error0 != null ? error0.error : null;
	}
	
	
	/**
	 * Setting default core error.
	 * @param error default core error.
	 */
	public void errorSet(Matrix error) {
		Error0 error0 = get();
		if (error0 != null) error0.error = error;
	}
	
	
	/**
	 * Getting second core error.
	 * @return second core error.
	 */
	public Matrix error2() {
		Error0 error0 = get2();
		return error0 != null ? error0.error : null;
	}

	
	/**
	 * Setting second core error.
	 * @param error second core error.
	 */
	public void errorSet2(Matrix error) {
		Error0 error0 = get2();
		if (error0 != null) error0.error = error;
	}

	
	/**
	 * Getting default input.
	 * @return default input of current layer.
	 */
	public Matrix input() {
		Error0 error0 = get();
		return error0 != null ? error0.input : null;
	}

	
	/**
	 * Getting default output.
	 * @return default output of current layer.
	 */
	public Matrix output() {
		Error0 error0 = get();
		return error0 != null ? error0.output : null;
	}
	
	
	/**
	 * Getting default real output.
	 * @return default real output of current layer.
	 */
	public Matrix realOutput() {
		Error0 error0 = get();
		return error0 != null ? error0.realOutput : null;
	}

	
	/**
	 * Getting second input.
	 * @return second input of current layer.
	 */
	public Matrix input2() {
		Error0 error0 = get2();
		return error0 != null ? error0.input : null;
	}

	
	/**
	 * Getting second output.
	 * @return second output of current layer.
	 */
	public Matrix output2() {
		Error0 error0 = get2();
		return error0 != null ? error0.output : null;
	}

	
	/**
	 * Getting second real output.
	 * @return second real output of current layer.
	 */
	public Matrix realOutput2() {
		Error0 error0 = get2();
		return error0 != null ? error0.realOutput : null;
	}

	
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
	 * Copying source errors to target errors.
	 * @param sourceErrors source errors.
	 * @param targetErrors target errors.
	 */
	public static void copyErrors(Matrix[] sourceErrors, Error[] targetErrors) {
		if (sourceErrors == null || targetErrors == null) return;
		int n = Math.min(sourceErrors.length, targetErrors.length);
		for (int i = 0; i < n; i++) targetErrors[i] = new Error(sourceErrors[i]);
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
		for (int i = 0; i < targetErrors.length; i++) targetErrors[i] = errors[i].error();
		return targetErrors;
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




