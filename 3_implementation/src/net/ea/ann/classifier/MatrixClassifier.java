/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.awt.Dimension;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.raster.Raster;

/**
 * This class is default implementation of classifier within context of matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixClassifier extends MatrixClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier nut.
	 */
	protected MatrixNetworkImpl adjuster = null;
	
	
	/**
	 * Adjusting baseline.
	 */
	protected Matrix adjustBaseline = null;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixClassifier(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixClassifier(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixClassifier(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixClassifier(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	@Override
	public void reset() {
		super.reset();
		adjuster = null;
		adjustBaseline = null;
	}


	@Override
	protected boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension nCoreClasses2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2)) return false;
		if (!paramIsAdjust() || !paramIsBaseline()) return true;
		
		int minAdjustDepth = Math.max((int)(Math.log(this.nut.size())/Math.log(MatrixNetworkImpl.BASE_DEFAULT)), 1);
		int maxAdjustDepth = Math.max(Math.max(this.nut.size()-1, 1), minAdjustDepth);
		int adjustDepth = Math.max(Math.max(depth1, depth2), minAdjustDepth);
		adjustDepth = Math.min(adjustDepth, maxAdjustDepth);
		Dimension size = this.nut.getOutputLayer().getSize();
		this.adjustBaseline = null;
		this.adjuster = new MatrixNetworkImpl(this.neuronChannel, this.nut.getActivateRef(), this.nut.getConvActivateRef(), this.idRef);
		this.adjuster.paramSetInclude(this);
		return new MatrixNetworkInitializer(adjuster).initialize(size, size, adjustDepth);
	}


	@Override
	double[] weightsOfOutput(Matrix output, int groupIndex) {
		if (adjuster == null) return super.weightsOfOutput(output, groupIndex);
		NeuronValue[] values = getOutput(output, groupIndex);
		if (this.baseline == null || this.adjustBaseline == null) return super.weightsOfOutput(output, groupIndex);
		
		NeuronValue zero = values[0].zero();
		for (int classIndex = 0; classIndex < values.length; classIndex++) {
			NeuronValue base = paramIsByColumn() ? this.baseline.get(classIndex, groupIndex) : this.baseline.get(groupIndex, classIndex);
			NeuronValue adjustBase = paramIsByColumn() ? this.adjustBaseline.get(classIndex, groupIndex) : this.adjustBaseline.get(groupIndex, classIndex);
			//Following code lines are important due to apply baseline into determining class.
			NeuronValue sim = values[classIndex].subtract(base);
			sim = sim.max(zero);
			sim = sim.multiply(adjustBase);
			values[classIndex] = sim;
		}
		return weightsOfOutput(values);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		List<Record> newsample = prelearn(sample);
		Error[] errors = learn(newsample);
		learnVerify(newsample);
		
		if (this.adjuster != null) {
			this.adjuster.paramSetInclude(this);
			List<Record> adjustSample = Util.newList(0);
			for (Record record : newsample) {
				Matrix output = evaluate(record.input());
				if (output != null) adjustSample.add(new Record(output, record.output()));
			}
			errors = adjustSample.size() > 0 ? this.adjuster.learn(adjustSample) : errors;
		}
		
		NeuronValue[] errorArray = null;
		for (Error error : errors) {
			NeuronValue[] values = Matrix.extractValues(error.error);
			errorArray = errorArray == null ? values : NeuronValue.concatArray(errorArray, values);
		}
		return errorArray;
	}

	
	@Override
	public void learnVerify(Iterable<Record> inouts) {
		if (adjuster == null) {
			super.learnVerify(inouts);
			return;
		}
		this.baseline = null;
		this.adjustBaseline = null;
		if (!paramIsBaseline()) return;
		
		List<Matrix> outputList = Util.newList(0);
		for (Record inout : inouts) {
			Matrix output = evaluate(inout.input());
			if (output != null) outputList.add(output);
		}
		if (outputList.size() == 0) return;
		this.baseline = calcBaseline(outputList.toArray(new Matrix[] {}));
		
		List<Matrix> adjustOutputList = Util.newList(0);
		for (Matrix output : outputList) {
			try {
				Matrix adjustOutput = adjuster.evaluate(output);
				if (adjustOutput != null) adjustOutputList.add(adjustOutput);
			} catch (Throwable e) {Util.trace(e);}
		}
		if (adjustOutputList.size() == 0) return;
		this.adjustBaseline = calcBaseline(adjustOutputList.toArray(new Matrix[] {}));
	}

	
	/**
	 * Creating matrix neural classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static MatrixClassifier create(int neuronChannel, boolean isNorm) {
		Function activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		MatrixClassifier mac = new MatrixClassifier(neuronChannel, activateRef, contentActivateRef, null);
		mac.paramSetNorm(isNorm);
		return mac;
	}


}



/**
 * This class is abstract implementation of classifier within context of matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class MatrixClassifierAbstract extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier nut.
	 */
	protected MatrixNetworkImpl nut = null;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixClassifierAbstract(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, idRef);
		
		this.nut = new MatrixNetworkImpl(this.neuronChannel, activateRef, convActivateRef, idRef);
		try {
			this.config.putAll(this.nut.getConfig());
		} catch (Throwable e) {Util.trace(e);}
		this.nut.setTrainer(new TaskTrainerLossEntropy());
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixClassifierAbstract(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixClassifierAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixClassifierAbstract(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	@Override
	protected Filter2D defaultFilter(Dimension filterStride) {
		return nut != null ? nut.defaultFilter(filterStride) : super.defaultFilter(filterStride);
	}


	@Override
	protected boolean initialize(Dimension inputSize1, Dimension outputSize1, Filter2D filter1, int depth1, boolean dual1, Dimension nCoreClasses2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2)) return false;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = paramIsByColumn() ? nCoreClasses2.width : nCoreClasses2.height;
		Dimension outputSize2 = paramIsByColumn() ? new Dimension(groupCount, outputCount) : new Dimension(outputCount, groupCount);
		boolean initialized = false;
		if (paramIsConv())
			initialized = nut.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2);
		else
			initialized = nut.initialize(inputSize1, outputSize2, (Filter2D)null, depth1, false, null, 0);
		if (!initialized) return false;
		
		Matrix output = getOutput();
		if (paramIsByColumn()) {
			if (output.rows() != this.outputClassMaps.get(0).size() ||
				output.columns() != this.outputClassMaps.size()) return false;
		}
		else {
			if (output.rows() != this.outputClassMaps.size() ||
				output.columns() != this.outputClassMaps.get(0).size()) return false;
		}
		
		return true;
	}


	@Override
	protected Matrix getOutput() {
		return nut.getOutput();
	}

	
	@Override
	protected Matrix toMatrix(Raster raster) {
		return nut.getInputLayer().toMatrix(raster);
	}

	
	@Override
	protected Matrix evaluate(Matrix input) {
		try {
			nut.getConfig().putAll(this.config);
			return nut.evaluate(input);
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
	@Override
	protected Error[] learn(Iterable<Record> sample) {
		try {
			nut.getConfig().putAll(this.config);
			return nut.learn(sample);
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}
	

}
