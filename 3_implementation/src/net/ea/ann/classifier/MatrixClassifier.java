/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.mane.MatrixNetworkInitializer;
import net.ea.ann.mane.Record;
import net.ea.ann.mane.TaskTrainerLossEntropy;
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

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
	MatrixNetworkImpl adjuster = null;
	
	
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
	}


	@Override
	void updateConfig() {
		super.updateConfig();
		if (adjuster != null) adjuster.paramSetInclude(this);
	}


	@Override
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size nCoreClasses2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, nCoreClasses2, depth2)) return false;
		this.adjuster = null;
		if (!paramIsAdjust() || !paramIsBaseline() || !paramIsCreateAdjuster()) return true;
		
		int minAdjustDepth = Math.max((int)(Math.log(this.nut.size())/Math.log(NetworkAbstract.ZOOMOUT_DEFAULT)), 1);
		int maxAdjustDepth = Math.max(Math.max(this.nut.size()-1, 1), minAdjustDepth);
		int adjustDepth = Math.max(Math.max(depth1, depth2), minAdjustDepth);
		adjustDepth = Math.min(adjustDepth, maxAdjustDepth);
		Size size = this.nut.getOutputLayer().getSize();
		this.adjuster = new MatrixNetworkImpl(this.neuronChannel, this.nut.getActivateRef(), this.nut.getConvActivateRef(), this.idRef);
		this.adjuster.paramSetInclude(this);
		if (paramIsEntropyTrainer()) this.adjuster.setTrainer(new TaskTrainerLossEntropy());
		return new MatrixNetworkInitializer(adjuster).initialize(size, size, adjustDepth);
	}


	@Override
	MatrixNetworkImpl paramGetAdjuster() {return adjuster;}


	/**
	 * Creating matrix neural classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param rasterChannel raster channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static MatrixClassifier create(int neuronChannel, int rasterChannel, boolean isNorm) {
		Function activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		MatrixClassifier mac = new MatrixClassifier(neuronChannel, activateRef, contentActivateRef, null);
		mac.paramSetRasterChannel(rasterChannel);
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
class MatrixClassifierAbstract extends ClassifierAbstract {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier nut.
	 */
	protected MatrixNetworkImpl nut = null;


	/**
	 * Sample weight.
	 */
	@Deprecated
	private Matrix sampleWeight = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixClassifierAbstract(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, idRef);
		
		this.nut = new MatrixNetworkImpl(this.neuronChannel, activateRef, convActivateRef, idRef) {

			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected Object[] defineOutputErrorParams() {
				return sampleWeight != null ? new Object[] {sampleWeight} : null;
			}
			
		};
		try {
			this.config.putAll(this.nut.getConfig());
			this.nut.getConfig().putAll(this.config);
		} catch (Throwable e) {Util.trace(e);}
		if (paramIsEntropyTrainer()) this.nut.setTrainer(new TaskTrainerLossEntropy());
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
	public void reset() {
		super.reset();
		nut.reset();
		sampleWeight = null;
	}

	@Override
	void updateConfig() {
		super.updateConfig();
		nut.paramSetInclude(this);
	}

	
	@Override
	protected boolean initialize(Size inputSize1, Size outputSize1, FilterSpec filter1, int depth1, boolean dual1, Size outputSize2, int depth2) {
		if (!super.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputSize2, depth2)) return false;
		this.sampleWeight = null;
		
		int outputCount = this.outputClassMaps.get(0).size();
		int groupCount = getNumberOfGroups();
		Size outputCombSize = paramIsByColumn() ? new Size(groupCount, outputCount, 1) : new Size(outputCount, groupCount, 1);
		if (outputSize2 == null) {
			if (paramGetMiddleSize() <= 0) {
				if (!nut.initializeByDepth(inputSize1, outputCombSize, filter1, depth1, dual1, outputSize2, depth2))
					return false;
			}
			else {
				if (!nut.initialize(inputSize1, outputCombSize, filter1, depth1, dual1, outputSize2, depth2))
					return false;
			}
		}
		else {
			if (paramGetMiddleSize() <= 0) {
				if (!nut.initializeByDepth(inputSize1, outputSize1, filter1, depth1, dual1, outputCombSize, depth2))
					return false;
			}
			else {
				if (!nut.initialize(inputSize1, outputSize1, filter1, depth1, dual1, outputCombSize, depth2))
					return false;
			}
		}
		
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
	protected Matrix evaluate(Matrix input, Object...params) {
		updateConfig();
		return nut.evaluate0(input, params);
	}

	
//	@Override
//	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
//		Matrix sampleWeight = null;
//		if (paramIsSampleWeight()) {
//			this.sampleWeight = null;
//			List<Record> newsample = prelearn(sample);
//			learn(newsample);
//			learnVerify(newsample);
//			if (this.baseline != null) sampleWeight = paramIsByColumn() ? Softmax.softmaxByColumnInverse(this.baseline) : Softmax.softmaxByRowInverse(this.baseline);
//		}
//		List<Record> newsample = prelearn(sample);
//		this.sampleWeight = sampleWeight;
//		Error[] errors = learn(newsample);
//		learnVerify(newsample);
//		if (this.sampleWeight != null) this.baseline = null;
//		this.sampleWeight = null;
//		
//		if (paramGetAdjuster() != null) {
//			paramGetAdjuster().paramSetInclude(this);
//			List<Record> adjustSample = Util.newList(0);
//			for (Record record : newsample) {
//				Matrix output = evaluate(record.input());
//				if (output != null) adjustSample.add(new Record(output, record.output()));
//			}
//			errors = adjustSample.size() > 0 ? paramGetAdjuster().learn(adjustSample) : errors;
//		}
//		
//		NeuronValue[] errorArray = null;
//		for (Error error : errors) {
//			NeuronValue[] values = Matrix.extractValues(error.error());
//			errorArray = errorArray == null ? values : NeuronValue.concatArray(errorArray, values);
//		}
//		return errorArray;
//	}

	
	@Override
	protected Error[] learn(Iterable<Record> sample) {
		try {
			updateConfig();
			return nut.learn(sample);
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}
	

}
