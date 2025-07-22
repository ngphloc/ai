/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.gan;

import java.rmi.RemoteException;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is the default implementation of Generative Adversarial Network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GANImpl extends GANAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field of discrimination steps.
	 */
	public final static String DISCRIMINATE_STEPS_FIELD = "gan_discriminate_steps";
	
	
	/**
	 * Default value of discrimination steps.
	 */
	public final static int DISCRIMINATE_STEPS_DEFAULT = 1;

	
	/**
	 * Internal decoder.
	 */
	protected NetworkStandardImpl decoder = null;
	
	
	/**
	 * Internal adversarial decoder.
	 */
	protected AdversarialNetwork decodeAdv = null;

	
	/**
	 * Internal randomizer.
	 */
	protected Random learnRnd = new Random();

	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	public GANImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
		
		this.config.put(DISCRIMINATE_STEPS_FIELD, DISCRIMINATE_STEPS_DEFAULT);
		
		GeneratorStandard.fillConfig(this.config);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public GANImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public GANImpl(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
	@Override
	protected NetworkStandardImpl createDecoder() {
		GeneratorStandard<?> generator = new GeneratorStandard<Trainer>(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
				NeuronValue error = super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
				NeuronValue errorAdv = AdversarialNetwork.calcDecodedErrorAdv(outputNeuron, decodeAdv);
				return error.add(errorAdv);
//				return AdversarialNetwork.calcDecodedErrorAdv(outputNeuron, decodeAdv);
			}
			
		};
		generator.setParent(this);
		return generator;
	}

	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param nHiddenNeuronAdversarial number of adversarial hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronDecode, int[] nHiddenNeuronAdversarial) {
		if (xDim <= 0 || zDim <= 0) return false;
		
		this.decoder = createDecoder();
		if(!this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode)) return false;
		
		this.decodeAdv = createAdversarialNetwork();
		if (!this.decodeAdv.initialize(xDim, 1, nHiddenNeuronAdversarial)) return false;
		
		return true;
	}
	
	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronDecode) {
		return initialize(xDim, zDim, nHiddenNeuronDecode,
			nHiddenNeuronDecode != null && nHiddenNeuronDecode.length > 0? reverse(nHiddenNeuronDecode) : null);
	}
	
	
	@Override
	public void reset() throws RemoteException {
		decoder = null;
		decodeAdv = null;
	}
	
	
	/**
	 * Creating adversarial network.
	 * @return adversarial network.
	 */
	protected AdversarialNetwork createAdversarialNetwork() {
		return new AdversarialNetwork(neuronChannel, activateRef, idRef);
	}
	
	
	/**
	 * Checking whether this model is valid.
	 * @return whether this model is valid.
	 */
	public boolean isValid() {
		return (decoder != null && decodeAdv != null);
	}

	
	@Override
	protected NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		int disSteps = config.getAsInt(DISCRIMINATE_STEPS_FIELD);
		disSteps = disSteps < 1 ? 1 : disSteps;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : sample) {
				if (record == null) continue;
				
//				//Learning decoder.
//				try {
//					decoder.learn(randomizeDataZ(learnRnd), record.input, lr, terminatedThreshold, 1);
//				} catch (Throwable e) {Util.trace(e);}
//				
//				//Learning decoding adversarial network.
//				try {
//					decodeAdv.setPrevOutput(decodeAdv.evaluate(record)); //Evaluate real input.
//					//Learning decoding adversarial network.
//					NeuronValue[] generatedX = decoder.getOutputLayer().getOutput(); //Getting generated X
//					decodeAdv.learn(generatedX, lr, terminatedThreshold, 1);
//				} catch (Throwable e) {Util.trace(e);}
				
				//Learning decoding adversarial network.
				for (int k = 0; k < disSteps && decodeAdv != null; k++) {
					NeuronValue[] generatedX = null;
					try {
						//Getting generated X.
						Record newRecord = new Record(randomizeDataZ(learnRnd));
						generatedX = decoder.evaluate(newRecord);
					} catch (Throwable e) {Util.trace(e);}
	
					try {
						decodeAdv.setPrevOutput(decodeAdv.evaluate(record));
						//Learning decoding adversarial network.
						decodeAdv.learn(generatedX, lr, terminatedThreshold, 1);
					} catch (Throwable e) {Util.trace(e);}
				}
				
				try {
					//Learning decoder.
					error = decoder.learn(randomizeDataZ(learnRnd), record.input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "gan_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		} //End while
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "gan_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	protected NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		int disSteps = config.getAsInt(DISCRIMINATE_STEPS_FIELD);
		disSteps = disSteps < 1 ? 1 : disSteps;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			//Learning decoding adversarial network.
			for (int k = 0; k < disSteps && decodeAdv != null; k++) {
				List<Record> decodeAdvSample = Util.newList(0);
				int n = 0;
				for (Record record : sample) {
					if (decodeAdv.evaluateSetPrevOutputAccum(new Record(record.input))) n++;
					
					//Getting generated X.
					try {
						NeuronValue[] generatedX = decoder.evaluate(new Record(randomizeDataZ(learnRnd)));
						decodeAdvSample.add(new Record(generatedX));
					} catch (Throwable e) {Util.trace(e);}
				}
				
				NeuronValue[] prevOutput = decodeAdv.getPrevOutput();
				if (prevOutput != null && n > 0) {
					for (int i = 0; i < prevOutput.length; i++) prevOutput[i] = prevOutput[i].divide(n);
					decodeAdv.setPrevOutput(prevOutput);
				}
				//Learning decoding adversarial network.
				decodeAdv.learnOne(decodeAdvSample, lr, terminatedThreshold, 1);
				decodeAdv.setPrevOutput(null);
			}

			//Learning decoder.
			List<Record> decodeSample = Util.newList(0);
			for (Record record : sample) decodeSample.add(new Record(randomizeDataZ(learnRnd), record.input));
			error = decoder.learn(decodeSample, lr, terminatedThreshold, 1);
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "gan_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
				if (errorMean < terminatedThreshold) doStarted = false;
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		} //End while
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "gan_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	public synchronized G generate() throws RemoteException {
		NeuronValue[] dataZ = randomizeDataZ(learnRnd);
		NeuronValue[] genX = generateByZ(dataZ);
		
		G g = new G();
		g.z = dataZ;
		g.xgen = genX;
		return g;
	}

	
	@Override
	public synchronized G generateBest() throws RemoteException {
		if (decoder == null) return null;
		LayerStandard layer = decoder.getInputLayer();
		if (layer == null) return null;
		
		NeuronValue[] dataZ = new NeuronValue[layer.size()];
		NeuronValue zero = layer.newNeuronValue().zero();
		for (int i = 0; i < dataZ.length; i++) dataZ[i] = zero;
		
		NeuronValue[] genX = generateByZ(dataZ);
		if (genX == null || genX.length == 0) return null;
		
		G g = new G();
		g.z = dataZ;
		g.xgen = genX;
		
		return g;
	}

	
	/**
	 * Generate X data.
	 * @param dataZ Z data is encoded data.
	 * @return generated values (X data).
	 */
	protected NeuronValue[] generateByZ(NeuronValue...dataZ) {
		if (dataZ == null) return null;
		
		Record record = new Record();
		record.input = dataZ;
		try {
			return decoder.evaluate(record);
		} catch (Throwable e) {}
		
		return null;
	}

	
	/**
	 * Randomize Z data which is encoded data.
	 * @param rnd specific randomizer.
	 * @return Z data which is encoded data.
	 */
	protected NeuronValue[] randomizeDataZ(Random rnd) {
		if (decoder == null) return null;
		LayerStandard layer = decoder.getInputLayer();
		if (layer == null) return null;
		
		NeuronValue[] dataZ = new NeuronValue[layer.size()];
		NeuronValue zero = layer.newNeuronValue().zero();
		for (int i = 0; i < dataZ.length; i++) {
			dataZ[i] = zero.valueOf(Util.randomGaussian(rnd));
		}
		
		return dataZ;
	}

	
}
