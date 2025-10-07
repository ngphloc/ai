/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.gen.ConvGenModelAbstract;
import net.ea.ann.gen.gan.AdversarialNetwork;
import net.ea.ann.raster.Size;

/**
 * This class implements an extended Adversarial Variational Autoencoders (AVA).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class AVAExt extends AVA {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag of encoding supervision.
	 */
	public final static String SUPERVISE_ENCODE_FIELD = "avaext_supervise_encode";
	

	/**
	 * Default value of encoding supervision.
	 */
	public final static boolean SUPERVISE_ENCODE_DEFAULT = true;

	
	/**
	 * Flag of decoding supervision.
	 */
	public final static String SUPERVISE_DECODE_FIELD = "avaext_supervise_decode";
	

	/**
	 * Default value of decoding supervision.
	 */
	public final static boolean SUPERVISE_DECODE_DEFAULT = true;

	
	/**
	 * Flag to lean to improve encoding.
	 */
	public final static String LEAN_ENCODE_FIELD = "avaext_lean_encode";
	

	/**
	 * Default value for leaning to improve encoding.
	 */
	public final static boolean LEAN_ENCODE_DEFAULT = false;

	
	/**
	 * Flag to lean to improve decoding.
	 */
	public final static String LEAN_DECODE_FIELD = "avaext_lean_decode";
	

	/**
	 * Default value for leaning to improve decoding.
	 */
	public final static boolean LEAN_DECODE_DEFAULT = false;

	
	/**
	 * Adversarial network for encoding.
	 */
	protected AdversarialNetwork encodeAdv = null;
	
	
	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	public AVAExt(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, rasterChannel, size, idRef);
		
		this.config.put(SUPERVISE_ENCODE_FIELD, SUPERVISE_ENCODE_DEFAULT);
		this.config.put(SUPERVISE_DECODE_FIELD, SUPERVISE_DECODE_DEFAULT);

		this.config.put(LEAN_ENCODE_FIELD, LEAN_ENCODE_DEFAULT);
		this.config.put(LEAN_DECODE_FIELD, LEAN_DECODE_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	public AVAExt(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}

	
	/**
	 * Constructor with neuron channel, and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	public AVAExt(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel or depth.
	 */
	public AVAExt(int neuronChannel) {
		this(neuronChannel, neuronChannel, new Size(1, 1, 1, 1), null);
	}


	/**
	 * Creating encoder with Z dimension.
	 * @param zDim Z dimension.
	 * @return encoder with Z dimension.
	 */
	private NetworkStandardImpl createEncoder(int zDim) {
		return new GeneratorStandard<Trainer>(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
				NeuronValue error = calcEncodedError(outputNeuron);
				int index = outputLayer.indexOf(outputNeuron);
				//It is only necessary to focus on improve mean muX.
				if (index < 0 || index >= zDim) return error;
				
				NeuronValue errorAdv = AdversarialNetwork.calcDecodedErrorAdv(outputNeuron, encodeAdv);
				return error.add(errorAdv);
			}
			
		};
	}

	
	@Override
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode, int[] nHiddenNeuronAdversarial) {
		if (xDim <= 0 || zDim <= 0) return false;
		
		this.encoder = createEncoder(zDim);
		if(!this.encoder.initialize(xDim, isVarXDiagonal() ? 2*zDim : zDim*(zDim+1), nHiddenNeuronEncode)) return false;
		
		LayerStandard encodeLayer = this.encoder.getOutputLayer();
		this.muX = new NeuronStandard[zDim];
		for (int i = 0; i < zDim; i++) {
			this.muX[i] = encodeLayer.get(i);
		}
		
		this.varX = new NeuronStandard[zDim][];
		NeuronValue zero = encodeLayer.newNeuronValue().zero();
		NeuronStandard zeroNeuron = encodeLayer.newNeuron();
		zeroNeuron.setInput(zero);
		zeroNeuron.setOutput(zero);
		for (int i = 0; i < zDim; i++) {
			this.varX[i] = new NeuronStandard[zDim];
			for (int j = 0; j < zDim; j++) {
				if (isVarXDiagonal())
					this.varX[i][j] = i == j ? encodeLayer.get(zDim + i) : zeroNeuron;
				else
					this.varX[i][j] = encodeLayer.get(zDim + i*zDim + j);
			}
		}

		this.decoder = createDecoder();
		if(!this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode)) return false;
		
		if (nHiddenNeuronAdversarial == null || nHiddenNeuronAdversarial.length == 0) return true;
		
		if (isDecodeSupervise()) {
			this.decodeAdv = createAdversarialNetwork();
			if (!this.decodeAdv.initialize(xDim, 1, nHiddenNeuronAdversarial)) return false;
		}
		else
			this.decodeAdv = null;
		
		return true;
	}

	
	/**
	 * Initialize with X dimension and Z dimension as well as other specifications.
	 * @param xDim X dimension.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param nHiddenNeuronAdversarial number of adversarial hidden neurons.
	 * @param nHiddenNeuronAdversarial2 number of adversarial hidden neurons for encoding.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode, int[] nHiddenNeuronAdversarial, int[] nHiddenNeuronAdversarial2) {
		if (!this.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode, nHiddenNeuronAdversarial))
			return false;
		
		if (isEncodeSupervise()) {
			this.encodeAdv = createAdversarialNetwork();
			if (!this.encodeAdv.initialize(zDim, 1, nHiddenNeuronAdversarial2)) return false;
		}
		else
			this.encodeAdv = null;
		
		return true;
	}
	

	@Override
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode) {
		return this.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode, nHiddenNeuronEncode, nHiddenNeuronDecode);
	}

	
	@Override
	public void reset() throws RemoteException {
		super.reset();
		encodeAdv = null;
	}


	@Override
	protected NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			for (Record record : subsample) {
				if (record == null) continue;
				
				NeuronValue[] input = null;
				if (record.input == null) {
					if (conv != null) {
						try {
							//Learning convolutional network.
							if (ConvGenModelAbstract.hasLearning(conv)) conv.learnOneByOne(Arrays.asList(record), lr, terminatedThreshold, 1);
							conv.evaluate(record);
							input = conv.getFeatureFitChannel().getData();
						} catch (Throwable e) {Util.trace(e);}
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
					else if (record.getRasterInput() != null) {
						input = record.getRasterInput().toNeuronValues(rasterChannel, new Size(width, height, depth, time), isNorm());
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
				}
				else
					input = record.input;
				
				NeuronValue[] encodeError = null;
				try {
					//Learning encoder.
					encodeError = encoder.learn(input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				NeuronValue[] dataZ = randomizeDataZ(learnRnd);
				
				try {
					//Learning decoder.
					error = decoder.learn(dataZ, input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				NeuronValue[] generatedX = null;

				//Learning decoding adversarial network.
				if (decodeAdv != null && isDecodeSupervise()) {
					try {
						Record newRecord = new Record(input);
						decodeAdv.setPrevOutput(decodeAdv.evaluate(newRecord));
						
						//This code is to improve decoding ability.
						if (error != null && isLeanDecode()) decodeAdv.setExtraError(error);
						
						//Getting generated X.
						generatedX = decoder.evaluate(new Record(dataZ));
						
						//Learning decoding adversarial network.
						decodeAdv.learn(generatedX, lr, terminatedThreshold, 1);
					} catch (Throwable e) {Util.trace(e);}
					
					try {
						decodeAdv.setPrevOutput(null);
						decodeAdv.setExtraError(null);
					} catch (Throwable e) {Util.trace(e);}
				}
				else if (isEncodeSupervise()) {
					try {
						generatedX = decoder.evaluate(new Record(dataZ)); //Getting generated X
					} catch (Throwable e) {Util.trace(e);}
				}
				
				//Learning encoding adversarial network.
				if (encodeAdv != null && isEncodeSupervise()) {
					try {
						Record newRecord = new Record(getMuXValue());
						encodeAdv.setPrevOutput(encodeAdv.evaluate(newRecord));
						
						//This code is to improve encoding ability.
						if (encodeError != null && isLeanEncode()) encodeAdv.setExtraError(encodeError);

						//Getting generated muX.
						encoder.evaluate(new Record(generatedX));
						NeuronValue[] generatedMuX = getMuXValue(); //Generated muX
						
						//Learning encoding adversarial network. It is only necessary to focus on improving mean muX.
						encodeAdv.learn(generatedMuX, lr, terminatedThreshold, 1); //Focus on improving muX.
					} catch (Throwable e) {Util.trace(e);}
					
					try {
						encodeAdv.setPrevOutput(null);
						encodeAdv.setExtraError(null);
					} catch (Throwable e) {Util.trace(e);}
				}
				
				//It is unnecessary to learn the deconvolutional encoding network because the deconvolutional encoding network has neither full network nor reversed full network. 
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "avaext_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
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

		}
		
		//Fix something
		adjustVarX();

		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "avaext_backpropogate",
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
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			//Learning convolutional network.
			try {
				if (conv != null && ConvGenModelAbstract.hasLearning(conv))
					conv.learn(subsample, lr, terminatedThreshold, 1);
			} catch (Throwable e) {Util.trace(e);}

			List<Record> encodeSample = Util.newList(0);
			for (Record record : subsample) {
				if (record == null) continue;
				
				NeuronValue[] input = null;
				if (record.input == null) {
					if (conv != null) {
						try {
							conv.evaluate(record);
							input = conv.getFeatureFitChannel().getData();
						} catch (Throwable e) {Util.trace(e);}
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
					else if (record.getRasterInput() != null) {
						input = record.getRasterInput().toNeuronValues(rasterChannel, new Size(width, height, depth, time), isNorm());
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
				}
				else
					input = record.input;
				
				encodeSample.add(new Record(input, null));
			}

			if (encodeSample.size() == 0) break;
			NeuronValue[] encodeError = null;
			//Learning encoder.
			encodeError = encoder.learn(encodeSample, lr, terminatedThreshold, 1);
			
			//Learning decoder.
			List<Record> decodeSample = Util.newList(encodeSample.size());
			for (Record encodeRecord : encodeSample) decodeSample.add(new Record(randomizeDataZ(learnRnd), encodeRecord.input));
			error = decoder.learn(decodeSample, lr, terminatedThreshold, 1);

			//Learning decoding adversarial network.
			if (decodeAdv != null && isDecodeSupervise()) {
				List<Record> decodeAdvSample = Util.newList(decodeSample.size());
				int n = 0;
				for (Record decodeAdvRecord : decodeSample) {
					if (decodeAdv.evaluateSetPrevOutputAccum(new Record(decodeAdvRecord.output))) n++;
					
					try {
						//Getting generated X.
						NeuronValue[] generatedX = decoder.evaluate(new Record(decodeAdvRecord.input));
						decodeAdvSample.add(new Record(generatedX, null));
					} catch (Throwable e) {Util.trace(e);}
				}
				NeuronValue[] prevOutput = decodeAdv.getPrevOutput();
				if (prevOutput != null && n > 0) {
					for (int i = 0; i < prevOutput.length; i++) prevOutput[i] = prevOutput[i].divide(n);
					decodeAdv.setPrevOutput(prevOutput);
				}
				
				//This code is to improve decoding ability.
				if (error != null && isLeanDecode()) decodeAdv.setExtraError(error);
				
				//Learning decoding adversarial network.
				decodeAdv.learn(decodeAdvSample, lr, terminatedThreshold, 1);
				decodeAdv.setPrevOutput(null);
				decodeAdv.setExtraError(null);
			}
			
			//Learning encoding adversarial network.
			if (encodeAdv != null && isEncodeSupervise()) {
				try {
					//Taking one time because the encoder was learned by batch. 
					encodeAdv.setPrevOutput(encodeAdv.evaluate(new Record(getMuXValue())));
				} catch (Throwable e) {Util.trace(e);}

				List<Record> encodeAdvSample = Util.newList(decodeSample.size());
				for (Record decodeRecord : decodeSample) {
					try {
						//Getting generated muX.
						encoder.evaluate(new Record(decodeRecord.input));
						NeuronValue[] generatedMuX = getMuXValue(); //Generated muX
						encodeAdvSample.add(new Record(generatedMuX, null));
					} catch (Throwable e) {Util.trace(e);}
				}
				
				//This code is to improve encoding ability.
				if (encodeError != null && isLeanEncode()) encodeAdv.setExtraError(encodeError);
				
				//Learning encoding adversarial network. It is only necessary to focus on improve mean muX.
				encodeAdv.learn(encodeAdvSample, lr, terminatedThreshold, 1);
				encodeAdv.setPrevOutput(null);
				encodeAdv.setExtraError(null);
			}
			
			//It is unnecessary to learn the deconvolutional encoding network because the deconvolutional encoding network has neither full network nor reversed full network. 

			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "avaext_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
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

		}
		
		//Fix something
		adjustVarX();

		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "avaext_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}
	
	
//	/**
//	 * Making Z data from some numbers.
//	 * @param rNumbers some numbers.
//	 * @return Z data made from some numbers.
//	 */
//	private NeuronValue[] makeDataZ(NeuronValue[] rNumbers) {
//		NeuronValue[][] varXValue = getVarXValue();
//		varXValue = NeuronValue.matrixSqrt(varXValue);
//		NeuronValue[] dataZ = NeuronValue.matrixMultiply(varXValue, rNumbers);
//		NeuronValue[] muXValue = getMuXValue();
//		for (int i = 0; i < dataZ.length; i++) dataZ[i] = dataZ[i].add(muXValue[i]);
//		
//		return dataZ;
//	}
//
//	
//	/**
//	 * Making random numbers for Z data.
//	 * @param rnd specific randomizer.
//	 * @return random numbers for Z data.
//	 */
//	private NeuronValue[] randomNumbersForDataZ(Random rnd) {
//		NeuronValue[] rNumbers = new NeuronValue[muX.length];
//		NeuronValue zero = muX[0].getOutput().zero();
//		for (int i = 0; i < muX.length; i++) {
//			rNumbers[i] = zero.valueOf(Util.randomGaussian(rnd));
//		}
//		
//		return rNumbers;
//	}

	
	/**
	 * Checking whether to supervise encoding.
	 * @return whether to supervise encoding.
	 */
	private boolean isEncodeSupervise() {
		return config.getAsBoolean(SUPERVISE_ENCODE_FIELD);
	}


	/**
	 * Checking whether to supervise decoding.
	 * @return whether to supervise decoding.
	 */
	private boolean isDecodeSupervise() {
		return config.getAsBoolean(SUPERVISE_DECODE_FIELD);
	}

	
	/**
	 * Checking whether to lean to improve encoding.
	 * @return whether to lean to improve encoding.
	 */
	private boolean isLeanEncode() {
		return config.getAsBoolean(LEAN_ENCODE_FIELD);
	}


	/**
	 * Checking whether to lean to improve decoding.
	 * @return whether to lean to improve decoding.
	 */
	private boolean isLeanDecode() {
		return config.getAsBoolean(LEAN_DECODE_FIELD);
	}
	
	
	/**
	 * Creating extended Adversarial Variational Autoencoders with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return extended Adversarial Variational Autoencoders (AVAExt).
	 */
	public static AVAExt create(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		size.width = size.width < 1 ? 1 : size.width;
		size.height = size.height < 1 ? 1 : size.height;
		size.depth = size.depth < 1 ? 1 : size.depth;
		size.time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		rasterChannel = rasterChannel < neuronChannel ? neuronChannel : rasterChannel;
		return new AVAExt(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Creating extended Adversarial Variational Autoencoders with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return extended Adversarial Variational Autoencoders (AVAExt).
	 */
	public static AVAExt create(int neuronChannel, Size size, Id idRef) {
		return create(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Creating extended Adversarial Variational Autoencoders with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @return extended Adversarial Variational Autoencoders (AVAExt).
	 */
	public static AVAExt create(int neuronChannel, Size size) {
		return create(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Creating extended Adversarial Variational Autoencoders with neuron channel and raster channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return extended Adversarial Variational Autoencoders (AVAExt).
	 */
	public static AVAExt create(int neuronChannel, int rasterChannel) {
		return create(neuronChannel, rasterChannel, Size.unit(), null);
	}
	
	
	/**
	 * Creating extended Adversarial Variational Autoencoders with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return extended Adversarial Variational Autoencoders (AVAExt).
	 */
	public static AVAExt create(int neuronChannel) {
		return create(neuronChannel, neuronChannel, Size.unit(), null);
	}

	
}
