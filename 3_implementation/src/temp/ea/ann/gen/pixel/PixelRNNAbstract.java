/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.gen.pixel;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.ea.ann.conv.ConvSupporter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.conv.stack.StackNetworkAbstract;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.gen.ConvGenModelAbstract;
import net.ea.ann.gen.ConvGenSetting;
import net.ea.ann.gen.GenModelAbstract;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;
import temp.ea.ann.rnn.RecurrentNetworkImpl;
import temp.ea.ann.rnn.RecurrentNetwork.Layout;

/**
 * This class is an abstract implementation of generative pixel recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class PixelRNNAbstract extends GenModelAbstract implements PixelRNN, ConvSupporter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field name of input rows.
	 */
	public final static String INPUT_ROWS_FIELD = "pixrnn_input_rows";
	
	
	/**
	 * Default value of input rows.
	 */
	public final static int INPUT_ROWS_DEFAULT = 0;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Raster channel which is often larger than or equal to neuron channel.
	 */
	protected int rasterChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;
	
	
	/**
	 * Layer width.
	 */
	protected int width = 1;
	
	
	/**
	 * Layer height.
	 */
	protected int height = 1;

	
	/**
	 * Layer depth.
	 */
	protected int depth = 1;

	
	/**
	 * Layer time.
	 */
	protected int time = 1;

	
	/**
	 * Thick-stack property. In thick-stack mode (true), every stack in convolutional network should have more than one element layer.
	 */
	protected boolean thickStack = ConvGenSetting.THICK_STACK_DEFAULT;

	
	/**
	 * Recurrent neural network.
	 */
	protected RecurrentNetworkImpl rnn = null;
	
	
	/**
	 * Convolutional network which can have fully connected network (FCN) or reversed fully connected network (RFCN) or both even.
	 */
	protected StackNetworkAbstract conv = null;

	
	/**
	 * Deconvolutional network which have neither fully connected network (FCN) nor reversed fully connected network (RFCN).
	 * Therefore, the deconvolutional network has only main convolutional network.
	 */
	protected StackNetworkAbstract deconv = null;

	
	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel which is often larger than or equal to neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected PixelRNNAbstract(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, null, idRef);
		this.config.put(INPUT_ROWS_FIELD, INPUT_ROWS_DEFAULT);

		this.rasterChannel = rasterChannel = ConvGenModelAbstract.fixRasterChannel(this.neuronChannel, rasterChannel);
		
		this.width = size.width;
		this.height = size.height;
		this.depth = size.depth;
		this.time = size.time;
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected PixelRNNAbstract(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	protected PixelRNNAbstract(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected PixelRNNAbstract(int neuronChannel) {
		this(neuronChannel, neuronChannel, Size.unit(), null);
	}


	@Override
	public void setSetting(ConvGenSetting setting) throws RemoteException {
		if (setting == null) return;
		
		this.width = setting.width;
		this.height = setting.height;
		this.depth = setting.depth;
		this.time = setting.time;
		this.thickStack = setting.thickStack;
	}
	

	@Override
	public ConvGenSetting getSetting() throws RemoteException {
		ConvGenSetting setting = new ConvGenSetting();
		setting.width = this.width;
		setting.height = this.height;
		setting.depth = this.depth;
		setting.time = this.time;
		setting.thickStack = this.thickStack;
		
		return setting;
	}


	@Override
	public void reset() throws RemoteException {
		rnn = null;
		conv = null;
		deconv = null;
	}

	
	/**
	 * Initialize with filters.
	 * @param convFilterArrays arrays of convolutional filters. Filters in the same array have the same size.
	 * @param deconvFilterArrays deconvolutional filters. Filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) {
		Size size = new Size(width, height, depth, time);
		if (convFilterArrays != null && convFilterArrays.length > 0) {
			conv = createConvNetwork();
			if (conv == null)
				return false;
			else if (thickStack) {
				if (!conv.initialize(new Size(width, height, depth, time), convFilterArrays)) return false;
			}
			else if (convFilterArrays.length == 1) {
				if (!conv.initialize(new Size(width, height, depth, time), convFilterArrays[0])) return false;
			}
			else {
				if (!conv.initialize(new Size(width, height, depth, time), convFilterArrays)) return false;
			}
			
			try {
				size = conv.getFeatureSize();
			} catch (Throwable e) {Util.trace(e);}
		}
		else
			conv = null;
		
		int ratio = rasterChannel / neuronChannel;
		ratio = ratio < 1 ? 1 : ratio; 
		rnn = new RecurrentNetworkImpl(neuronChannel, activateRef, idRef);
		if (!rnn.initialize(ratio*size.width, size.height, Layout.parallel)) return false;
		
		if (deconvFilterArrays != null && deconvFilterArrays.length > 0) {
			Size deconvSize = new Size(width, height, depth, time);
			if (conv != null) {
				try {
					deconvSize = conv.getUnifiedOutputContentSize();
				} catch (Throwable e) {Util.trace(e);}
			}
			
			deconv = createDeconvNetwork();
			if (deconv == null)
				return false;
			else if (thickStack) {
				if (!deconv.initialize(deconvSize, deconvFilterArrays)) return false;
			}
			else if (deconvFilterArrays.length == 1) {
				if (!deconv.initialize(deconvSize, deconvFilterArrays[0])) return false;
			}
			else {
				if (!deconv.initialize(deconvSize, deconvFilterArrays)) return false;
			}
		}
		else
			deconv = null;
		
		return true;
	}

	
	@Override
	public boolean initialize(int zDim,
			Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) throws RemoteException {
		return initialize(convFilterArrays, deconvFilterArrays);
	}


	/**
	 * Initialize with filters.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Filter[] convFilters, Filter[] deconvFilters) {
		Filter[][] convFilterArrays = null, deconvFilterArrays = null;
		if (convFilters != null && convFilters.length > 0) convFilterArrays = new Filter[][] {convFilters};
		if (deconvFilters != null && deconvFilters.length > 0) deconvFilterArrays = new Filter[][] {deconvFilters};
		return initialize(convFilterArrays, deconvFilterArrays);
	}
	
	
	@Override
	public boolean initialize(int zDim, Filter[] convFilters, Filter[] deconvFilters) throws RemoteException {
		return initialize(convFilters, deconvFilters);
	}


	/**
	 * Initialize with convolutional filters.
	 * @param convFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Filter[] convFilters) {
		return this.initialize(convFilters, (Filter[])null);
	}
	
	
	/**
	 * Initialize without other specifications.
	 * @return true if initialization is successful.
	 */
	public boolean initialize() {
		return this.initialize((Filter[])null);
	}

	
	/**
	 * Initialize with zooming out ratio.
	 * @param zoomOutRatio zooming out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zoomOutRatio) {
//		try {
			Filter[] convFilters = null;
			Filter[] deconvFilters = null;
			if (zoomOutRatio > 1) {
				FilterFactory factory = getFilterFactory();
				convFilters = new Filter[] {factory.zoomOut(zoomOutRatio, zoomOutRatio, zoomOutRatio)};
				deconvFilters = new Filter[] {factory.zoomIn(zoomOutRatio, zoomOutRatio, zoomOutRatio)};
			}
			
			return this.initialize(convFilters, deconvFilters);
//		}
//		catch (RemoteException e) {Util.trace(e);}
		
//		return false;
	}

	
	/**
	 * Creating convolutional neural network.
	 * @return convolutional neural network.
	 */
	protected StackNetworkAbstract createConvNetwork() {
		return ConvGenModelAbstract.defaultConvNetwork(this, isNorm(), idRef);
	}
	
	
	/**
	 * Creating deconvolutional neural network.
	 * @return deconvolutional neural network.
	 */
	protected StackNetworkAbstract createDeconvNetwork() {
		return createConvNetwork();
	}


	/**
	 * Converting feature to X.
	 * @param feature feature array.
	 * @return X array.
	 */
	private NeuronValue[] convertFeatureToX(NeuronValue[] feature) {
		if (feature == null || feature.length == 0 || rasterChannel == neuronChannel)
			return feature;
		else
			return feature[0].flatten(feature, neuronChannel);
	}
	
	
	/**
	 * Converting X to feature.
	 * @param dataX X array.
	 * @return feature array.
	 */
	private NeuronValue[] convertXToFeature(NeuronValue[] dataX) {
		if (dataX == null || dataX.length == 0 || rasterChannel == neuronChannel)
			return dataX;
		else
			return dataX[0].aggregate(dataX, rasterChannel);
	}

	
	@Override
	public NeuronValueCreator getConvNeuronValueCreator() {
		return createConvNetwork().newStack(Size.unit());
	}

	
	@Override
	public FilterFactory getFilterFactory() {
		return createConvNetwork().getFilterFactory();
	}

	
	@Override
	public NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learnRasterOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learnRaster(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning neural network one-by-one record over sample.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (rnn == null || rnn.size() < 1) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : sample) {
				if (record == null) continue;
				
				NeuronValue[] input = null;
				if (record.input == null) {
					//Learning convolutional encoding network.
					if (conv != null) {
						try {
							conv.learnOne(Arrays.asList(record), lr, terminatedThreshold, 1);
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
				
				int w = rnn.get(0).getBackbone().size(); //Raster width.
				for (int t = 0; t < rnn.size(); t++) {
					final int startIndex = t*w;
					if (startIndex >= input.length) break;
					
					Map<Integer, NeuronValue[]> stateInput = Util.newMap(0);
					Map<Integer, NeuronValue[]> stateOutput = Util.newMap(0);
					int r = getInputRows();
					if (r > 0) {
						Map<Integer, NeuronValue[]> map = t < r ? stateInput : stateOutput;
						for (int i = 0; i < w; i++) {
							int index = startIndex + i;
							if (index >= input.length) break;
							map.put(i, new NeuronValue[] {input[index]});
						}
					}
					else if (t == 0) {
						stateInput.put(0, new NeuronValue[] {input[0]});
						for (int i = 1; i < w; i++) {
							if (i < input.length) stateOutput.put(i, new NeuronValue[] {input[i]});
						}
					}
					else {
						for (int i = 0; i < w; i++) {
							int index = startIndex + i;
							if (index >= input.length) break;
							stateOutput.put(i, new NeuronValue[] {input[index]});
						}
					}

					rnn.get(t).learn(stateInput, stateOutput, lr, terminatedThreshold, 1);
				}
				
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "pixrnn_backpropogate",
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

		}
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "pixrnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	protected NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		System.out.println("Method PixelRNNAbstract.learn(Iterable<Record>, double, double, int) calls method PixelRNNAbstract.learn(Iterable<Record>, double, double, int) instead because pixel recurrent neural network do not support batch learning.");
		return learnOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learnRasterOne(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return learnOne(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learnRaster(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return learn(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
	}

	
	@Override
	public synchronized G generate() throws RemoteException {
		if (rnn == null || rnn.size() == 0) return null;
		NeuronValue zero = rnn.get(0).getInputLayer().newNeuronValue().zero();
		int r = getInputRows();
		int w = rnn.get(0).getBackbone().size();
		NeuronValue[] dataX = null;
		if (r <= 0) {
			dataX = new NeuronValue[] {zero.valueOf(Util.randomGaussian(new Random()))};
			rnn.evaluate(dataX);
		}
		else {
			dataX = new NeuronValue[w*r];
			Random rnd = new Random();
			for (int i = 0; i < dataX.length; i++) dataX[i] = zero.valueOf(Util.randomGaussian(rnd));
			setXDataToRNN(dataX);
			rnn.evaluate(r);
		}
		
		G g = new G();
		g.z = g.x = dataX;
		g.xgen = extractXDataFromRNN();
		return g;
	}
	

	@Override
	public G generateBest() throws RemoteException {
		return generate();
	}


	/**
	 * Converting recurrent network to X data.
	 * @return X data converted from recurrent network.
	 */
	private NeuronValue[] extractXDataFromRNN() {
		if (rnn == null || rnn.size() == 0) return null;
		
		int w = rnn.get(0).getBackbone().size();
		NeuronValue[] dataX = new NeuronValue[w*rnn.size()];
		for (int t = 0; t < rnn.size(); t++) {
			NetworkStandardImpl state = rnn.get(t);
			List<LayerStandard> bone = state.getBackbone();
			int nIndex = t*w;
			for (int i = 0; i < w; i++) {
				dataX[nIndex + i] = bone.get(i).get(0).getOutput();
			}
		}
		
		return dataX;
	}
	
	
	/**
	 * Setting X data to recurrent neural network.
	 * @param dataX specified X data.
	 * @return true if setting is successful.
	 */
	private boolean setXDataToRNN(NeuronValue[] dataX) {
		if (rnn == null || rnn.size() == 0 || dataX == null || dataX.length == 0) return false;
		
		int w = rnn.get(0).getBackbone().size();
		for (int t = 0; t < rnn.size(); t++) {
			int startIndex = t*w;
			if (startIndex >= dataX.length) break;
			
			List<LayerStandard> backbone = rnn.get(t).getBackbone();
			for (int i = 0; i < w; i++) {
				int index = startIndex + i;
				if (index >= dataX.length) break;
				backbone.get(i).setInput(dataX[index]);
				backbone.get(i).setOutput(dataX[index]);
			}
		}

		return true;
	}
	
	
	/**
	 * Converting X data to raster.
	 * @param dataX X data.
	 * @return raster converted from X data.
	 */
	private Raster convertXDataToRaster(NeuronValue[] dataX) {
		if (dataX == null)
			return null;
		else if (conv == null && deconv == null)
			return createRaster(dataX);
		else if (conv != null && deconv == null)
			return conv.createRaster(convertXToFeature(dataX));
		else
			return deconv.createRaster(convertXToFeature(dataX));
	}
	
	
	@Override
	public synchronized G generateRaster() throws RemoteException {
		try {
			G g = generate();
			if (g == null || g.xgen == null) return null;
			g.xgenUndefined = convertXDataToRaster(g.xgen);
			return g;
		}
		catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
	@Override
	public G generateRasterBest() throws RemoteException {
		return generateRaster();
	}


	@Override
	public G generateRaster(NeuronValue...dataZ) throws RemoteException {
		throw new RuntimeException("Method PixelRNNAbstract.generateRaster((NeuronValue...) not implemented yet");
	}


	@Override
	public G recoverRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException {
		throw new RuntimeException("Method PixelRNNAbstract.recoverRaster((Raster, Cube) not implemented yet");
	}


	@Override
	public G reproduceRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException {
		return ConvGenModelAbstract.reproduceRaster(this, raster, region, random, calcError);
	}


	/**
	 * Converting neuron values to raster. This method is only used in case that there is no convolutional network.
	 * @param values neuron values. This is X data, not feature.
	 * @return converted raster.
	 */
	private Raster createRaster(NeuronValue[] values) {
		return RasterAssoc.createRaster(convertXToFeature(values), rasterChannel, new Size(width, height, depth, time),
			isNorm(), getDefaultAlpha());
	}

	
	@Override
	public int getRasterChannel() throws RemoteException {
		return rasterChannel;
	}

	
	/**
	 * Getting input rows.
	 * @return input rows.
	 */
	private int getInputRows() {
		int r = config.getAsInt(INPUT_ROWS_FIELD);
		r = Math.min(r, rnn.size());
		return r < 0 ? 0 : r;
	}
	
	
}
