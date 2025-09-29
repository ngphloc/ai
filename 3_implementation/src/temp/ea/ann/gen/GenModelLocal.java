/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.gen;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.ContentImpl;
import net.ea.ann.conv.ConvSupporter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterAssoc;
import net.ea.ann.conv.filter.FilterAssoc.PlainRaster;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.conv.filter.FilterFactoryImpl;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.LayerStandardImpl;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.gen.ConvGenModel;
import net.ea.ann.gen.ConvGenModelAbstract;
import net.ea.ann.gen.ConvGenSetting;
import net.ea.ann.gen.GenModelAbstract;
import net.ea.ann.gen.vae.ConvVAEImpl;
import net.ea.ann.gen.vae.VAEAbstract;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Point;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;

/**
 * This class represents a generative model with local optimization (local generative model).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class GenModelLocal extends GenModelAbstract implements ConvGenModel, ConvSupporter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Raster channel.
	 */
	protected int rasterChannel = 1;
	
	
	/**
	 * Z dimension.
	 */
	protected int zDim = 1;
	
	
	/**
	 * Original size of layers.
	 */
	protected Size size = Size.unit();
	
	
	/**
	 * Global size of layers.
	 */
	protected Size globalSize = new Size(1, 1, 1, 1);

	
	/**
	 * Local size of layers.
	 */
	protected Size localSize = new Size(1, 1, 1, 1);

	
	/**
	 * Global filter.
	 */
	protected Filter globalFilter = null;
	
	
	/**
	 * Global generative model.
	 */
	protected ConvGenModel globalModel = null;
	
	
	/**
	 * List of local generative models.
	 */
	protected List<ConvGenModel> localModels = Util.newList(0);
	
	
	/**
	 * Global error.
	 */
	protected NeuronValue[] globalError = null;
	
	
	/**
	 * Constructor with neuron channel, size, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size original size.
	 * @param idRef ID reference.
	 */
	protected GenModelLocal(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, null, idRef);
		this.rasterChannel = rasterChannel = ConvGenModelAbstract.fixRasterChannel(this.neuronChannel, rasterChannel);
		this.size = size;
	}
	

	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size original size.
	 */
	protected GenModelLocal(int neuronChannel, int rasterChannel, Size size) {
		this(neuronChannel, rasterChannel, size, new Id());
	}


	@Override
	public int getNeuronChannel() throws RemoteException {
		return neuronChannel;
	}


	@Override
	public void reset() throws RemoteException {
		globalFilter = null;
		globalModel = null;
		localModels.clear();
		globalError = null;
	}
	
	
	@Override
	public void setSetting(ConvGenSetting setting) throws RemoteException {
		if (setting == null) return;
		
		size.width = setting.width;
		size.height = setting.height;
		size.depth = setting.depth;
	}
	

	@Override
	public ConvGenSetting getSetting() throws RemoteException {
		ConvGenSetting setting = new ConvGenSetting();
		setting.width = size.width;
		setting.height = size.height;
		setting.depth = size.depth;
		
		return setting;
	}

	
	@Override
	public boolean initialize(int zDim, Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) throws RemoteException {
		this.zDim = zDim < 1 ? 1 : zDim;
		
		Size zoomOut = new Size(1, 1, 1, 1);
		if (convFilterArrays != null) {
			SizeZoom zoom = Filter.zoomRatioOfSimply(convFilterArrays);
			zoomOut.width = zoom.widthZoom;
			zoomOut.height = zoom.heightZoom;
			zoomOut.depth = zoom.depthZoom;
			zoomOut.time = zoom.timeZoom;
		}
		
		return initialize(zoomOut);
	}


	@Override
	public boolean initialize(int zDim, Filter[] convFilters, Filter[] deconvFilters) throws RemoteException {
		Filter[][] convFilterArrays = null, deconvFilterArrays = null;
		if (convFilters != null && convFilters.length > 0) convFilterArrays = new Filter[][] {convFilters};
		if (deconvFilters != null && deconvFilters.length > 0) deconvFilterArrays = new Filter[][] {deconvFilters};
		return initialize(zDim, convFilterArrays, deconvFilterArrays);
	}


	/**
	 * Initializing this generative model.
	 * @param size specified original size.
	 * @param zoomOut zooming-out ratio.
	 * @return true if initialization is successful.
	 */
	private boolean initialize(Size zoomOut) {
		zoomOut.width = zoomOut.width < 1 ? 1 : zoomOut.width;
		zoomOut.height = zoomOut.height < 1 ? 1 : zoomOut.height;
		zoomOut.depth = zoomOut.depth < 1 ? 1 : zoomOut.depth;
		if (size.width < zoomOut.width || size.height < zoomOut.height || size.depth < zoomOut.depth)
			return false;
		
		this.localSize = zoomOut;
		this.globalSize = new Size(size.width/zoomOut.width, size.height/zoomOut.height, size.depth/zoomOut.depth, size.time/zoomOut.time);
		try {
			this.globalFilter = getFilterFactory().mean(zoomOut.width, zoomOut.height, zoomOut.depth);
		} catch (Throwable e) {
			Util.trace(e);
			return false;
		}
		
		this.globalModel = createGlobalGenModel();
		try {
			this.globalModel.initialize(zDim, (Filter[])null, (Filter[])null);
		} catch (Throwable e) {
			Util.trace(e);
			return false;
		}
		
		this.localModels.clear();
		for (int z = 0; z < globalSize.depth; z++) {
			for (int y = 0; y < globalSize.height; y++) {
				for (int x = 0; x < globalSize.width; x++) {
					ConvGenModel localModel = createLocalGenModel(new Point(x, y, z, 0));
					this.localModels.add(localModel);
				}
			}
			
		}
		
		return true;
	}
	

	@Override
	public NeuronValueCreator getConvNeuronValueCreator() {
		return ContentImpl.create(rasterChannel, null, Size.unit());
	}

	
	@Override
	public FilterFactory getFilterFactory() {
		return new FilterFactoryImpl(getConvNeuronValueCreator());
	}
	
	
	/**
	 * Getting value creator.
	 * @return value creator.
	 */
	private NeuronValueCreator getNeuronValueCreator() {
		return new LayerStandardImpl(neuronChannel);
	}
	
	
	/**
	 * Getting global activation function.
	 * @return global activation function.
	 */
	private Function getGlobalActivateRef() {
		return globalModel instanceof VAEAbstract ? ((VAEAbstract)globalModel).getActivateRef() : null;
	}
	
	
	/**
	 * Creating global generative model.
	 * @return global generative model.
	 */
	protected ConvGenModel createGlobalGenModel() {
		return ConvVAEImpl.create(neuronChannel, globalSize, idRef);
	}
	
	
	/**
	 * Creating local generative model.
	 * @param block specified block.
	 * @return local generative model.
	 */
	protected ConvGenModel createLocalGenModel(Point block) {
		int index = block.z*globalSize.width*globalSize.height + block.y*globalSize.width + block.x; 
		ConvVAEImpl vae = new ConvVAEImpl(neuronChannel, localSize, idRef) {

			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NetworkStandardImpl createDecoder() {
				 return new NetworkStandardImpl(neuronChannel, activateRef, idRef) {
					 
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					protected NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
						NeuronValue error = super.calcOutputError(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
						if (globalError == null)
							return error;
						else
							return error.add(globalError[index]);
					}
					 
				 };
			}

		};
		
		int zoom = Math.max(globalFilter.getStrideWidth(), Math.max(globalFilter.getStrideHeight(), globalFilter.getStrideDepth()));
		vae.initialize(zDim < zoom ? 1 : zDim/zoom);
		return vae;
	}
	
	
	@Override
	public NeuronValue[] learnOne(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = getLearingRate();
		return bpLearnOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = getLearingRate();
		return bpLearn(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public synchronized NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = getLearingRate();
		return bpLearnRasterOne(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = getLearingRate();
		return bpLearnRaster(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	protected NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		throw new RuntimeException("Not implemented yet");
	}


	@Override
	protected NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		throw new RuntimeException("Not implemented yet");
	}


	/**
	 * Learning generative model by back propagate algorithm with specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected NeuronValue[] bpLearnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
//			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : subsample) {
				if (record == null) continue;
				
				NeuronValue[] input = null;
				if (record.input == null) {
					if (record.getRasterInput() != null) {
						input = record.getRasterInput().toNeuronValues(neuronChannel, size, isNorm());
						if (input == null) continue;
					}
					else
						continue;
				}
				else
					input = record.input;

				PlainRaster raster = new FilterAssoc(neuronChannel, getGlobalActivateRef(), globalFilter).apply3D(input, size);
				if (raster == null) continue;
				try {
					error /*= globalError*/ = globalModel.learnRasterOne(Arrays.asList(raster.toRaster(isNorm(), getDefaultAlpha())));
				} catch (Throwable e) {Util.trace(e);}
				
				int blockWidth = size.width / globalFilter.getStrideWidth();
				int blockHeight = size.height / globalFilter.getStrideHeight();
				int blockDepth = size.depth / globalFilter.getStrideDepth();
				for (int globalZ = 0; globalZ < globalSize.depth; globalZ++) {
					int localZ = 0;
					int zBlock = globalZ < blockDepth ? globalZ : blockDepth-1;
					localZ = zBlock*globalFilter.getStrideDepth();
					
					int globalZIndex = globalZ*globalSize.width*globalSize.height;
					for (int globalY = 0; globalY < globalSize.height; globalY++) {
						int localY = 0;
						int yBlock = globalY < blockHeight ? globalY : blockHeight-1;
						localY = yBlock*globalFilter.getStrideHeight();
						
						for (int globalX = 0; globalX < globalSize.width; globalX++) {
							int localX = 0;
							int xBlock = globalX < blockWidth ? globalX : blockWidth-1;
							localX = xBlock*globalFilter.getStrideWidth();
							NeuronValue[] localInput = RasterAssoc.extractRange3D(NeuronValue.class, input, size,
								new Cube(localX, localY, localZ, blockWidth, blockHeight, blockDepth));
							
							try {
								int globaIndex = globalZIndex + globalY*globalSize.width + globalX;
								localModels.get(globaIndex).learnOne(Arrays.asList(new Record(localInput, null)));
							} catch (Throwable e) {Util.trace(e);}
						}
					}
				}
				
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "gm_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0)
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
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "gm_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Learning generative model by back propagate algorithm with specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected NeuronValue[] bpLearn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return bpLearnOne(sample, learningRate, terminatedThreshold, maxIteration);
	}
	
	
	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] bpLearnRasterOne(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return bpLearnOne(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] bpLearnRaster(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return bpLearn(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Generating data.
	 * @param best flag to indicate whether best data is generated.
	 * @return generated data.
	 */
	private G generate0(boolean best) {
		Size size = new Size();
		size.width = localSize.width*globalSize.width;
		size.height = localSize.height*globalSize.height;
		size.depth = localSize.depth*globalSize.depth;
		size.time = localSize.time*globalSize.time;
		int length = size.width*size.height*size.depth;
		NeuronValue[] xgen = new NeuronValue[length];
		NeuronValue zero = getNeuronValueCreator().newNeuronValue().zero();
		for (int i = 0; i < xgen.length; i++) xgen[i] = zero;
		
		int blockWidth = size.width / globalFilter.getStrideWidth();
		int blockHeight = size.height / globalFilter.getStrideHeight();
		int blockDepth = size.depth / globalFilter.getStrideDepth();
		for (int globalZ = 0; globalZ < globalSize.depth; globalZ++) {
			int localZ = 0;
			int zBlock = globalZ < blockDepth ? globalZ : blockDepth-1;
			localZ = zBlock*globalFilter.getStrideDepth();
			
			int globalZIndex = globalZ*globalSize.width*globalSize.height;
			for (int globalY = 0; globalY < globalSize.height; globalY++) {
				int localY = 0;
				int yBlock = globalY < blockHeight ? globalY : blockHeight-1;
				localY = yBlock*globalFilter.getStrideHeight();
				
				for (int globalX = 0; globalX < globalSize.width; globalX++) {
					int localX = 0;
					int xBlock = globalX < blockWidth ? globalX : blockWidth-1;
					localX = xBlock*globalFilter.getStrideWidth();
					
					try {
						int globaIndex = globalZIndex + globalY*globalSize.width + globalX;
						G g = best ? localModels.get(globaIndex).generateBest() : localModels.get(globaIndex).generate();
						if (g == null) continue;
						RasterAssoc.copyRange3D(g.xgen, localSize, new Cube(0, 0, 0, localSize.width, localSize.height, localSize.depth),
								xgen, size, new Point(localX, localY, localZ, 0));
					} catch (Throwable e) {Util.trace(e);}
				}
			}
		}
		
		G g = new G();
		g.xgen = xgen;
		return g;
	}


	@Override
	public synchronized G generate() throws RemoteException {
		return generate0(false);
	}


	@Override
	public synchronized G generateBest() throws RemoteException {
		return generate0(true);
	}


	@Override
	public G generateRaster() throws RemoteException {
		try {
			G g = generate();
			if (g == null || g.xgen == null) return null;
			g.xgenUndefined = createRaster(g.xgen, size);
			return g;
		}
		catch (Throwable e) {Util.trace(e);}
		
		return null;
	}

	
	@Override
	public G generateRasterBest() throws RemoteException {
		try {
			G g = generateBest();
			if (g == null || g.xgen == null) return null;
			g.xgenUndefined = createRaster(g.xgen, size);
			return g;
		}
		catch (Throwable e) {Util.trace(e);}
		
		return null;
	}

	
	@Override
	public G generateRaster(NeuronValue...dataZ) throws RemoteException {
		throw new RuntimeException("GenModelLocal.generateRaster(NeuronValue...) not implemented yet");
	}


	@Override
	public G recoverRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException {
		if (raster == null) return null;
		
		NeuronValue[] dataX = raster.toNeuronValues(neuronChannel, size, isNorm());
		G g = recover(dataX, region, random, calcError);
		if (g == null) return null;
		
		if (g.xgen != null) {
			g.xgenUndefined = createRaster(g.xgen, size);
		}
		return g;
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
	private Raster createRaster(NeuronValue[] values, Size size) {
		return RasterAssoc.createRaster(values, neuronChannel, size, isNorm(), getDefaultAlpha());
	}

	
	@Override
	public int getRasterChannel() throws RemoteException {
		return rasterChannel;
	}

	
	/**
	 * Creating local generative model with neuron channel, size, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size original size.
	 * @param idRef ID reference.
	 * @return local generative model.
	 */
	public static GenModelLocal create(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		return new GenModelLocal(neuronChannel, rasterChannel, size, idRef);
				
	}
	
	
	/**
	 * Creating local generative model with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size original size.
	 * @return local generative model.
	 */
	public static GenModelLocal create(int neuronChannel, int rasterChannel, Size size) {
		return create(neuronChannel, rasterChannel, size, null);
	}


	/**
	 * Creating local generative model with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return local generative model.
	 */
	public static GenModelLocal create(int neuronChannel, int rasterChannel) {
		return create(neuronChannel, rasterChannel, Size.unit(), null);
	}


}
