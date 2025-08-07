/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.Util;
import net.ea.ann.gen.GenModel.G;
import net.ea.ann.gen.gan.ConvGANImpl;
import net.ea.ann.gen.pixel.PixelRNNImpl;
import net.ea.ann.gen.vae.AVAExt;
import net.ea.ann.gen.vae.ConvVAEImpl;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterWrapper;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;

/**
 * This class is an associator of convolutional generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvGenModelAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional generative model.
	 */
	protected ConvGenModel convGM = null;
	
	
	/**
	 * Batch learning one-by-one.
	 */
	protected boolean learnOne = false;
	
	
	/**
	 * Constructor with convolutional generative model and batch learning mode.
	 * @param convGM convolutional generative model.
	 * @param learnOne one-by-one learning mode.
	 */
	public ConvGenModelAssoc(ConvGenModel convGM, boolean learnOne) {
		this.convGM = convGM;
		this.learnOne = learnOne;
	}
	
	
	/**
	 * Constructor with convolutional generative model.
	 * @param convGM convolutional generative model.
	 */
	public ConvGenModelAssoc(ConvGenModel convGM) {
		this(convGM, false);
	}

	
	/**
	 * Setting parameter: learning one-by-one mode.
	 * @param learnOne one-by-one learning mode.
	 * @return this associator.
	 */
	public ConvGenModelAssoc setParamLearnOne(boolean learnOne) {
		this.learnOne = learnOne;
		return this;
	}
	
	
	/**
	 * Getting configuration.
	 * @return model configuration.
	 */
	public NetworkConfig getConfig() {
		if (!(convGM instanceof Network)) return null;
		try {
			return ((Network)convGM).getConfig();
		} catch (Throwable e) {Util.trace(e);}
		
		return null;
	}
	
	
	/**
	 * Getting neuron channel.
	 * @return neuron channel.
	 */
	public int getNeuronChannel() {
		try {
			return convGM.getNeuronChannel();
		} catch (Throwable e) {Util.trace(e);}
		
		return 0;
	}
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	public boolean isNorm() {
		NetworkConfig config = getConfig();
		if (config != null && config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}

	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	public int getDefaultAlpha() {
		NetworkConfig config = getConfig();
		if (config != null && config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}

	
	/**
	 * Getting size of layer in model.
	 * @return size of layer in model.
	 */
	public Size getSize() {
		ConvGenSetting setting = null;
		try {
			setting = convGM.getSetting();
			return new Size(setting.width, setting.height, setting.depth, setting.time);
		} catch (Throwable e) {Util.trace(e);}
		
		return null;
	}
	
	
	/**
	 * Generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @return list of generated rasters.
	 */
	public List<Raster> genRasters(Iterable<Raster> sample, int nGens) {
		List<Raster> result = Util.newList(0);
		try {
			if (learnOne)
				convGM.learnRasterOne(sample);
			else
				convGM.learnRaster(sample);
		}
		catch (Exception e) {
			Util.trace(e);
			return result;
		}
		
		for (int i = 0; i < nGens; i++) {
			try {
				G g = nGens <= 1 ? convGM.generateRasterBest() : convGM.generateRaster();
				Raster raster = g != null ? g.getXGenRaster() : null;
				if (raster != null) result.add(raster);
			} catch (Exception e) {Util.trace(e);}
		}
		return result;
	}
	
	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize) {
		if (new ConvGenInitializer(convGM).initialize(sample, zDim, convFilters, deconvFilters, minSize))
			return genRasters(sample, nGens);
		else
			return Util.newList(0);
	}

	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim, Filter[] convFilters, Filter[] deconvFilters) {
		return initGenRasters(sample, nGens, zDim, convFilters, deconvFilters, null);
	}
	
	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param minSize minimum size.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim, Filter[] convFilters, Size minSize) {
		return initGenRasters(sample, nGens, zDim, convFilters, null, minSize);
	}

	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim, Filter[] convFilters) {
		return initGenRasters(sample, nGens, zDim, convFilters, null, null);
	}

	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim, SizeZoom zoomOut, Size minSize) {
		if (new ConvGenInitializer(convGM).initialize(sample, zDim, zoomOut, minSize))
			return genRasters(sample, nGens);
		else
			return Util.newList(0);
	}
	
	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim, SizeZoom zoomOut) {
		return initGenRasters(sample, nGens, zDim, zoomOut, null);
	}

	
	/**
	 * Initial generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRasters(Iterable<Raster> sample, int nGens, int zDim) {
		if (new ConvGenInitializer(convGM).initialize(sample, zDim))
			return genRasters(sample, nGens);
		else
			return Util.newList(0);
	}

	
	/**
	 * Initial generating rasters from sample given 2D feature extractor.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @return list of generated rasters.
	 */
	public List<Raster> initGenRastersFeatureExtractor2D(Iterable<Raster> sample, int nGens, int zDim, SizeZoom zoomOut, Size minSize) {
		if (new ConvGenInitializer(convGM).initializeFeatureExtractor2D(sample, zDim, zoomOut, minSize))
			return genRasters(sample, nGens);
		else
			return Util.newList(0);
	}
	
	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @param memory flag to indicate whether to store recovery results.
	 * @return recovery results.
	 */
	private List<G> recoverRasters(String gmName, Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir, boolean memory) {
		List<G> results = Util.newList(0);
		int index = 0;
		for (Raster recoverRaster : rasters) {
			index++;
			try {
				ConvGenModel clonedGM = (ConvGenModel)net.ea.ann.core.Util.cloneBySerialize(convGM);
				if (learnOne)
					clonedGM.learnRasterOne(Arrays.asList(recoverRaster));
				else
					clonedGM.learnRaster(Arrays.asList(recoverRaster));
				for (int k = 0; k < nGens; k++) {
					G g = clonedGM.recoverRaster(recoverRaster, region, randomGen, true);
					if ((g == null) || (g.xgenUndefined == null) || !(g.xgenUndefined instanceof Raster))
						continue;
					
					if (memory) {
						g.tag = recoverRaster;
						results.add(g);
					}
					
					if (recoverDir != null) {
						String name = gmName != null && !gmName.isEmpty() ? gmName + "." : "";
						String genName = recoverRaster instanceof RasterWrapper ? name + ((RasterWrapper)recoverRaster).getNamePlain() : name;
						genName = genName.isEmpty() ? Util.NONAME : genName;
						Path path = RasterAssoc.genDefaultPath(recoverDir, genName, recoverRaster.getDefaultFormat(), index);
						((Raster)g.xgenUndefined).save(path);
					}
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		
		return results;
	}

	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return recoverRasters(gmName, rasters, region, randomGen, nGens, null, true);
	}
	
	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, boolean randomGen, int nGens) {
		return recoverRasters(gmName, rasters, null, randomGen, nGens, null, true);
	}

	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, Cube region, int nGens) {
		return recoverRasters(gmName, rasters, region, true, nGens, null, true);
	}

	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, int nGens) {
		return recoverRasters(gmName, rasters, null, true, nGens, null, true);
	}

	
	/**
	 * Recovering rasters
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return recoverRasters("", rasters, region, randomGen, nGens);
	}

	
	/**
	 * Recovering rasters
	 * @param rasters recovering rasters.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(Iterable<Raster> rasters, boolean randomGen, int nGens) {
		return recoverRasters("", rasters, null, randomGen, nGens);
	}

	
	/**
	 * Recovering rasters
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(Iterable<Raster> rasters, Cube region, int nGens) {
		return recoverRasters("", rasters, region, true, nGens);
	}

	
	/**
	 * Recovering rasters
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(Iterable<Raster> rasters, int nGens) {
		return recoverRasters("", rasters, null, true, nGens);
	}

	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return recoverRasters(gmName, rasters, region, randomGen, nGens, recoverDir, true);
	}
	
	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, boolean randomGen, int nGens, Path recoverDir) {
		return recoverRasters(gmName, rasters, null, randomGen, nGens, recoverDir, true);
	}

	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, Cube region, int nGens, Path recoverDir) {
		return recoverRasters(gmName, rasters, region, true, nGens, recoverDir, true);
	}

	
	/**
	 * Recovering rasters
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> recoverRasters(String gmName, Iterable<Raster> rasters, int nGens, Path recoverDir) {
		return recoverRasters(gmName, rasters, null, true, nGens, recoverDir, true);
	}

	
	/**
	 * Recovering rasters with only saving.
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void recoverRastersOnlySave(String gmName, Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		recoverRasters(gmName, rasters, region, randomGen, nGens, recoverDir, false);
	}

	
	/**
	 * Recovering rasters with only saving.
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void recoverRastersOnlySave(String gmName, Iterable<Raster> rasters, boolean randomGen, int nGens, Path recoverDir) {
		recoverRasters(gmName, rasters, null, randomGen, nGens, recoverDir, false);
	}

	
	/**
	 * Recovering rasters with only saving.
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void recoverRastersOnlySave(String gmName, Iterable<Raster> rasters, Cube region, int nGens, Path recoverDir) {
		recoverRasters(gmName, rasters, region, true, nGens, recoverDir, false);
	}

	
	/**
	 * Recovering rasters with only saving.
	 * @param gmName name of generative model.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void recoverRastersOnlySave(String gmName, Iterable<Raster> rasters, int nGens, Path recoverDir) {
		recoverRasters(gmName, rasters, null, true, nGens, recoverDir, false);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @param memory flag to indicate whether to store recovery results.
	 * @return recovery results.
	 */
	private List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir, boolean memory) {
		if (!new ConvGenInitializer(convGM).initialize(sample, zDim, zoomOut, minSize))
			return Util.newList(0);
		
		try {
			if (learnOne)
				convGM.learnRasterOne(sample);
			else
				convGM.learnRaster(sample);
		}
		catch (Exception e) {
			Util.trace(e);
			return Util.newList(0);
		}

		return recoverRasters(gmName, rasters, region, randomGen, nGens, recoverDir, memory);
	}
	
	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, zoomOut, minSize,
				rasters, region, randomGen, nGens, null, true);
	}
	
	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, zoomOut, null,
				rasters, region, randomGen, nGens, null, true);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, null, null,
				rasters, region, randomGen, nGens, null, true);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, null, null,
				rasters, null, true, nGens, null, true);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, zoomOut, minSize,
				rasters, region, randomGen, nGens, recoverDir, true);
	}
	
	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, zoomOut, null,
				rasters, region, randomGen, nGens, recoverDir, true);
	}


	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, null, null,
				rasters, region, randomGen, nGens, recoverDir, true);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, null, null,
				rasters, null, true, nGens, recoverDir, true);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters("", sample, zDim, zoomOut, minSize,
				rasters, region, randomGen, nGens);
	}
	
	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, SizeZoom zoomOut,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters("", sample, zDim, zoomOut,
				rasters, region, randomGen, nGens);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, int nGens) {
		return initRecoverRasters("", sample, zDim, (SizeZoom)null,
				rasters, null, true, nGens);
	}

	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, zoomOut, minSize,
				rasters, region, randomGen, nGens, recoverDir);
	}
	
	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, SizeZoom zoomOut,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, zoomOut,
				rasters, region, randomGen, nGens, recoverDir);
	}

	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, (SizeZoom)null,
				rasters, region, randomGen, nGens, recoverDir);
	}

	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, (SizeZoom)null,
				rasters, null, true, nGens, recoverDir);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, zoomOut, minSize,
				rasters, region, randomGen, nGens, recoverDir, false);
	}
	
	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, zoomOut, null,
				rasters, region, randomGen, nGens, recoverDir, false);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, null, null,
				rasters, null, true, nGens, recoverDir, false);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @param memory flag to indicate whether to store recovery results.
	 * @return recovery results.
	 */
	private List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir, boolean memory) {
		if (!new ConvGenInitializer(convGM).initialize(sample, zDim, convFilters, deconvFilters, minSize))
			return Util.newList(0);
		
		try {
			if (learnOne)
				convGM.learnRasterOne(sample);
			else
				convGM.learnRaster(sample);
		}
		catch (Exception e) {
			Util.trace(e);
			return Util.newList(0);
		}

		return recoverRasters(gmName, rasters, region, randomGen, nGens, recoverDir, memory);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, deconvFilters, minSize,
				rasters, region, randomGen, nGens, null, true);
	}
	
	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, deconvFilters, null,
				rasters, region, randomGen, nGens, null, true);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, null, null,
				rasters, region, randomGen, nGens, null, true);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, int nGens) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, null, null,
				rasters, null, true, nGens, null, true);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, deconvFilters, minSize,
				rasters, region, randomGen, nGens, recoverDir, true);
	}
	
	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, deconvFilters, null,
				rasters, region, randomGen, nGens, recoverDir, true);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, null, null,
				rasters, region, randomGen, nGens, recoverDir, true);
	}


	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, int nGens, Path recoverDir) {
		return initRecoverRasters(gmName, sample, zDim, convFilters, null, null,
				rasters, null, true, nGens, recoverDir, true);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters("", sample, zDim, convFilters, deconvFilters, minSize,
				rasters, region, randomGen, nGens);
	}
	
	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters("", sample, zDim, convFilters, deconvFilters,
				rasters, region, randomGen, nGens);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters("", sample, zDim, convFilters,
				rasters, region, randomGen, nGens);
	}

	
	/**
	 * Initial recovering rasters.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRasters("", sample, zDim, (Filter[])null,
				rasters, region, randomGen, nGens);
	}

	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, convFilters, deconvFilters, minSize,
				rasters, region, randomGen, nGens, recoverDir);
	}
	
	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, convFilters, deconvFilters,
				rasters, region, randomGen, nGens, recoverDir);
	}

	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, convFilters,
				rasters, region, randomGen, nGens, recoverDir);
	}

	
	/**
	 * Initial recovering rasters
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @return recovery results.
	 */
	public List<G> initRecoverRasters(Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, int nGens, Path recoverDir) {
		return initRecoverRasters("", sample, zDim, convFilters,
				rasters, null, true, nGens, recoverDir);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, convFilters, deconvFilters, minSize,
				rasters, region, randomGen, nGens, recoverDir, false);
	}
	
	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters, Filter[] deconvFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, convFilters, deconvFilters, null,
				rasters, region, randomGen, nGens, recoverDir, false);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param convFilters convolutional filters.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim, Filter[] convFilters,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, convFilters, null, null,
				rasters, region, randomGen, nGens, recoverDir, false);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 */
	public void initRecoverRastersOnlySave(String gmName, Iterable<Raster> sample, int zDim,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir) {
		initRecoverRasters(gmName, sample, zDim, null, null, null,
				rasters, region, randomGen, nGens, recoverDir, false);
	}

	
	/**
	 * Initial recovering rasters
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @param recoverDir recovering directory.
	 * @param memory flag to indicate whether to store recovery results.
	 * @return recovery results.
	 */
	private List<G> initRecoverRastersFeatureExtractor2D(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens, Path recoverDir, boolean memory) {
		if (!new ConvGenInitializer(convGM).initializeFeatureExtractor2D(sample, zDim, zoomOut, minSize))
			return Util.newList(0);
		
		try {
			if (learnOne)
				convGM.learnRasterOne(sample);
			else
				convGM.learnRaster(sample);
		}
		catch (Exception e) {
			Util.trace(e);
			return Util.newList(0);
		}

		return recoverRasters(gmName, rasters, region, randomGen, nGens, recoverDir, memory);
	}
	
	
	/**
	 * Initial recovering rasters.
	 * @param gmName name of generative model.
	 * @param sample raster sample.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param zoomOut zoom out ratio.
	 * @param minSize minimum size.
	 * @param rasters recovering rasters.
	 * @param region specified region.
	 * @param randomGen flag to indicate whether to randomize generated rasters.
	 * @param nGens number of generated rasters.
	 * @return recovery results.
	 */
	public List<G> initRecoverRastersFeatureExtractor2D(String gmName, Iterable<Raster> sample, int zDim, SizeZoom zoomOut, Size minSize,
			Iterable<Raster> rasters, Cube region, boolean randomGen, int nGens) {
		return initRecoverRastersFeatureExtractor2D(gmName, sample, zDim, zoomOut, minSize,
				rasters, region, randomGen, nGens, null, true);
	}

	
	/**
	 * This class is a wrapper of generative model.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	private static class GM implements Serializable, Cloneable {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Generative model.
		 */
		public ConvGenModel gm = null;
		
		/**
		 * Name of generative model.
		 */
		public String name = null;
		
		/**
		 * Constructor with generative model and its name.
		 * @param gm specified generative model.
		 * @param name specified name.
		 */
		public GM(ConvGenModel gm, String name) {
			this.gm = gm;
			this.name = name;
		}

		@Override
		public Object clone() throws CloneNotSupportedException {
			return super.clone();
		}
		
	}


	/**
	 * Test of generation.
	 * @param in input stream.
	 * @param out output stream.
	 * @throws RemoteException if any error raises.
	 */
	public static void gen(InputStream in, OutputStream out) throws RemoteException {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);

		boolean load3d = false;
		printer.print("Load 3D (true) or load 2D (false) (false is default):");
		try {
			load3d = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println(load3d ? "Load 3D\n" : "Load 2D\n");
		
		int defaultZoomOut = 1;
		int zoomOut = defaultZoomOut;
		printer.print("Zoom ratio (1, 2, 3,...) (default 1):");
		try {
			zoomOut = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(zoomOut)) zoomOut = defaultZoomOut;
		if (zoomOut < defaultZoomOut) zoomOut = defaultZoomOut;
		printer.println("Zoom ratio is " + zoomOut + "\n");

		int defaultNeuronChannel = 1;
		int neuronChannel = defaultNeuronChannel;
		printer.print("Neuron channel (1, 2, 3, 4) (default 1):");
		try {
			neuronChannel = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(neuronChannel)) neuronChannel = defaultNeuronChannel;
		if (neuronChannel < defaultNeuronChannel) neuronChannel = defaultNeuronChannel;
		printer.println("Neuron channel is " + neuronChannel + "\n");

		int defaultRasterChannel = 3;
		int rasterChannel = defaultRasterChannel;
		printer.print("Raster channel (1, 2, 3, 4) (default 3):");
		try {
			rasterChannel = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(rasterChannel)) rasterChannel = defaultRasterChannel;
		if (rasterChannel < defaultRasterChannel) rasterChannel = defaultRasterChannel;
		rasterChannel = ((int)(rasterChannel/neuronChannel)) * neuronChannel;
		if (rasterChannel < neuronChannel) rasterChannel = neuronChannel;
		printer.println("Raster channel is " + rasterChannel + "\n");
		
		int defaultRecoverNumber = 10;
		int recoverNumber = defaultRecoverNumber;
		printer.print("Recovery number (default 10):");
		try {
			recoverNumber = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(recoverNumber)) recoverNumber = defaultRecoverNumber;
		if (recoverNumber < defaultRecoverNumber) recoverNumber = defaultRecoverNumber;
		printer.println("Recovery number is " + recoverNumber + "\n");

		int defaultZDim = 10;
		int zDim = defaultZDim;
		printer.print("Enter Z data dimension (default 10):");
		try {
			zDim = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(zDim)) zDim = defaultZDim;
		if (zDim <= 0) zDim = defaultZDim;
		printer.println("Z data dimension is " + zDim + "\n");

		double defaultlr = 1;
		double lr = defaultlr;
		printer.print("Enter starting learning rate (default 1.0):");
		try {
			lr = Double.parseDouble(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(lr)) lr = defaultlr;
		if (lr <= 0 || lr > 1) lr = defaultlr;
		printer.println("Starting learning rate is " + lr + "\n");

		int defaultMaxIteration = 10;
		int maxIteration = defaultMaxIteration;
		printer.print("Max iteration (default 10 and increasing by 10 except 1):");
		try {
			maxIteration = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(maxIteration)) maxIteration = defaultMaxIteration;
		if (maxIteration < defaultMaxIteration) maxIteration = defaultMaxIteration;
		printer.println("Max iteration is " + maxIteration + "\n");

		printer.print("Enter base directory (" + Util.WORKING_DIRECTORY + "/base" + "):");
		String base = scanner.nextLine().trim();
		if (base.isEmpty()) base = Util.WORKING_DIRECTORY + "/base";
		printer.println("Base directory is \"" + base + "\".\n");
		Path baseDir = Paths.get(base);
		if (!Files.exists(baseDir) || !Files.isDirectory(baseDir)) {
			printer.println("Wrong base directory");
			return;
		}
		
		printer.print("Enter test directory (" + Util.WORKING_DIRECTORY + "/test" + "):");
		String test = scanner.nextLine().trim();
		if (test.isEmpty()) test = Util.WORKING_DIRECTORY + "/test";
		printer.println("Test directory is \"" + test + "\".\n");
		Path testDir = Paths.get(test);
		try {
			if (!Files.exists(testDir)) Files.createDirectory(testDir);
			if (!Files.isDirectory(testDir)) {
				printer.println("Wrong test directory");
				return;
			}
		} catch (Throwable e) {Util.trace(e);}

		printer.print("Enter test result directory (" + Util.WORKING_DIRECTORY + "/testresult" + "):");
		String testresult = scanner.nextLine().trim();
		if (testresult.isEmpty()) testresult = Util.WORKING_DIRECTORY + "/testresult";
		printer.println("Test result directory is \"" + testresult + "\".\n");
		Path testresultDir = Paths.get(testresult);
		try {
			if (!Files.exists(testresultDir)) Files.createDirectory(testresultDir);
		} catch (Throwable e) {Util.trace(e);}

		printer.print("Enter generation directory (" + Util.WORKING_DIRECTORY + "/gen" + "):");
		String gen = scanner.nextLine().trim();
		if (gen.isEmpty()) gen = Util.WORKING_DIRECTORY + "/gen";
		printer.println("Generating directory is \"" + gen + "\".\n");
		Path genDir = Paths.get(gen);
		try {
			if (!Files.exists(genDir)) Files.createDirectory(genDir);
		} catch (Throwable e) {Util.trace(e);}

		printer.print("Enter recovery directory (" + Util.WORKING_DIRECTORY + "/recover" + "):");
		String recover = scanner.nextLine().trim();
		if (recover.isEmpty()) recover = Util.WORKING_DIRECTORY + "/recover";
		printer.println("Recovering directory is \"" + recover + "\".\n");
		Path recoverDir = Paths.get(recover);
		try {
			if (!Files.exists(recoverDir)) Files.createDirectory(recoverDir);
		} catch (Throwable e) {Util.trace(e);}

		boolean saveRecover = false;
		printer.print("Save recovered rasters (true | false) (default is false):");
		try {
			String saveRecoverText = scanner.nextLine().trim().toLowerCase();
			saveRecover = saveRecoverText.equals("true");
		} catch (Throwable e) {}
		printer.println(saveRecover ? "Save recovered rasters.\n" : "Not save recovered rasters.\n");
		RasterAssoc.storePath = saveRecover;

		boolean randomRecover = true;
		printer.print("Random recovery (true | false) (default is true):");
		try {
			String randomGenText = scanner.nextLine().trim().toLowerCase();
			randomRecover = !randomGenText.equals("false");
		} catch (Throwable e) {}
		printer.println(randomRecover ? "Random recovery.\n" : "Best recovery.\n");

		List<Raster> baseRasters = Util.newList(0);
		List<Raster> testRasters = load3d ? RasterAssoc.load3D(testDir) : RasterAssoc.load(testDir);
		
		if (testRasters.size() > 0) {
			baseRasters = load3d ? RasterAssoc.load3D(baseDir) : RasterAssoc.load(baseDir);
		}
		else { //Randomizing testing rasters based on base rasters.
			List<Path> basePaths = Util.newList(0);
			try {
				Files.list(baseDir).filter(Files::isRegularFile).forEach((basePath) -> {
					basePaths.add(basePath);
				});
			} catch (Exception e) {Util.trace(e);}
			if (basePaths.size() < 2) {
				printer.println("Wrong base directory");
				return;
			}
			
			double r = 0.25;
			int k = (int)(basePaths.size()*r);
			if (k == 0) {
				printer.println("Empty base directory for creating test directory.");
				return;
			}
			Random rnd = new Random();
			for (int i = 0; i < k; i++) {
				int index = rnd.nextInt(basePaths.size());
				Path basePath = basePaths.remove(index);
				try {
					Files.move(basePath, testDir.resolve(basePath.getFileName()), StandardCopyOption.REPLACE_EXISTING);
				} catch (Exception e) {Util.trace(e);}
			}
			
			baseRasters = load3d ? RasterAssoc.load3D(baseDir) : RasterAssoc.load(baseDir);
			testRasters = load3d ? RasterAssoc.load3D(testDir) : RasterAssoc.load(testDir);
		}
		
		if (baseRasters.size() == 0 || testRasters.size() == 0) {
			printer.println("Empty base directory or empty test directory");
			return;
		}
		
		printer.println("Running...");

		List<GM> gms = Util.newList(0);
		
		ConvGenModel ava1 = AVAExt.create(neuronChannel, rasterChannel); //AVA1 is standard AVA defined by class AVA too.
		ava1.getConfig().put(AVAExt.SUPERVISE_ENCODE_FIELD, false);
		ava1.getConfig().put(AVAExt.SUPERVISE_DECODE_FIELD, true);
		ava1.getConfig().put(AVAExt.LEAN_ENCODE_FIELD, false);
		ava1.getConfig().put(AVAExt.LEAN_DECODE_FIELD, false);
		gms.add(new GM(ava1, "AVA1"));

		ConvGenModel ava2 = AVAExt.create(neuronChannel, rasterChannel);
		ava2.getConfig().put(AVAExt.SUPERVISE_ENCODE_FIELD, false);
		ava2.getConfig().put(AVAExt.SUPERVISE_DECODE_FIELD, true);
		ava2.getConfig().put(AVAExt.LEAN_ENCODE_FIELD, false);
		ava2.getConfig().put(AVAExt.LEAN_DECODE_FIELD, true);
		gms.add(new GM(ava2, "AVA2"));

		ConvGenModel ava3 = AVAExt.create(neuronChannel, rasterChannel);
		ava3.getConfig().put(AVAExt.SUPERVISE_ENCODE_FIELD, true);
		ava3.getConfig().put(AVAExt.SUPERVISE_DECODE_FIELD, false);
		ava3.getConfig().put(AVAExt.LEAN_ENCODE_FIELD, false);
		ava3.getConfig().put(AVAExt.LEAN_DECODE_FIELD, false);
		gms.add(new GM(ava3, "AVA3"));

		ConvGenModel ava4 = AVAExt.create(neuronChannel, rasterChannel);
		ava4.getConfig().put(AVAExt.SUPERVISE_ENCODE_FIELD, true);
		ava4.getConfig().put(AVAExt.SUPERVISE_DECODE_FIELD, false);
		ava4.getConfig().put(AVAExt.LEAN_ENCODE_FIELD, true);
		ava4.getConfig().put(AVAExt.LEAN_DECODE_FIELD, false);
		gms.add(new GM(ava4, "AVA4"));
		
		ConvGenModel ava5 = AVAExt.create(neuronChannel, rasterChannel);
		ava5.getConfig().put(AVAExt.SUPERVISE_ENCODE_FIELD, true);
		ava5.getConfig().put(AVAExt.SUPERVISE_DECODE_FIELD, true);
		ava5.getConfig().put(AVAExt.LEAN_ENCODE_FIELD, true);
		ava5.getConfig().put(AVAExt.LEAN_DECODE_FIELD, true);
		gms.add(new GM(ava5, "AVA5"));

		ConvGenModel vae = ConvVAEImpl.create(neuronChannel, rasterChannel);
		gms.add(new GM(vae, "VAE"));

		ConvGenModel gan = ConvGANImpl.create(neuronChannel, rasterChannel);
		gms.add(new GM(gan, "GAN"));

		ConvGenModel pixrnn = PixelRNNImpl.create(neuronChannel, rasterChannel);
		gms.add(new GM(pixrnn, "PIXRNN"));

		BufferedWriter writer = null;
		try {
			writer = Files.newBufferedWriter(testresultDir.resolve("TestResult" + System.currentTimeMillis() + ".txt"),
				StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		} catch (Throwable e) {Util.trace(e);}
		
		Size minSize = new Size(3, 3, load3d ? 3 : 1, 1);
		int nRecover = randomRecover ? recoverNumber : 1;
		for (int iteration = 0; iteration <= maxIteration; iteration += 10) {
			for (GM gm : gms) {
				gm.gm.getConfig().put(NetworkAbstract.LEARN_RATE_FIELD, lr);
				int iter = iteration < 1 ? 1 : iteration;
				gm.gm.getConfig().put(NetworkAbstract.LEARN_MAX_ITERATION_FIELD, iter);
				String name = gm.name + "-" + Util.format(lr) + "-" + iter;
				
				ConvGenModelAssoc assoc = new ConvGenModelAssoc(gm.gm, false);
				List<Raster> genRasters = assoc.initGenRasters(baseRasters, 1, zDim,
					SizeZoom.zoom(zoomOut, zoomOut, load3d ? zoomOut : 1, 1),
					minSize);
				RasterAssoc.saveDirector(genRasters, genDir, name);
				
				double bm = 0;
				int count = 0;
				for (Raster raster : testRasters) {
					GM clonedGM = (GM)Util.cloneBySerialize(gm);
					clonedGM.gm.learnRaster(Arrays.asList(raster));
					
					for (int k = 0; k < nRecover; k++) {
						G g = clonedGM.gm.recoverRaster(raster, null, randomRecover, true);
						if ((g == null) || (g.xgenUndefined == null) || !(g.xgenUndefined instanceof Raster))
							continue;
						
						bm += g.error;
						count++;
						
						if (saveRecover) {
							String genName = raster instanceof RasterWrapper ? name + "." + ((RasterWrapper)raster).getNamePlain() : name;
							Path path = RasterAssoc.genDefaultPath(recoverDir, genName, raster.getDefaultFormat());
							g.getXGenRaster().save(path);
						}
					}
				}
				if (count == 0) continue;
				
				bm = bm / (double)count;
				try {
					writer.write(name + ", bm=" + Util.format(bm) + "\n");
					writer.flush();
				} catch (Throwable e) {Util.trace(e);}
				
				gm.gm.reset();
				try {System.gc();} catch (Throwable e) {Util.trace(e);}
			} //GM.
		} //Iteration.
		
		try {
			writer.write("\n"); writer.flush();
		} catch (Throwable e) {Util.trace(e);}
		
		try {writer.close();} catch (Throwable e) {Util.trace(e);}
		
	}
	
	
//	/**
//	 * Main method.
//	 * @param args arguments.
//	 */
//	public static void main(String[] args) {
//		try {
//			gen(System.in, System.out);
//		}
//		catch (Throwable e) {Util.trace(e);}
//	}
	
	
}



