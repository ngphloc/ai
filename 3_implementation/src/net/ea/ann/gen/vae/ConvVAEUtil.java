/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.awt.Dimension;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import net.ea.ann.conv.Raster;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.core.Util;

/**
 * This utilitt class provides utility methods with convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvVAEUtil implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional Variational Autoencoders.
	 */
	protected ConvVAEImpl convVAE = null;
	
	
	/**
	 * Constructor with convolutional Variational Autoencoders.
	 * @param convVAE convolutional Variational Autoencoders
	 */
	protected ConvVAEUtil(ConvVAEImpl convVAE) {
		this.convVAE = convVAE;
	}

	
	/**
	 * Creating this utility with convolutional Variational Autoencoders.
	 * @param convVAE convolutional Variational Autoencoders.
	 * @return utility with convolutional Variational Autoencoders.
	 */
	public static ConvVAEUtil create(ConvVAEImpl convVAE) {
		if (convVAE == null)
			return null;
		else
			return new ConvVAEUtil(convVAE);
	}
	
	
	/**
	 * Getting X dimension.
	 * @return X dimension.
	 */
	public int getXDim() {
		if (convVAE.encoder == null)
			return 0;
		else
			return convVAE.encoder.getInputLayer().size();
	}
	
	
	/**
	 * Getting Z dimension.
	 * @return Z dimension.
	 */
	public int getZDim() {
		if (convVAE.decoder == null)
			return 0;
		else
			return convVAE.decoder.getInputLayer().size();
	}

	
	/**
	 * Getting width.
	 * @return width.
	 */
	public int getWidth() {
		return convVAE.width;
	}
	
	
	/**
	 * Getting height.
	 * @return height.
	 */
	public int getHeight() {
		return convVAE.height;
	}
	
	
	/**
	 * Getting average with and height of rasters in sample.
	 * @param sample specified sample.
	 * @return average with and height of rasters in sample.
	 */
	private static Dimension getAverageSize(Iterable<Raster> sample) {
		if (sample == null) new Dimension(0, 0);

		int n = 0, width = 0, height = 0;
		for (Raster raster : sample) {
			width += raster.getWidth();
			height += raster.getHeight();
			n++;
		}
		if (n == 0) return new Dimension(0, 0);
		
		width = width / n;
		height = height / n;
		return new Dimension(width, height);
	}
	
	
	/**
	 * Generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @return list of generated rasters.
	 */
	public List<Raster> generateRasters(Iterable<Raster> sample, int nGens) {
		List<Raster> result = Util.newList(0);
		try {
			convVAE.learnByRaster(sample);
		}
		catch (Exception e) {
			Util.trace(e);
			return result;
		}
		
		nGens = nGens < 1 ? 1 : nGens;
		for (int i = 0; i < nGens; i++) {
			try {
				Raster raster = convVAE.generateRaster();
				if (raster != null) result.add(raster);
			} catch (Exception e) {Util.trace(e);}
		}
		
		try {
			convVAE.close();
		} catch (Exception e) {Util.trace(e);}
		
		return result;
	}
	
	
	/**
	 * Generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters convolutional filters.
	 * @return list of generated rasters.
	 */
	public List<Raster> generateRasters(Iterable<Raster> sample, int nGens, int zDim, Filter[] convFilters, Filter[] deconvFilters) {
		if (sample == null || zDim <= 0) return Util.newList(0);

		Dimension size = getAverageSize(sample);
		if (size.width == 0 || size.height == 0) return Util.newList(0);
		ConvVAESetting setting = convVAE.getSetting();
		setting.width = size.width;
		setting.height = size.height;
		convVAE.setSetting(setting);
		
		if (!convVAE.initialize(zDim, convFilters, deconvFilters)) return Util.newList(0);
		
		return generateRasters(sample, nGens);
	}
	
	
	/**
	 * Generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension.
	 * @param convFilters convolutional filters.
	 * @return list of generated rasters.
	 */
	public List<Raster> generateRasters(Iterable<Raster> sample, int nGens, int zDim, Filter[] convFilters) {
		return generateRasters(sample, nGens, zDim, convFilters, null);
	}

	
	/**
	 * Generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension.
	 * @return list of generated rasters.
	 */
	public List<Raster> generateRasters(Iterable<Raster> sample, int nGens, int zDim) {
		return generateRasters(sample, nGens, zDim, null, null);
	}

	
	/**
	 * Generating rasters from sample.
	 * @param sample raster sample.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension.
	 * @param zoomOutRatio zoom out ration.
	 * @return list of generated rasters.
	 */
	public List<Raster> generateRasters(Iterable<Raster> sample, int nGens, int zDim, int zoomOutRatio) {
		if (sample == null || zDim <= 0) return Util.newList(0);

		Dimension size = getAverageSize(sample);
		if (size.width == 0 || size.height == 0) return Util.newList(0);
		ConvVAESetting setting = convVAE.getSetting();
		setting.width = size.width;
		setting.height = size.height;
		convVAE.setSetting(setting);
		
		Filter[] convFilters = null;
		Filter[] deconvFilters = null;
		if (zoomOutRatio > 1) {
			FilterFactory factory = convVAE.getFilterFactory();
			convFilters = new Filter[] {factory.zoomOut(zoomOutRatio, zoomOutRatio)};
			deconvFilters = new Filter[] {factory.zoomIn(zoomOutRatio, zoomOutRatio)};
		}
		
		if (!convVAE.initialize(zDim, convFilters, deconvFilters)) return Util.newList(0);
		
		return generateRasters(sample, nGens);
	}

	
	/**
	 * Generating rasters from source directory to target directory.
	 * @param sourceDirectory source directory.
	 * @param targetDirectory target directory.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension.
	 * @param zoomOutRatio zoom out ration.
	 * @return number of generated rasters.
	 */
	public int generateRasters(Path sourceDirectory, Path targetDirectory, int nGens, int zDim, int zoomOutRatio) {
		if ((!Files.isDirectory(sourceDirectory)) || (!Files.isDirectory(targetDirectory))) return 0;
		
		List<Raster> sample = Raster.loadDirectory(sourceDirectory);
		List<Raster> rasters = generateRasters(sample, nGens, zDim, zoomOutRatio);
		for (Raster raster : rasters) {
			Path path = targetDirectory.resolve("gen" + System.currentTimeMillis() + "." + Raster.IMAGE_FORMAT_DEFAULT);
			raster.save(path);
		}
		
		return rasters.size();
	}
	
	
	/**
	 * Generating rasters from source directory to target directory.
	 * @param sourceDirectory source directory.
	 * @param targetDirectory target directory.
	 * @param nGens number of generated raster.
	 * @param zDim Z dimension.
	 * @return list of generated rasters.
	 */
	public void generateRasters(Path sourceDirectory, Path targetDirectory, int nGens, int zDim) {
		generateRasters(sourceDirectory, targetDirectory, nGens, zDim, 1);
	}

	

	/**
	 * Main method.
	 * @param args arguments.
	 */
	public static void main(String[] args) {
		ConvVAEUtil util = new ConvVAEUtil(ConvVAEImpl.create(3));
		util.generateRasters(Paths.get("working/sample"), Paths.get("working/gen"),
			5, 10, 3);
	}

	
}
