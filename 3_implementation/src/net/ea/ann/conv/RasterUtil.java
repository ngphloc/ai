/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Dimension;
import java.io.File;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import net.ea.ann.core.Record;
import net.ea.ann.core.Util;

/**
 * This utilitt class provides utility methods for convolutional generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RasterUtil implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional generative model.
	 */
	protected Raster raster = null;

	
	/**
	 * Constructor with raster.
	 * @param raster specified raster.
	 */
	public RasterUtil(Raster raster) {
		this.raster = raster;
	}


//	/**
//	 * Load rasters from directory. This method cause some Reflections (old version) trace because of the new Lambda expression (like for each) in new Java newer as 8.0.
//	 * However this trace is not serious. It is possible to use the other version of this method.
//	 * @param directory specified directory.
//	 * @return list of rasters loaded from directory.
//	 */
//	public static List<Raster> loadDirectory2(Path directory) {
//		List<Raster> rasters = Util.newList(0);
//		if (!Files.isDirectory(directory)) return rasters;
//		
//		try {
//			Files.walk(directory).filter(Files::isRegularFile).forEach((path) -> {
//				try {
//					Raster raster = Raster.load(path);
//					if (raster != null) rasters.add(raster);
//				} catch (Throwable e) {}
//			});
//		} catch (Exception e) {
//			Util.trace(e);
//		}
//		
//		return rasters;
//	}

	
	/**
	 * Load rasters from directory.
	 * @param directory source directory.
	 * @return list of rasters loaded from directory.
	 */
	public static List<Raster> loadDirectory(Path directory) {
		List<Raster> rasters = Util.newList(0);
		if (directory == null || !Files.isDirectory(directory)) return rasters;
		
		try {
			File[] files = directory.toFile().listFiles();
			for (File file : files) {
				if (!file.isFile()) continue;
				
				try {
					Raster raster = Raster.load(file.toPath());
					if (raster != null) rasters.add(raster);
				} catch (Throwable e) {}
			}
			
		} catch (Exception e) {Util.trace(e);}
		
		return rasters;
	}


	/**
	 * Saving rasters to directory.
	 * @param directory target directory.
	 * @param prefix prefix name.
	 * @return number of generated rasters.
	 */
	public static int saveDirector(Iterable<Raster> rasters, Path directory, String prefix) {
		if (rasters == null || !Files.isDirectory(directory)) return 0;
	
		int count = 0;
		for (Raster raster : rasters) {
			Path path = RasterUtil.genDefaultPath(directory, prefix);
			if (raster.save(path)) count++;
		}
		
		return count;
	}


	/**
	 * Getting default path for generated raster.
	 * @param parent parent directory.
	 * @param prefix prefix name.
	 * @return default path for generated raster.
	 */
	public static Path genDefaultPath(Path parent, String prefix) {
		String name = (prefix != null && !prefix.isEmpty()) ? prefix + ".gen." : "gen.";
		return parent.resolve(name + System.currentTimeMillis() + "." + Raster.IMAGE_FORMAT_DEFAULT);
	}


	/**
	 * Converting raster to record sample.
	 * @param rasters specified rasters.
	 * @return converted record sample.
	 */
	public static List<Record> toInputSample(Iterable<Raster> rasters) {
		List<Record> sample = Util.newList(0);
		for (Raster raster : rasters) {
			if (raster == null) continue;
			
			Record record = new Record();
			record.undefinedInput = raster;
			sample.add(record);
		}
		
		return sample;
	}


	/**
	 * Resizing the size according to zooming ratio, minimum width, and minimum height.
	 * @param width original width.
	 * @param height original height.
	 * @param zoomRatio zooming ratio.
	 * @param xMinWidth X minimum width.
	 * @param xMinHeight X minimum height.
	 * @return the fit size including fitSize[0] = width, fitSize[1] = height, and fitSize[0] = zooming-ratio according to minimum width, minimum height, and zooming ratio.
	 */
	public static int[] getFitSize(int width, int height, int zoomRatio, int xMinWidth, int xMinHeight) {
		if (width < 1 || height < 1 || zoomRatio <= 1 || xMinWidth < 1 || xMinHeight < 1) {
			width = width < 1 ? 0 : width;
			height = height < 1 ? 0 : height;
			return new int[] {width, height, 1};
		}
		
		Dimension size = new Dimension(width, height);
		double ratio = (double)height / (double)width;
		int newMinHeight = (int)(ratio*xMinWidth + 0.5);
		if (newMinHeight < xMinHeight && newMinHeight > 3/*pixels*/) {
			xMinHeight = newMinHeight; //Reserve the raster ratio.
		}
		
		if (width/zoomRatio < xMinWidth || height/zoomRatio < xMinHeight) {
			zoomRatio = Math.max(width/xMinWidth, height/xMinHeight);
			size.width = xMinWidth*zoomRatio;
			size.height = xMinHeight*zoomRatio;
		}
		else {
			size.width = width;
			size.height = height;
		}
		
		return new int[] {size.width, size.height, zoomRatio};
	}

	
}
