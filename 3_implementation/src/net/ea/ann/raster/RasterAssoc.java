/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.Dimension;
import java.awt.Rectangle;
import java.io.File;
import java.io.FileFilter;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.ImageAssoc.LabeledImage;

/**
 * This class is associator of raster.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RasterAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Flag to indicate whether to store raster path.
	 */
	public static boolean storePath = false;
	
	
	/**
	 * Internal convolutional generative model.
	 */
	protected Raster raster = null;

	
	/**
	 * Constructor with raster.
	 * @param raster specified raster.
	 */
	public RasterAssoc(Raster raster) {
		this.raster = raster;
	}


	/**
	 * Getting dimension of raster.
	 * @return dimension of raster.
	 */
	public int getDim() {
		if (raster == null)
			return 0;
		else if (raster.getTime() > 1)
			return 4;
		else if (raster.getDepth() > 1)
			return 3;
		else if (raster.getHeight() > 1)
			return 2;
		else if (raster.getWidth() > 1)
			return 1;
		else
			return 0;
	}
	
	
	/**
	 * Fixing raster channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return fixed raster channel.
	 */
	public static int fixRasterChannel(int neuronChannel, int rasterChannel) {
		if (rasterChannel <= neuronChannel) return neuronChannel;
		int ratio = (int)(rasterChannel / neuronChannel);
		return ratio*neuronChannel;
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param layer specified layer.
	 * @param values neuron values.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster created from neuron values.
	 */
	public static Raster createRaster(ConvLayerSingle layer, NeuronValue[] values,
			boolean isNorm, int defaultAlpha) {
		if (layer == null)
			return null;
		else if (layer.getTime() > 1)
			return Raster4DImpl.create(layer, values, isNorm, defaultAlpha);
		else if (layer.getDepth() > 1)
			return Raster3DImpl.create(layer, values, isNorm, defaultAlpha);
		else if (layer.getHeight() > 1)
			return Raster2DImpl.create(layer, values, isNorm, defaultAlpha);
		else if (layer.getWidth() > 1)
			return Raster1DImpl.create(layer, values, isNorm);
		else
			return Raster2DImpl.create(layer, values, isNorm, defaultAlpha);
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param values neuron values.
	 * @param neuronChannel neuron channel.
	 * @param size specified size.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster created from neuron values.
	 */
	public static Raster createRaster(NeuronValue[] values, int neuronChannel, Size size,
			boolean isNorm, int defaultAlpha) {
		if (values == null || size == null)
			return null;
		else if (size.time > 1)
			return Raster4DImpl.create(values, neuronChannel, size, isNorm, defaultAlpha);
		else if (size.depth > 1)
			return Raster3DImpl.create(values, neuronChannel, size, isNorm, defaultAlpha);
		else if (size.height > 1)
			return Raster2DImpl.create(values, neuronChannel, size, isNorm, defaultAlpha);
		else if (size.width > 1)
			return Raster1DImpl.create(values, neuronChannel, size, isNorm);
		else
			return Raster1DImpl.create(values, neuronChannel, size, isNorm);
	}

	
//	/**
//	 * Load rasters from directory. This method cause some Reflections (old version) trace because of the new Lambda expression (like for each) in new Java newer as 8.0.
//	 * However this trace is not serious. It is possible to use the other version of this method.
//	 * @param directory specified directory.
//	 * @return list of rasters loaded from directory.
//	 */
//	public static List<Raster> loadDirectory(Path directory) {
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
	 * Load rasters from directory or file.
	 * @param dirOrFile source directory or file.
	 * @return list of rasters loaded from directory or file.
	 */
	public static List<Raster> load(Path dirOrFile) {
		List<Raster> rasters = Util.newList(0);
		if (dirOrFile == null) return rasters;
		File[] files = null;
		if (Files.isDirectory(dirOrFile))
			files = dirOrFile.toFile().listFiles();
		else
			files = new File[] {dirOrFile.toFile()};
		if (files == null || files.length == 0) return rasters;

		for (File file : files) {
			if (!file.isFile()) continue;
			try {
				Path path = file.toPath();
				String fileName = file.getName();
				String fileExt = fileName.contains(".") ? fileName.substring(fileName.lastIndexOf(".") + 1) : null;
				if (fileExt == null || fileExt.isEmpty()) {
					Raster raster = Raster2DImpl.load(path);
					if (raster != null) rasters.add(storePath ? new RasterWrapper(raster, path) : raster);
					continue;
				}
				
				fileExt = fileExt.toLowerCase();
				if (fileExt.equals(Image.IMAGE_FORMAT_GIF.toLowerCase())) {
					ImageList imageList = ImageList.load(path);
					if (imageList != null) rasters.addAll(imageList.extractRasters());
				}
				else if (fileExt.equals(Image.VIDEO_FORMAT_DEFAULT.toLowerCase())) {
					ImageList imageList = ImageList.load(path);
					if (imageList != null) rasters.addAll(imageList.extractRasters());
				}
				else {
					Raster raster = Raster2DImpl.load(path);
					if (raster != null) rasters.add(storePath ? new RasterWrapper(raster, path) : raster);
				}
			} catch (Exception e) {Util.trace(e);}
		}
		return rasters;
	}


	/**
	 * Load rasters from directory or file.
	 * @param dirOrFile source directory or file.
	 * @return list of rasters loaded from directory or file.
	 */
	public static List<Raster> load3D(Path dirOrFile) {
		List<Raster> rasters = Util.newList(0);
		if (dirOrFile == null) return rasters;
		
		File[] files = null;
		if (Files.isDirectory(dirOrFile))
			files = dirOrFile.toFile().listFiles();
		else
			files = new File[] {dirOrFile.toFile()};
		if (files == null || files.length == 0) return rasters;
		
		List<Image> images = Util.newList(0);
		for (File file : files) {
			if (!file.isFile()) continue;
			try {
				String fileName = file.getName();
				String fileExt = fileName.contains(".") ? fileName.substring(fileName.lastIndexOf(".") + 1) : null;
				Path path = file.toPath();
				if (fileExt == null || fileExt.isEmpty()) {
					Raster raster = Raster3DImpl.load(path);
					if (raster != null) rasters.add(storePath ? new RasterWrapper(raster, path) : raster);
					continue;
				}
				
				fileExt = fileExt.toLowerCase();
				if (fileExt.equals(Image.IMAGE_FORMAT_GIF.toLowerCase())) {
					Raster raster = Raster3DImpl.load(path);
					if (raster != null) rasters.add(storePath ? new RasterWrapper(raster, path) : raster);
				}
				else if (fileExt.equals(Image.VIDEO_FORMAT_DEFAULT.toLowerCase())) {
					Raster raster = Raster3DImpl.load(path);
					if (raster != null) rasters.add(storePath ? new RasterWrapper(raster, path) : raster);
				}
				else {
					ImageWrapper image = ImageWrapper.load(path);
					if (image != null) images.add(image);
				}
			} catch (Exception e) {Util.trace(e);}
		}

		if (images.size() > 0) {
			Raster raster = Raster3DImpl.create(ImageList.create(images));
			if (raster != null) rasters.add(raster);
		}
		return rasters;
	}

	
	/**
	 * Load CIFAR rasters from directory or file.
	 * @param dirOrFile source directory or file.
	 * @param nImages number of images to be loaded.
	 * @return list of rasters loaded from directory or file.
	 */
	public static List<Raster> loadCIFAR(Path dirOrFile, int nImages) {
		List<Raster> rasters = Util.newList(0);
		if (dirOrFile == null) return rasters;
		File[] files = null;
		if (Files.isDirectory(dirOrFile))
			files = dirOrFile.toFile().listFiles();
		else
			files = new File[] {dirOrFile.toFile()};
		if (files == null || files.length == 0) return rasters;

		for (File file : files) {
			if (!file.isFile()) continue;
			try {
				Path path = file.toPath();
				List<LabeledImage> images = ImageAssoc.loadCIFAR10(path, nImages);
				for (LabeledImage image : images) {
					Raster raster = image.toRaster();
					if (raster != null) rasters.add(raster);
				}
			} catch (Exception e) {Util.trace(e);}
		}
		return rasters;
	}
	
	
	/**
	 * Load CIFAR rasters from directory or file.
	 * @param dirOrFile source directory or file.
	 * @return list of rasters loaded from directory or file.
	 */
	public static List<Raster> loadCIFAR(Path dirOrFile) {
		return loadCIFAR(dirOrFile, -1);
	}
	
	
	/**
	 * Load rasters by folders from directory.
	 * @param dir source directory.
	 * @param load3D flag to 3D loading.
	 * @return list of rasters loaded by folders from directory.
	 */
	private static List<Raster> loadFolders0(Path dir, boolean load3D) {
		List<Raster> rasters = Util.newList(0);
		if (dir == null || !Files.isDirectory(dir)) return rasters;
		File[] folders = dir.toFile().listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return pathname.isDirectory();
			}
		});
		if (folders == null || folders.length == 0) return rasters;
		
		for (int i = 0; i < folders.length; i++) {
			try {
				List<Raster> subRasters = load3D ? load3D(folders[i].toPath()) : load(folders[i].toPath());
				for (Raster subRaster : subRasters) {
					subRaster.getProperty().setLabelId(i);
					subRaster.getProperty().setLabelName(folders[i].getName());
					rasters.add(subRaster);
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		return rasters;
	}

	
	/**
	 * Load rasters by folders from directory.
	 * @param dir source directory.
	 * @return list of rasters loaded by folders from directory.
	 */
	public static List<Raster> loadFolders(Path dir) {
		return loadFolders0(dir, false);
	}
	
	
	/**
	 * Load 3D rasters by folders from directory.
	 * @param dir source directory.
	 * @return list of 3D rasters loaded by folders from directory.
	 */
	public static List<Raster> loadFolders3D(Path dir) {
		return loadFolders0(dir, true);
	}

	
	/**
	 * Saving rasters to directory.
	 * @param rasters collection of rasters.
	 * @param directory target directory.
	 * @param prefix prefix name.
	 * @param indexing flag to indicate whether to index names.
	 * @return number of generated rasters.
	 */
	public static int saveDirector(Iterable<Raster> rasters, Path directory, String prefix, boolean indexing) {
		if (rasters == null || !Files.isDirectory(directory)) return 0;
	
		int count = 0;
		for (Raster raster : rasters) {
			Path path = RasterAssoc.genDefaultPath(directory, prefix, raster.getDefaultFormat(), indexing ? count+1 : -1);
			if (raster.save(path)) count++;
		}
		
		return count;
	}


	/**
	 * Saving rasters to directory.
	 * @param rasters collection of rasters.
	 * @param directory target directory.
	 * @param prefix prefix name.
	 * @return number of generated rasters.
	 */
	public static int saveDirector(Iterable<Raster> rasters, Path directory, String prefix) {
		return saveDirector(rasters, directory, prefix, false);
	}
	
	
	/**
	 * Getting default name for generated raster.
	 * @param prefix prefix name.
	 * @param format raster format.
	 * @param index name index.
	 * @return default name for generated raster.
	 */
	public static String genDefaultName(String prefix, String format, int index) {
		String name = (prefix != null && !prefix.isEmpty()) ? prefix + ".gen." : "gen.";
		return name + System.currentTimeMillis() + (index < 0 ? "" : "_" + index) +
			(format != null && !format.isEmpty() ? "." + format : "");
	}
	
	
	/**
	 * Getting default name for generated raster.
	 * @param prefix prefix name.
	 * @param format raster format.
	 * @return default name for generated raster.
	 */
	public static String genDefaultName(String prefix, String format) {
		return genDefaultName(prefix, format, -1);
	}

	
	/**
	 * Getting default path for generated raster.
	 * @param parent parent directory.
	 * @param prefix prefix name.
	 * @param format raster format.
	 * @param index name index.
	 * @return default path for generated raster.
	 */
	public static Path genDefaultPath(Path parent, String prefix, String format, int index) {
		return parent.resolve(genDefaultName(prefix, format, index));
	}

	
	/**
	 * Getting default path for generated raster.
	 * @param parent parent directory.
	 * @param prefix prefix name.
	 * @param format raster format.
	 * @return default path for generated raster.
	 */
	public static Path genDefaultPath(Path parent, String prefix, String format) {
		return genDefaultPath(parent, prefix, format, -1);
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
			Record record = new Record(raster);
			sample.add(record);
		}
		return sample;
	}

	
	/**
	 * Extracting rasters from sample.
	 * @param sample sample.
	 * @return extracted rasters.
	 */
	public static List<Raster> toInputRasters(Iterable<Record> sample) {
		List<Raster> rasters = Util.newList(0);
		for (Record record : sample) {
			Raster raster = record != null ? record.getRasterInput() : null;
			if (raster != null) rasters.add(raster);
		}
		return rasters;
	}
	
	
	/**
	 * Resizing the size according to with, height, depth, and their zoom ratio.
	 * @param size original size.
	 * @param minSize minimum size.
	 * @return the fit size.
	 */
	public static SizeZoom calcFitSize(SizeZoom size, Size minSize) {
		if (size.width < 1 || size.height < 1 || size.depth < 1 || size.time < 1 ||
				size.widthZoom < 1 || size.heightZoom < 1 || size.depthZoom < 1 || size.timeZoom < 1 ||
				minSize.width < 0 || minSize.height < 0 || minSize.depth < 0|| minSize.time < 0) {
			size.width = size.width < 1 ? 0 : size.width;
			size.height = size.height < 1 ? 0 : size.height;
			size.depth = size.depth < 1 ? 0 : size.depth;
			size.time = size.time < 1 ? 0 : size.time;
			size.widthZoom = size.heightZoom = size.depthZoom = size.timeZoom = 1;
			return size;
		}
		
		double ratio = (double)size.height / (double)size.width;
		int newMinHeight = (int)(ratio*minSize.width + 0.5);
		if (newMinHeight < minSize.height && newMinHeight > 3/*pixels*/) {
			minSize.height = newMinHeight; //Reserve the ratio height/width.
		}
		if ((size.width/size.widthZoom < minSize.width || size.height/size.heightZoom < minSize.height) && (minSize.width != 0 && minSize.height != 0)) {
			size.widthZoom = Math.max(size.widthZoom, size.width/minSize.width);
			size.heightZoom = Math.max(size.heightZoom, size.height/minSize.height);
			
			int zoom = Math.max(size.widthZoom, size.heightZoom);
			size.widthZoom = size.heightZoom = zoom;
			size.width = minSize.width*zoom;
			size.height = minSize.height*zoom;
		}
		
		if ((size.depth/size.depthZoom < minSize.depth) && (minSize.depth != 0)) {
			size.depthZoom = Math.max(size.depthZoom, size.depth/minSize.depth);
			size.depth = minSize.depth*size.depthZoom;
		}

		if ((size.time/size.timeZoom < minSize.time) && (minSize.time != 0)) {
			size.timeZoom = Math.max(size.timeZoom, size.time/minSize.time);
			size.time = minSize.time*size.timeZoom;
		}

		return size;
	}

	
	/**
	 * Getting average with and height of rasters in sample.
	 * @param sample specified sample.
	 * @return average with and height of rasters in sample.
	 */
	public static Size getAverageSize(Iterable<Raster> sample) {
		if (sample == null) return new Size(0, 0, 0);
	
		int n = 0, width = 0, height = 0, depth = 0, time = 0;
		for (Raster raster : sample) {
			width += raster.getWidth();
			height += raster.getHeight();
			depth += raster.getDepth();
			time += raster.getTime();
			n++;
		}
		if (n == 0) return new Size(0, 0, 0, 0);
		
		return new Size(width/n, height/n, depth/n, time/n);
	}


	/**
	 * Extracting data source from index with range.
	 * @param <T> Template type.
	 * @param tClass Class of template type.
	 * @param source specified data source.
	 * @param sourceIndex specified index.
	 * @param sourceRange specified length.
	 * @return data extracted from index with specified length.
	 */
	public static <T> T[] extractRange1D(Class<T> tClass, T[] source, int sourceIndex, int sourceRange) {
		if (source == null || source.length == 0 || sourceRange <= 0) return null;
		sourceIndex = sourceIndex < 0 ? 0 : sourceIndex;
		sourceRange = sourceIndex + sourceRange <= source.length ? sourceRange : source.length - sourceIndex;
		if (sourceRange <= 0) return null;

		T[] extractedData = Util.newArray(tClass, sourceRange);
		for (int i = 0; i < sourceRange; i++) extractedData[i] = source[sourceIndex + i];
		
		return extractedData;
	}
	
	
	/**
	 * Extracting source data from specified rectangle.
	 * @param <T> Template type.
	 * @param tClass Class of template type.
	 * @param source specified data.
	 * @param sourceSize size (width and height) of source data.
	 * @param sourceRange specified rectangle.
	 * @return data extracted from specified rectangle.
	 */
	public static <T> T[] extractRange2D(Class<T> tClass, T[] source, Dimension sourceSize, Rectangle sourceRange) {
		if (source == null || source.length == 0 || sourceSize == null || sourceRange == null)
			return null;
		if (sourceSize.width <= 0 || sourceSize.height <= 0 || sourceRange.width <= 0 || sourceRange.height <= 0)
			return null;

		sourceRange.x = sourceRange.x < 0 ? 0 : sourceRange.x;
		sourceRange.y = sourceRange.y < 0 ? 0 : sourceRange.y;
		sourceRange.width = sourceRange.x + sourceRange.width <= sourceSize.width ? sourceRange.width : sourceSize.width - sourceRange.x;
		sourceRange.height = sourceRange.y + sourceRange.height <= sourceSize.height ? sourceRange.height : sourceSize.height - sourceRange.y;
		if (sourceRange.width <= 0 || sourceRange.height <= 0)
			return null;
		
		T[] subRaster = Util.newArray(tClass, sourceRange.width*sourceRange.height);
		int subIndex = 0;
		for (int i = 0; i < sourceRange.height; i++) {
			for (int j = 0; j < sourceRange.width; j++) {
				int index = (sourceRange.y+i)*sourceSize.width + (sourceRange.x+j);
				subRaster[subIndex] = source[index];
				subIndex++;
			}
		}

		return subRaster;
	}
	
	
	/**
	 * Extracting source data from specified rectangle.
	 * @param <T> Template type.
	 * @param tClass Class of template type.
	 * @param source specified data.
	 * @param sourceSize size (width and height) of source data.
	 * @param sourceRange specified rectangle.
	 * @return data extracted from specified rectangle.
	 */
	public static <T> T[] extractRange3D(Class<T> tClass, T[] source, Size sourceSize, Cube sourceRange) {
		if (source == null || source.length == 0 || sourceSize == null || sourceRange == null)
			return null;
		if (sourceSize.width <= 0 || sourceSize.height <= 0 || sourceSize.depth <= 0 || sourceRange.width <= 0 || sourceRange.height <= 0 || sourceRange.depth <= 0)
			return null;

		sourceRange.x = sourceRange.x < 0 ? 0 : sourceRange.x;
		sourceRange.y = sourceRange.y < 0 ? 0 : sourceRange.y;
		sourceRange.z = sourceRange.z < 0 ? 0 : sourceRange.z;
		sourceRange.width = sourceRange.x + sourceRange.width <= sourceSize.width ? sourceRange.width : sourceSize.width - sourceRange.x;
		sourceRange.height = sourceRange.y + sourceRange.height <= sourceSize.height ? sourceRange.height : sourceSize.height - sourceRange.y;
		sourceRange.depth = sourceRange.z + sourceRange.depth <= sourceSize.depth ? sourceRange.depth : sourceSize.depth - sourceRange.z;
		if (sourceRange.width <= 0 || sourceRange.height <= 0 || sourceRange.depth <= 0)
			return null;
		
		T[] subRaster = Util.newArray(tClass, sourceRange.width*sourceRange.height*sourceRange.depth);
		int subIndex = 0;
		for (int i = 0; i < sourceRange.depth; i++) {
			int indexZ = (sourceRange.z+i)*sourceSize.width*sourceSize.height;
			for (int j = 0; j < sourceRange.height; j++) {
				for (int k = 0; k < sourceRange.width; k++) {
					int index = indexZ + (sourceRange.y+j)*sourceSize.width + (sourceRange.x+k);
					subRaster[subIndex] = source[index];
					subIndex++;
				}
			}
		}
		
		return subRaster;
	}

	
	/**
	 * Copying from source data to target data.
	 * @param <T> Template type.
	 * @param source specified source data.
	 * @param sourceIndex specified source index.
	 * @param sourceRange specified source rage.
	 * @param target specified target data.
	 * @param targetIndex specified target index.
	 * @return true if copying is successful.
	 */
	public static <T> boolean copyRange1D(T[] source, int sourceIndex, int sourceRange, T[] target, int targetIndex) {
		if (source == null || source.length == 0 || sourceRange <= 0 || target == null || target.length == 0) return false;
		sourceIndex = sourceIndex < 0 ? 0 : sourceIndex;
		sourceRange = sourceIndex + sourceRange <= source.length ? sourceRange : source.length - sourceIndex;
		if (sourceRange <= 0) return false;

		targetIndex = targetIndex < 0 ? 0 : targetIndex;
		targetIndex = targetIndex < target.length ? targetIndex : target.length-1;
		
		int n = Math.min(sourceRange, target.length - targetIndex);
		for (int i = 0; i < n; i++) {
			target[targetIndex + i] = source[sourceIndex + i];
		}
		
		return true;
	}
	
	
	/**
	 * Copying from source data to target data.
	 * @param <T> Template type.
	 * @param source specified source data.
	 * @param sourceIndex specified source index.
	 * @param sourceRange specified source rage.
	 * @param target specified target data.
	 * @return true if copying is successful.
	 */
	public static <T> boolean copyRange1D(T[] source, int sourceIndex, int sourceRange, T[] target) {
		return copyRange1D(source, sourceIndex, sourceRange, target, 0);
	}
	
	
	/**
	 * Copying an area from source data to target data.
	 * @param <T> Template type.
	 * @param source source data.
	 * @param sourceSize size of source data.
	 * @param sourceRange range of source data.
	 * @param target target data.
	 * @param targetSize size of target data.
	 * @param targetPoint place to copy at target data.
	 * @return true if copying is successful.
	 */
	public static <T> boolean copyRange2D(T[] source, Dimension sourceSize, Rectangle sourceRange, T[] target, Dimension targetSize, Point targetPoint) {
		if (source == null || source.length == 0 || sourceSize == null || sourceRange == null || target == null || targetSize == null || targetPoint == null)
			return false;
		if (sourceSize.width <= 0 || sourceSize.height <= 0 || sourceRange.width <= 0 || sourceRange.height <= 0)
			return false;
		if (targetSize.width <= 0 || targetSize.height <= 0)
			return false;
		
		sourceRange.x = sourceRange.x < 0 ? 0 : sourceRange.x;
		sourceRange.y = sourceRange.y < 0 ? 0 : sourceRange.y;
		sourceRange.width = sourceRange.x + sourceRange.width <= sourceSize.width ? sourceRange.width : sourceSize.width - sourceRange.x;
		sourceRange.height = sourceRange.y + sourceRange.height <= sourceSize.height ? sourceRange.height : sourceSize.height - sourceRange.y;
		if (sourceRange.width <= 0 || sourceRange.height <= 0)
			return false;

		targetPoint.x = targetPoint.x < 0 ? 0 : targetPoint.x;
		targetPoint.x = targetPoint.x < targetSize.width ? targetPoint.x : targetSize.width-1;
		targetPoint.y = targetPoint.y < 0 ? 0 : targetPoint.y;
		targetPoint.y = targetPoint.y < targetSize.height ? targetPoint.y : targetSize.height-1;
		
		int m = Math.min(sourceRange.height, targetSize.height - targetPoint.y);
		int n = Math.min(sourceRange.width, targetSize.width - targetPoint.x);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int sourceIndex = (sourceRange.y+i)*sourceSize.width + (sourceRange.x+j);
				int targetIndex = (targetPoint.y+i)*targetSize.width + (targetPoint.x+j);
				target[targetIndex] = source[sourceIndex];
			}
		}
		
		return true;
	}
	
	
	/**
	 * Copying an area from source data to target data.
	 * @param <T> Template type.
	 * @param source source data.
	 * @param sourceSize size of source data.
	 * @param sourceRange range of source data.
	 * @param target target data.
	 * @param targetSize size of target data.
	 * @return true if copying is successful.
	 */
	public static <T> boolean copyRange2D(T[] source, Dimension sourceSize, Rectangle sourceRange, T[] target, Dimension targetSize) {
		return copyRange2D(source, sourceSize, sourceRange, target, targetSize, new Point(0, 0));
	}
	
	
	/**
	 * Copying an area from source data to target data.
	 * @param <T> Template type.
	 * @param source source data.
	 * @param sourceSize size of source data.
	 * @param sourceRange range of source data.
	 * @param target target data.
	 * @param targetSize size of target data.
	 * @param targetPoint place to copy at target data.
	 * @return true if copying is successful.
	 */
	public static <T> boolean copyRange3D(T[] source, Size sourceSize, Cube sourceRange, T[] target, Size targetSize, Point targetPoint) {
		if (source == null || source.length == 0 || sourceSize == null || sourceRange == null || target == null || targetSize == null || targetPoint == null)
			return false;
		if (sourceSize.width <= 0 || sourceSize.height <= 0 || sourceSize.depth <= 0 || sourceRange.width <= 0 || sourceRange.height <= 0 || sourceRange.depth <= 0)
			return false;
		if (targetSize.width <= 0 || targetSize.height <= 0 || targetSize.depth <= 0)
			return false;
		
		sourceRange.x = sourceRange.x < 0 ? 0 : sourceRange.x;
		sourceRange.y = sourceRange.y < 0 ? 0 : sourceRange.y;
		sourceRange.z = sourceRange.z < 0 ? 0 : sourceRange.z;
		sourceRange.width = sourceRange.x + sourceRange.width <= sourceSize.width ? sourceRange.width : sourceSize.width - sourceRange.x;
		sourceRange.height = sourceRange.y + sourceRange.height <= sourceSize.height ? sourceRange.height : sourceSize.height - sourceRange.y;
		sourceRange.depth = sourceRange.z + sourceRange.depth <= sourceSize.depth ? sourceRange.depth : sourceSize.depth - sourceRange.z;
		if (sourceRange.width <= 0 || sourceRange.height <= 0 || sourceRange.depth <= 0)
			return false;

		targetPoint.x = targetPoint.x < 0 ? 0 : targetPoint.x;
		targetPoint.x = targetPoint.x < targetSize.width ? targetPoint.x : targetSize.width-1;
		targetPoint.y = targetPoint.y < 0 ? 0 : targetPoint.y;
		targetPoint.y = targetPoint.y < targetSize.height ? targetPoint.y : targetSize.height-1;
		targetPoint.z = targetPoint.z < 0 ? 0 : targetPoint.z;
		targetPoint.z = targetPoint.z < targetSize.depth ? targetPoint.z : targetSize.depth-1;
		
		int l = Math.min(sourceRange.depth, targetSize.depth - targetPoint.z);
		int m = Math.min(sourceRange.height, targetSize.height - targetPoint.y);
		int n = Math.min(sourceRange.width, targetSize.width - targetPoint.x);
		for (int i = 0; i < l; i++) {
			int sourceIndexZ = (sourceRange.z+i)*sourceSize.width*sourceSize.height;
			int targetIndexZ = (targetPoint.z+i)*targetSize.width*targetSize.height;
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < n; k++) {
					int sourceIndex = sourceIndexZ + (sourceRange.y+j)*sourceSize.width + (sourceRange.x+k);
					int targetIndex = targetIndexZ + (targetPoint.y+j)*targetSize.width + (targetPoint.x+k);
					target[targetIndex] = source[sourceIndex];
				}
			}
		}
		
		return true;
	}


	/**
	 * Copying an area from source data to target data.
	 * @param <T> Template type.
	 * @param source source data.
	 * @param sourceSize size of source data.
	 * @param sourceRange range of source data.
	 * @param target target data.
	 * @param targetSize size of target data.
	 * @return true if copying is successful.
	 */
	public static <T> boolean copyRange3D(T[] source, Size sourceSize, Cube sourceRange, T[] target, Size targetSize) {
		return copyRange3D(source, sourceSize, sourceRange, target, targetSize, new Point(0, 0, 0, 0));
	}


}
