/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.com.madgag.gif.fmsware.AnimatedGifEncoder;
import net.ea.ann.raster.com.madgag.gif.fmsware.GifDecoder;

/**
 * This class represents a list of images.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ImageList implements Image {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default for delay in milisecond.
	 */
	protected final static int DELAY = 1000;
	
	
	/**
	 * List of image wrappers.
	 */
	private List<Image> images = Util.newList(0);
	
	
	/**
	 * Image width.
	 */
	private int width = 0;
	
	
	/**
	 * Image height.
	 */
	private int height = 0;
	

	/**
	 * Delay in each frame in milisecond.
	 */
	private int delay = DELAY;
	
	
	/**
	 * Default constructor.
	 */
	protected ImageList() {

	}

	
	/**
	 * Constructor with image and delay.
	 * @param image specified image.
	 * @param delay specified delay.
	 */
	protected ImageList(Image image, int delay) {
		this.width = image.getWidth();
		this.height = image.getHeight();
		this.delay = delay;
		this.images.add(image);
	}

	
	/**
	 * Constructor with image.
	 * @param image specified image.
	 */
	protected ImageList(Image image) {
		this.width = image.getWidth();
		this.height = image.getHeight();
		this.images.add(image);
	}

	
	@Override
	public int getWidth() {
		return width;
	}

	
	@Override
	public int getHeight() {
		return height;
	}

	
	/**
	 * Getting size of images.
	 * @return size of images.
	 */
	public int size() {
		return images.size();
	}
	
	
	/**
	 * Getting image at specified index.
	 * @param index specified index.
	 * @return image at specified index.
	 */
	public Image get(int index) {
		return images.get(index);
	}
	
	
	/**
	 * Setting size.
	 * @param size specified size.
	 */
	protected void setSize(int size) {
		if (size < 0 || images.size() == size) return;
		if (size == 0) {
			images.clear();
			return;
		}
		if (images.size() == 0) return;
		
		if (size > images.size()) {
			int n = size - images.size();
			Image image = images.get(images.size() - 1);
			for (int i = 0; i < n; i++) images.add(image);
		}
		else {
			List<Image> imageList = Util.newList(0);
			imageList.addAll(images);
			images.clear();
			images.addAll(imageList.subList(0, size));
		}
	}
	
	
	/**
	 * Getting default format of image.
	 * @return default format of image.
	 */
	public static String getDefaultFormat() {
		return Image.VIDEO_FORMAT_DEFAULT;
	}

	
	@Override
	public boolean save(Path path) {
		return saveAsGif(path);
	}


	/**
	 * Saving this images list as a gift file.
	 * @param path GIF file path.
	 * @return true if saving is successful.
	 */
	private boolean saveAsGif(Path path) {
		try {
			if (this.images.size() == 0) return false;
			OutputStream os = Files.newOutputStream(path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			
			AnimatedGifEncoder gif = new AnimatedGifEncoder();
			gif.start(os);
			gif.setDelay(delay);
			gif.setRepeat(0);
			for (Image image : images) {
				if (image instanceof ImageWrapper) gif.addFrame(((ImageWrapper)image).getImage());
			}
			gif.finish();
			
			os.close();;
			return true;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return false;
		
	}
	
	
	/**
	 * Converting neuron values to image list.
	 * @param values neuron values.
	 * @param neuronChannel neuron channel.
	 * @param width image width.
	 * @param height image height.
	 * @param depth image depth.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return image list converted from neuron values.
	 */
	public static ImageList convertFromNeuronValuesToImage(NeuronValue[] values, int neuronChannel, int width, int height, int depth,
			boolean isNorm, int defaultAlpha) {
		int wh = width*height;
		int length = wh*depth;
		if (values.length < length) return null;
		
		ImageList imageList = new ImageList();
		imageList.width = width;
		imageList.height = height;
		for (int i = 0; i < depth; i++) {
			NeuronValue[] iValues = Arrays.copyOfRange(values, i*wh, i*wh + wh);
			ImageWrapper image = ImageWrapper.convertFromNeuronValuesToImage(iValues, neuronChannel, width, height, isNorm, defaultAlpha);
			imageList.images.add(image);
		}
		
		return imageList;
	}

	
	@Override
	public NeuronValue[] convertFromImageToNeuronValues(int neuronChannel, int width, int height,
			boolean isNorm) {
		if (images.size() == 0) return null;
		
		NeuronValue[] values = null;
		for (Image image : images) {
			NeuronValue[] newValues = image.convertFromImageToNeuronValues(neuronChannel, width, height, isNorm);
			values = NeuronValue.concatArray(values, newValues);
		}
		
		return values;
	}

	
	/**
	 * Loading image list from specified path.
	 * @param path specified path which is directory or GIF file path (in this current version).
	 * @return image list loaded from path.
	 */
	public static ImageList load(Path path) {
		if (path == null) return null;
		if (Files.isDirectory(path))
			return loadFromDirectory(path);
		else
			return loadFromFile(path);
	}
	
	
	/**
	 * Loading image list from specified path.
	 * @param path specified path which is directory or GIF file path (in this current version).
	 * @return image list loaded from path.
	 */
	private static ImageList loadFromFile(Path path) {
		if (path == null) return null;
		if (Files.isDirectory(path)) return null;
		
		String fileName = path.getFileName().toString();
		String fileExt = null;
		if (fileName.contains(".")) fileExt = fileName.substring(fileName.lastIndexOf(".") + 1);
		if (fileExt == null || fileExt.isEmpty()) {
			ImageWrapper image = ImageWrapper.load(path);
			return image != null ? new ImageList(image) : null;
		}
		
		fileExt = fileExt.toLowerCase();
		if (fileExt.equals(Image.IMAGE_FORMAT_GIF.toLowerCase()))
			return loadFromGif(path);
		else if (fileExt.equals(Image.VIDEO_FORMAT_DEFAULT.toLowerCase()))
			return loadFromGif(path);
//		else if (fileExt.equals("mp4")) //Fixing later to load MP4.
//			return loadFromMP4(path);
		else {
			ImageWrapper image = ImageWrapper.load(path);
			return image != null ? new ImageList(image) : null;
		}
	}

	
	/**
	 * Load rasters from directory.
	 * @param directory source directory.
	 * @return image list loaded from directory.
	 */
	private static ImageList loadFromDirectory(Path directory) {
		if (directory == null || !Files.isDirectory(directory)) return null;
		
		File[] files = directory.toFile().listFiles();
		List<BufferedImage> frames = Util.newList(0);
		int width = 0, height = 0, n = 0;
		for (File file : files) {
			if (!file.isFile()) continue;
			try {
				BufferedImage frame = ImageIO.read(file);
				if (frame == null) continue;
				frames.add(frame);
				width += frame.getWidth();
				height += frame.getHeight();
				n++;
			} catch (Throwable e) {Util.trace(e);}
		}

		if (n == 0) return null;
		width /= n;
		height /= n;
		if (width < 1 || height < 1) return null;
		
		ImageList imageList = new ImageList();
		imageList.width = width;
		imageList.height = height;
		for (BufferedImage frame : frames) {
			frame = ImageWrapper.resize(frame, width, height);
			imageList.images.add(new ImageWrapper(frame));
		}
		return imageList;
	}


	/**
	 * Loading image list from GIF file path.
	 * @param path specific GIF file path.
	 * @return image list loaded from GIF file path.
	 */
	private static ImageList loadFromGif(Path path) {
		InputStream is = null;
		try {
			is = Files.newInputStream(path);
			
			GifDecoder gif = new GifDecoder();
			gif.read(is);
			int count = gif.getFrameCount(); 
			List<BufferedImage> frames = Util.newList(count);
			int width = 0, height = 0, delay = 0, n = 0;
			for (int i = 0; i < count; i++) {
				BufferedImage frame = gif.getFrame(i);
				if (frame == null) continue;
				frames.add(frame);
				width += frame.getWidth();
				height += frame.getHeight();
				delay += gif.getDelay(i);
				n++;
			}
			is.close();
			is = null;
			
			if (n == 0) return null;
			width /= n;
			height /= n;
			if (width < 1 || height < 1) return null;
			delay /= n;
			delay = delay < 1 ? DELAY : delay;
			
			ImageList imageList = new ImageList();
			imageList.width = width;
			imageList.height = height;
			imageList.delay = delay;
			for (BufferedImage frame : frames) {
				frame = ImageWrapper.resize(frame, width, height);
				imageList.images.add(new ImageWrapper(frame));
			}
			return imageList;
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		finally {
			try {
				if (is != null) is.close();
			} catch (Throwable e) {}
		}
		
		return null;

	}


	/**
	 * Creating image list from specified list of images.
	 * @param images specified list of images.
	 * @return image list created from specified list of images.
	 */
	public static ImageList create(Iterable<Image> images) {
		List<Image> frames = Util.newList(0);
		int width = 0, height = 0, n = 0;
		for (Image image : images) {
			if (image == null) continue;
			frames.add(image);
			width += image.getWidth();
			height += image.getHeight();
			n++;
		}
		if (n == 0) return null;
		width /= n;
		height /= n;
		if (width < 1 || height < 1) return null;
		
		ImageList imageList = new ImageList();
		imageList.width = width;
		imageList.height = height;
		for (Image frame : frames) {
			if (frame instanceof ImageWrapper) {
				BufferedImage image = ImageWrapper.resize(((ImageWrapper)frame).getImage(), width, height);
				imageList.images.add(new ImageWrapper(image));
			}
			else
				imageList.images.add(frame);
		}
		return imageList;
	}
	
	
	/**
	 * Extracting list of rasters.
	 * @return list of rasters.
	 */
	protected List<Raster> extractRasters() {
		List<Raster> rasters = Util.newList(0);
		for (Image image : images) rasters.add(Raster2DImpl.create(image));
		return rasters;
	}
	
	
}
