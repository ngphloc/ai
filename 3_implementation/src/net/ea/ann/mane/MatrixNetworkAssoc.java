/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.rmi.RemoteException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster3DImpl;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterWrapper;
import net.ea.ann.raster.Size;

/**
 * This utility class provides utility methods for matrix neural network.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixNetworkAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix neural network.
	 */
	protected MatrixNetworkImpl mane = null;

	
	/**
	 * Constructor with matrix neural network.
	 * @param mane matrix neural network.
	 */
	public MatrixNetworkAssoc(MatrixNetworkImpl mane) {
		this.mane = mane;
	}

	
	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	public void initParams(Random rnd) {
		for (int i = 0; i < mane.layers.length; i++) {
			MatrixLayerAbstract layer = mane.layers[i];
			if (layer instanceof MatrixLayerImpl)
				new MatrixLayerAssoc((MatrixLayerImpl)layer).initParams(rnd);
		}
	}
	
	
	/**
	 * Initializing parameters.
	 */
	public void initParams() {
		initParams(new Random());
	}
	
	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		for (int i = 0; i < mane.layers.length; i++) {
			MatrixLayerAbstract layer = mane.layers[i];
			if (layer instanceof MatrixLayerImpl)
				new MatrixLayerAssoc((MatrixLayerImpl)layer).initParams(v);
		}
	}


	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	public int sizeOfParams() {
		int size = 0;
		for (int i = 0; i < mane.layers.length; i++) {
			if (!(mane.layers[i] instanceof MatrixLayerImpl)) continue;
			MatrixLayerImpl layer = (MatrixLayerImpl)mane.layers[i];
			if (layer.weight1 != null) size += layer.weight1.rows()*layer.weight1.columns();
			if (layer.weight2 != null) size += layer.weight2.rows()*layer.weight2.columns();
			if (layer.bias != null) size += layer.bias.rows()*layer.bias.columns();
			
			if (layer.filter != null) size += layer.filter.height()*layer.filter.width();
			if (layer.filterBias != null) size++;
		}
		return size;
	}
	
	
	/**
	 * Test of transformation.
	 * @param in input stream.
	 * @param out output stream.
	 * @throws RemoteException if any error raises.
	 */
	public static void transform(InputStream in, OutputStream out) throws Exception {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);

		int defaultZoomOut = 1;
		int zoomOut = defaultZoomOut;
		printer.print("Zoom out ratio (1, 2, 3,...) (default " + defaultZoomOut + "):");
		try {
			zoomOut = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(zoomOut)) zoomOut = defaultZoomOut;
		if (zoomOut < defaultZoomOut) zoomOut = defaultZoomOut;
		printer.println("Zoom out ratio is " + zoomOut + "\n");

		int defaultDepth = 3;
		int depth = defaultDepth;
		printer.print("Depth - size of neural network (2, 3,...) (default " + defaultDepth + "):");
		try {
			depth = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(depth)) depth = defaultDepth;
		if (depth < defaultDepth) depth = defaultDepth;
		printer.println("Depth is " + depth + "\n");

		int defaultRasterChannel = 3;
		int rasterChannel = defaultRasterChannel;
		printer.print("Raster channel (1, 2, 3, 4) (default " + defaultRasterChannel + "):");
		try {
			rasterChannel = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(rasterChannel)) rasterChannel = defaultRasterChannel;
		if (rasterChannel < defaultRasterChannel) rasterChannel = defaultRasterChannel;
		printer.println("Raster channel is " + rasterChannel + "\n");
		
		double defaultlr = Network.LEARN_RATE_DEFAULT/10;
		double lr = defaultlr;
		printer.print("Enter starting learning rate (default " + defaultlr + "):");
		try {
			lr = Double.parseDouble(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(lr)) lr = defaultlr;
		if (lr <= 0 || lr > 1) lr = defaultlr;
		printer.println("Starting learning rate is " + lr + "\n");

		int defaultMaxIteration = Network.LEARN_MAX_ITERATION_DEFAULT;
		int maxIteration = defaultMaxIteration;
		printer.print("Max iteration (default " + maxIteration + "):");
		try {
			maxIteration = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(maxIteration)) maxIteration = defaultMaxIteration;
		if (maxIteration < defaultMaxIteration) maxIteration = defaultMaxIteration;
		printer.println("Max iteration is " + maxIteration + "\n");

		boolean vectorized = false;
		printer.print("Vectorization (" + vectorized + " is default):");
		try {
			vectorized = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println("Vectorization is " + vectorized + "\n");

		boolean filtering = false;
		printer.print("Filtering (" + filtering + " is default):");
		try {
			filtering = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println("Filtering is " + filtering + "\n");

		boolean dual = false;
		printer.print("Dual mode (" + dual + " is default):");
		try {
			dual = Boolean.parseBoolean(scanner.nextLine().trim());
		} catch (Throwable e) {}
		printer.println("Dual mode is " + dual + "\n");

		printer.print("Enter source directory (" + Util.WORKING_DIRECTORY + "/base" + "):");
		String source = scanner.nextLine().trim();
		if (source.isEmpty()) source = Util.WORKING_DIRECTORY + "/base";
		printer.println("Source directory is \"" + source + "\".\n");
		Path sourceDir = Paths.get(source);
		if (!Files.exists(sourceDir) || !Files.isDirectory(sourceDir)) {
			printer.println("Wrong source directory");
			return;
		}
		
		printer.print("Enter target directory (" + Util.WORKING_DIRECTORY + "/test" + "):");
		String target = scanner.nextLine().trim();
		if (target.isEmpty()) target = Util.WORKING_DIRECTORY + "/test";
		printer.println("Target directory is \"" + target + "\".\n");
		Path targetDir = Paths.get(target);
		try {
			if (!Files.exists(targetDir)) Files.createDirectory(targetDir);
			if (!Files.isDirectory(targetDir)) {
				printer.println("Wrong target directory");
				return;
			}
		} catch (Throwable e) {Util.trace(e);}

		printer.print("Enter generation directory (" + Util.WORKING_DIRECTORY + "/gen" + "):");
		String gen = scanner.nextLine().trim();
		if (gen.isEmpty()) gen = Util.WORKING_DIRECTORY + "/gen";
		printer.println("Generating directory is \"" + gen + "\".\n");
		Path genDir = Paths.get(gen);
		try {
			if (!Files.exists(genDir)) Files.createDirectory(genDir);
		} catch (Throwable e) {Util.trace(e);}


		List<Raster> sourceRasters = RasterAssoc.load(sourceDir);
		List<Raster> targetRasters = RasterAssoc.load(targetDir);
		if (sourceRasters.size() == 0 || targetRasters.size() == 0) {
			printer.println("Empty base directory or empty test directory");
			return;
		}
		
		printer.println("Running...");

		//Initializing matrix neural network.
		MatrixNetworkImpl mane = new MatrixNetworkImpl(defaultRasterChannel);
		mane.getConfig().put(NetworkAbstract.LEARN_MAX_ITERATION_FIELD, maxIteration);
		mane.getConfig().put(NetworkAbstract.LEARN_RATE_FIELD, lr);
		mane.paramSetVectorized(vectorized);
		Filter2D filter = filtering ? mane.defaultFilter(new Size(MatrixNetworkImpl.BASE_DEFAULT, MatrixNetworkImpl.BASE_DEFAULT)) : null;
		//
		Size sourceSize = RasterAssoc.getAverageSize(sourceRasters).divide(zoomOut);
		Size targetSize = RasterAssoc.getAverageSize(targetRasters).divide(zoomOut);
		boolean initialized = true;
		if (dual)
			initialized = new MatrixNetworkInitializer(mane).initializeDual(sourceSize, targetSize, filter, depth);
		else
			initialized = new MatrixNetworkInitializer(mane).initialize(sourceSize, targetSize, filter, depth);
		if (!initialized) {
			printer.println("Error in initializing network");
			mane.close();
			return;
		}
		
		//Learning matrix neural network.
		int n = Math.min(sourceRasters.size(), targetRasters.size());
		List<Raster[]> sample = Util.newList(n);
		for (int i = 0; i < n; i++) sample.add(new Raster[] {sourceRasters.get(i), targetRasters.get(i)});
		mane.learnByRaster(sample);
		
		//Generating transformed rasters.
		for (int i = 0; i < sourceRasters.size(); i++) {
			Raster raster = sourceRasters.get(i);
			mane.evaluate(raster);
			
			List<Raster> rasters = Util.newList(0);
			for (int j = 0; j < mane.size(); j++) {
				MatrixLayerAbstract layer = mane.get(j);
				Raster outputRaster = layer.toRaster(layer.queryOutput());
				rasters.add(outputRaster);
			}
			Raster outputRaster = Raster3DImpl.createByRasters(rasters);
			
			String name = "trans" + "-deep" + (mane.size()-1) + "-lr" + Util.format(lr) + "-maxiter" + maxIteration + "-" + (i+1);
			String genName = raster instanceof RasterWrapper ? name + "." + ((RasterWrapper)raster).getNamePlain() : name;
			Path path = RasterAssoc.genDefaultPath(genDir, genName, outputRaster.getDefaultFormat());
			outputRaster.save(path);
			//
			path = RasterAssoc.genDefaultPath(genDir, genName, rasters.get(rasters.size()-1).getDefaultFormat());
			rasters.get(rasters.size()-1).save(path);
		}
		
		mane.close();
		printer.println("Finished.");
	}

	
}
