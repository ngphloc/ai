/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.rmi.RemoteException;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.DeconvConvFilter2DImpl;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.ProductFilter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.TextParsable;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2DImpl;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of convolutional network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvNetworkImpl extends ConvNetworkAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	protected ConvNetworkImpl(int neuronChannel, Function activateRef, Id idRef) {
		super(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Default constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 */
	protected ConvNetworkImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}


	@Override
	public ConvLayerSingle newLayer(Size size, Filter filter) {
		ConvLayerSingle convLayer = null;
		if (size.time > 1)
			convLayer = ConvLayer4DImpl.create(neuronChannel, activateRef, size.width, size.height, size.depth, size.time, filter);
		else if (size.depth > 1)
			convLayer = ConvLayer3DImpl.create(neuronChannel, activateRef, size.width, size.height, size.depth, filter);
		else if (size.height > 1)
			convLayer = ConvLayer2DImpl.create(neuronChannel, activateRef, size.width, size.height, filter);
		else if (size.width > 1)
			convLayer = ConvLayer1DImpl.create(neuronChannel, activateRef, size.width, filter);
		else
			convLayer = ConvLayer1DImpl.create(neuronChannel, activateRef, size.width, filter);
			
		convLayer.setPadZeroFilter(isPadZeroFilter);
		return convLayer;
	}

	
	/**
	 * Creating convolutional network with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return convolutional network.
	 */
	public static ConvNetworkImpl create(int neuronChannel, Function activateRef, Id idRef) {
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ConvNetworkImpl(neuronChannel, activateRef, idRef);
	}

	
	/**
	 * Creating convolutional network with neuron channel and activation function.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @return convolutional network.
	 */
	public static ConvNetworkImpl create(int neuronChannel, Function activateRef) {
		return create(neuronChannel, activateRef, null);
	}
	
	
	/**
	 * Learning filter.
	 * @param in input stream.
	 * @param out output stream.
	 * @throws RemoteException if any error raises.
	 */
	public static void learnFilter(InputStream in, OutputStream out) throws RemoteException {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);

		double defaultlr = 1;
		double lr = defaultlr;
		printer.print("Enter starting learning rate (default 1.0):");
		try {
			lr = Double.parseDouble(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(lr)) lr = defaultlr;
		if (lr <= 0 | lr > 1) lr = defaultlr;
		printer.println("Starting learning rate is " + lr + "\n");

		printer.print("Enter base directory (" + Util.WORKING_DIRECTORY + "/base" + "):");
		String base = scanner.nextLine().trim();
		if (base.isEmpty()) base = Util.WORKING_DIRECTORY + "/base";
		printer.println("Base directory is \"" + base + "\".\n");
		Path baseDir = Paths.get(base);
		if (!Files.exists(baseDir) || !Files.isDirectory(baseDir)) {
			printer.println("Wrong base directory");
			return;
		}
		
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

		List<Raster> baseRasters = RasterAssoc.load(baseDir);
		if (baseRasters.size() == 0) {
			printer.println("Empty base directory.");
			return;
		}
		
		ConvNetworkImpl conv = ConvNetworkImpl.create(3, Raster.toConvActivationRef(3, true));
		ConvNetworkImpl deconv = ConvNetworkImpl.create(3, Raster.toConvActivationRef(3, true));
		NeuronValueCreator creator = conv.newLayer(new Size(1, 1, 1, 1), null);
		
		ProductFilter2D f = null;
		Map<String, ProductFilter2D> filters = Util.newMap(0);
		
		//Blur filter
		f = ProductFilter2D.create(new double[][] {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, 1.0/9.0, creator);
		filters.put("blur", f);
		
		//Sharpening filter
		f = ProductFilter2D.create(new double[][] {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}, 1, creator);
		filters.put("sharpening", f);
		
		//Edge detection filter
		f = ProductFilter2D.create(new double[][] {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}, 1.0, creator);
		filters.put("edge-detection", f);
	
		BufferedWriter writer = null;
		try {
			writer = Files.newBufferedWriter(testresultDir.resolve("TestResult" + System.currentTimeMillis() + ".txt"),
				StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		} catch (Throwable e) {Util.trace(e);}

		while (lr >= 0.01) {
			try {
				writer.write("\nLearning rate " + Util.format(lr) + "\n");
			} catch (Throwable e) {Util.trace(e);}

			Set<String> filterNames = filters.keySet();
			for (String filterName : filterNames) {
				List<NeuronValue[]> dataList = Util.newList(0);
				List<Raster> raster1List = Util.newList(0);
				Filter filter = filters.get(filterName);
				BiasFilter newFilter = null;
				
				for (int j = 0; j < baseRasters.size(); j++) {
					Raster raster = baseRasters.get(j);
					conv.initialize(new Size(raster.getWidth(), raster.getHeight(), 1, 1), new Filter[] {filter});
					
					NeuronValue[] data1 = conv.evaluateRaster(raster);
					ConvLayer2DAbstract convOutputLayer = (ConvLayer2DAbstract)conv.unifyOutputContent();
					Raster raster1 = convOutputLayer.createRaster(data1,
						true, Image.ALPHA_DEFAULT);
					Path path1 = RasterAssoc.genDefaultPath(genDir, "conv." + "lr" + Util.format(lr) + "." + filterName + ".image" + (j+1), raster.getDefaultFormat());
					raster1.save(path1);
					raster1List.add(raster1);
	
					ConvLayer2DAbstract convInputLayer = (ConvLayer2DAbstract)conv.convLayers.get(0);
					newFilter = convInputLayer.learnFilter(newFilter, false, lr, 1);
					
					NeuronValue[] data = ((ConvLayer2DAbstract)conv.convLayers.get(0)).getData();
					dataList.add(data);
				} // End rasters
				
				double MAE0 = 0, MAE = 0;
				int n = 0;
				for (int j = 0; j < dataList.size(); j++) {
					Raster raster = baseRasters.get(j);
					deconv.initialize(new Size(raster.getWidth()/3, raster.getHeight()/3, 1, 1), new Filter[] {DeconvConvFilter2DImpl.create((ProductFilter2D)newFilter.filter)});
					
					Raster raster1 = raster1List.get(j);
					NeuronValue[] data2 = deconv.evaluateRaster(raster1);
					Raster raster2 = Raster2DImpl.create(data2, 3, new Size(raster.getWidth(), raster.getHeight()),
							true, Image.ALPHA_DEFAULT);
					Path path2 = RasterAssoc.genDefaultPath(genDir, "deconv." + "lr" + Util.format(lr) + "." + filterName + ".image" + (j+1), raster.getDefaultFormat());
					raster2.save(path2);
					
					NeuronValue[] data = dataList.get(j);
					NeuronValue[] data1 = raster1.toNeuronValues(3, new Size(raster.getWidth(), raster.getHeight(), 1, 1),
						true);
					data2 = raster2.toNeuronValues(3, new Size(raster.getWidth(), raster.getHeight(), 1, 1),
						true);
					
					double mae0 = 0, mae = 0;
					for (int k = 0; k < data.length; k++) {
						NeuronValueV bias1 = (NeuronValueV)(data1[k].subtract(data[k]));
						mae0 += (Math.abs(bias1.get(0)) + Math.abs(bias1.get(1)) + Math.abs(bias1.get(2))) / 3;
	
						NeuronValueV bias2 = (NeuronValueV)(data2[k].subtract(data[k]));
						mae += (Math.abs(bias2.get(0)) + Math.abs(bias2.get(1)) + Math.abs(bias2.get(2))) / 3;
					}
					mae0 /= data.length;
					mae /= data.length;
					
					MAE0 += mae0;
					MAE += mae;
					n++;
				} //End data list
				
				MAE0 /= n;
				MAE /= n;
				double loss = Math.abs(((MAE-MAE0) / MAE0) * 100);
				String newFilterText = newFilter instanceof TextParsable ? ((TextParsable)newFilter).toText() : "";
				try {
					writer.write("Filter " + filterName + " has\n" +
						"    MAE    = " + Util.format(MAE) + "\n" + 
						"    MAE0   = " + Util.format(MAE0) + "\n" +
					    "    loss   = " + Util.format(loss) + "%\n" +
			    		"    filter = " + newFilterText + "\n\n");
				} catch (Throwable e) {Util.trace(e);}
			}
			
			try {
				writer.flush();
			} catch (Throwable e) {Util.trace(e);}
			
			lr = lr > 0.15 ? lr-0.1 : lr-0.01;
		} //End learning rate

	}
	
	
//	/**
//	 * Main method.
//	 * @param args arguments.
//	 */
//	public static void main(String[] args) {
//		try {
//			learnFilter(System.in, System.out);
//		}
//		catch (Throwable e) {Util.trace(e);}
//	}


}



