/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import net.ea.ann.classifier.ClassifierBuilder.ClassifierModel;
import net.ea.ann.core.Network;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.MatrixNetworkAssoc;
import net.ea.ann.mane.MatrixNetworkImpl;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;

/**
 * This class provides utility methods for classifier.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ClassifierAssoc implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal classifier.
	 */
	protected Classifier classifier = null;
	
	
	/**
	 * Constructor with classifier.
	 * @param classifier specified classifier.
	 */
	public ClassifierAssoc(Classifier classifier) {
		this.classifier = classifier;
	}


	/**
	 * Getting size of parameters.
	 * @return size of parameters.
	 */
	public int sizeOfParams() {
		int size = 0;
		if (classifier instanceof MatrixClassifier) {
			MatrixClassifier mac = (MatrixClassifier)classifier;
			size = new MatrixNetworkAssoc(mac).sizeOfParams();
			if (mac.adjuster != null) size += new MatrixNetworkAssoc(mac.adjuster).sizeOfParams();
		}
		else if (classifier instanceof MatrixNetworkImpl) {
			size = new MatrixNetworkAssoc((MatrixNetworkImpl)classifier).sizeOfParams();
		}
		return size;
	}
	
	
	/**
	 * Getting depth.
	 * @return depth.
	 */
	public int depth() {
		int depth = 0;
		if (classifier instanceof MatrixClassifier)
			depth = ((MatrixClassifier)classifier).size() - 1;
		else if (classifier instanceof MatrixNetworkImpl)
			depth = ((MatrixNetworkImpl)classifier).size() - 1;
		return depth;
	}
	
	
	/**
	 * This class represents classified information.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class ClassifyParams implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Classifier model.
		 */
		public String model = ClassifierModel.mac.toString();
		
		/**
		 * Learning rate.
		 */
		public double learningRate = Network.LEARN_RATE_DEFAULT;
		
		/**
		 * Number of batches.
		 */
		public int batches = Network.LEARN_MAX_ITERATION_DEFAULT;
		
		/**
		 * Including convolutional neural network.
		 */
		public boolean conv = MatrixClassifier0.CONV_DEFAULT;
		
		/**
		 * Vectorization mode.
		 */
		public boolean vectorized = MatrixNetworkAbstract.VECTORIZED_DEFAULT;
		
		/**
		 * Adjustment mode.
		 */
		public boolean adjust = MatrixClassifier.ADJUST_DEFAULT;
		
		/**
		 * Dual mode.
		 */
		public boolean dual = MatrixClassifier0.DUAL_DEFAULT;
		
		/**
		 * Baseline mode.
		 */
		public boolean baseline = MatrixClassifier0.BASELINE_DEFAULT;

		/**
		 * Dataset name.
		 */
		public String dataset = "cifar10";
		
		/**
		 * Maximum iteration.
		 */
		public int maxIteration = Network.LEARN_MAX_ITERATION_DEFAULT;
		
		/**
		 * Depth.
		 */
		public int depth = 0;

		/**
		 * Parametric size.
		 */
		public int paramSize = 0;
		
		/**
		 * Time in miliseconds.
		 */
		public long time = 0;
		
		/**
		 * Default constructor.
		 */
		public ClassifyParams() {
			
		}
		
		/**
		 * Importing parameters from builder.
		 * @param builder builder
		 */
		public void importParams(ClassifierBuilder builder) {
			this.model = builder.model.toString();
			this.learningRate = builder.learningRate;
			this.batches = builder.batches;
			this.conv = builder.conv;
			this.vectorized = builder.vectorized;
			this.adjust = builder.adjust;
			this.dual = builder.dual;
			this.baseline = builder.baseline;
		}
		
	}

	
	/**
	 * This class represents classified information.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class ClassifyInfo implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Total number of classifications.
		 */
		public int N = 0;
		
		/**
		 * Total number of corrections.
		 */
		public int correctTotal = 0;
		
		/**
		 * Map of source labels.
		 */
		public Map<String, Integer> sourceLabelMap = Util.newMap(0);
		
		/**
		 * Map of result labels.
		 */
		public Map<String, Integer> resultLabelMap = Util.newMap(0);
		
		/**
		 * Map of correct labels.
		 */
		public Map<String, Integer> correctLabelMap = Util.newMap(0);

		/**
		 * Resetting information.
		 */
		public void reset() {
			this.N = 0;
			this.correctTotal = 0;
			this.sourceLabelMap.clear();; 
			this.resultLabelMap.clear(); 
			this.correctLabelMap.clear();; 
		}
		
		/**
		 * Collecting results.
		 * @param sources source rasters.
		 * @param results result rasters.
		 * @return true if analyzing is successful.
		 */
		public void collect(List<Raster> sources, List<Raster> results) {
			reset();
			int n = Math.min(sources.size(), results.size());
			if (n == 0) return;
			
			int labelCount = 0;
			for (int i = 0; i < n; i++) {
				Raster source = sources.get(i);
				Raster result = results.get(i);
				RasterProperty sourceProperty = source.getProperty();
				RasterProperty resultProperty = result.getProperty();
				int count = Math.min(sourceProperty.getLabelCount(), resultProperty.getLabelCount());
				if (i > 0)
					labelCount = Math.min(labelCount, count);
				else
					labelCount = count;
			}
			if (labelCount == 0) return;

			for (int i = 0; i < n; i++) {
				Raster source = sources.get(i);
				Raster result = results.get(i);
				RasterProperty sourceProperty = source.getProperty();
				RasterProperty resultProperty = result.getProperty();
				int correct = 0;
				for (int l = 0; l < labelCount; l++) {
					int sourceLabelId = sourceProperty.getLabelId(l);
					int resultLabelId = resultProperty.getLabelId(l);
					if (sourceLabelId == resultLabelId) correct++;
					
					String sourceLabelIdx = "" + l + "$" + sourceLabelId;
					String resultLabelIdx = "" + l + "$" + resultLabelId;
					int sourceCount = this.sourceLabelMap.containsKey(sourceLabelIdx) ? this.sourceLabelMap.get(sourceLabelIdx) : 0;
					this.sourceLabelMap.put(sourceLabelIdx, sourceCount+1);
					int resultCount = this.resultLabelMap.containsKey(resultLabelIdx) ? this.resultLabelMap.get(resultLabelIdx) : 0;
					this.resultLabelMap.put(resultLabelIdx, resultCount+1);
					if (sourceLabelId == resultLabelId) {
						int correctCount = correctLabelMap.containsKey(resultLabelIdx) ? correctLabelMap.get(resultLabelIdx) : 0;
						this.correctLabelMap.put(resultLabelIdx, correctCount+1);
					}
					this.N += labelCount;
				}
				this.correctTotal += correct;
			}
			
		} //End collection.
	
		/**
		 * Accumulating with other information.
		 * @param info other information.
		 */
		public void accum(ClassifyInfo info) {
			this.N += info.N;
			this.correctTotal += info.correctTotal;
			
			Set<String> labelSet = Util.newSet(0);
			labelSet.addAll(info.sourceLabelMap.keySet());
			labelSet.addAll(info.resultLabelMap.keySet());
			for (String label : labelSet) {
				if (info.sourceLabelMap.containsKey(label)) {
					int sourceCount = this.sourceLabelMap.containsKey(label) ?
						this.sourceLabelMap.get(label) + info.sourceLabelMap.get(label) : info.sourceLabelMap.get(label);
					this.sourceLabelMap.put(label, sourceCount);
				}
				
				if (info.resultLabelMap.containsKey(label)) {
					int resultCount = this.resultLabelMap.containsKey(label) ?
						this.resultLabelMap.get(label) + info.resultLabelMap.get(label) : info.resultLabelMap.get(label);
					this.resultLabelMap.put(label, resultCount);
				}
				
				if (info.correctLabelMap.containsKey(label)) {
					int correctCount = this.correctLabelMap.containsKey(label) ?
						this.correctLabelMap.get(label) + info.correctLabelMap.get(label) : info.correctLabelMap.get(label);
					this.correctLabelMap.put(label, correctCount);
				}
			}
		}
		
		/**
		 * Accumulating information. 
		 * @param infos array of information.
		 * @return accumulated information.
		 */
		public static ClassifyInfo accum(ClassifyInfo...infos) {
			if (infos == null || infos.length == 0) return null;
			ClassifyInfo accum = new ClassifyInfo();
			for (ClassifyInfo info : infos) accum.accum(info);
			return accum;
		}
		
	}
	
	
	/**
	 * Test of classification.
	 * @param in input stream.
	 * @param out output stream.
	 */
	public static void classify(InputStream in, OutputStream out) {
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);

		int defaultDataset = 0;
		int dataset = defaultDataset;
		printer.print("Dataset (0-cifar10) (default " + defaultDataset + " is cifar10):");
		try {
			dataset = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(dataset)) dataset = defaultDataset;
		if (dataset <= 0) dataset = defaultDataset;
		printer.println("Dataset is " + dataset + "\n");

		switch (dataset) {
		case 0:
			classifyCIFAR10(in, out);
			break;
		default:
			classifyCIFAR10(in, out);
			break;
		}
	}

	
	/**
	 * Test of classification with CIFAR-10 dataset.
	 * @param in input stream.
	 * @param out output stream.
	 */
	static void classifyCIFAR10(InputStream in, OutputStream out) {
		ClassifierBuilder builder = ClassifierBuilder.enter(in, out);
		if (builder == null) return;
		@SuppressWarnings("resource")
		Scanner scanner = new Scanner(in);
		PrintStream printer = new PrintStream(out);

		int defaultMaxIteration = 1;
		int maxIteration = defaultMaxIteration;
		printer.print("Maximum iteration (default " + defaultMaxIteration + "):");
		try {
			maxIteration = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(maxIteration)) maxIteration = defaultMaxIteration;
		if (maxIteration <= 0) maxIteration = defaultMaxIteration;
		printer.println("Maximum iteration is " + maxIteration + "\n");
	
		int defaultTrainSize = -1;
		int trainSize = defaultTrainSize;
		printer.print("Training size (default " + defaultTrainSize + "):");
		try {
			trainSize = Integer.parseInt(scanner.nextLine().trim());
		} catch (Throwable e) {}
		if (Double.isNaN(trainSize)) trainSize = defaultTrainSize;
		if (trainSize < 0) trainSize = defaultTrainSize;
		printer.println("Training size is " + trainSize + "\n");
	
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
		
		List<List<Raster>> baseRastersList = Util.newList(0);
		List<List<Raster>> testRastersList = Util.newList(0);
		try {
			final int size = trainSize;
			Files.list(baseDir).filter(Files::isRegularFile).forEach((basePath) -> {
				List<Raster> baseRasters = RasterAssoc.loadCIFAR(basePath, size);
				if (baseRasters.size() > 0) baseRastersList.add(baseRasters);
			});
	
			Files.list(testDir).filter(Files::isRegularFile).forEach((testPath) -> {
				List<Raster> testRasters = RasterAssoc.loadCIFAR(testPath, size);
				if (testRasters.size() > 0) testRastersList.add(testRasters);
			});
		} catch (Exception e) {Util.trace(e);}
		if (baseRastersList.size() == 0 || testRastersList.size() == 0) return;
	
		int minBaseSize = baseRastersList.get(0).size();
		for (List<Raster> baseRasters : baseRastersList) minBaseSize = Math.min(minBaseSize, baseRasters.size());
		if (builder.getBatches() > minBaseSize) {
			builder.setBatches(1);
			printer.println("Batches are re-calculated as " + builder.getBatches() + "\n");
		}
	
		Classifier classifier = builder.build();
		ClassifyInfo info = new ClassifyInfo();
		long time = 0;
		SimpleDateFormat df = new SimpleDateFormat(Util.DATE_FORMAT);
		System.out.println("Begin task at " + df.format(new Date()));
		for (int iteration = 0; iteration < maxIteration; iteration++) {
			for (List<Raster> baseRasters : baseRastersList) {
				for (List<Raster> sources : testRastersList) {
					try {
						long beginTime = System.currentTimeMillis();
						classifier.learnRaster(baseRasters);
						List<Raster> results = classifier.classify(sources);
						long endTime = System.currentTimeMillis();
						time += endTime - beginTime;
						
						ClassifyInfo infoOne = new ClassifyInfo();
						infoOne.collect(sources, results);
						info.accum(infoOne);
					} catch (Throwable e) {Util.trace(e);}
				}
			}
		}
		System.out.println("End task at " + df.format(new Date()));
	
		ClassifyParams params = new ClassifyParams();
		params.importParams(builder);
		params.dataset = "cifar10";
		params.maxIteration = maxIteration;
		params.depth = new ClassifierAssoc(classifier).depth();
		params.paramSize = new ClassifierAssoc(classifier).sizeOfParams();
		params.time = time;
		try {
			String classifiedName = RasterAssoc.genDefaultName(params.model + "-" + Util.format(params.learningRate) + "-" + "stat", null);
			BufferedWriter csvWriter = Files.newBufferedWriter(testresultDir.resolve(classifiedName + ".csv"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			saveClassifyInfo(csvWriter, info, params);
			csvWriter.close();
	
		} catch (Throwable e) {Util.trace(e);}
			
	}


	/**
	 * Saving classification information.
	 * @param writer writer.
	 * @param info classification information.
	 * @param params classification parameters.
	 * @param builder builder.
	 * @param model classifier model.
	 * @param dataset dataset.
	 */
	public static void saveClassifyInfo(Writer writer, ClassifyInfo info, ClassifyParams params) {
		Set<String> labelSet = Util.newSet(0);
		labelSet.addAll(info.sourceLabelMap.keySet());
		labelSet.addAll(info.resultLabelMap.keySet());
		if (labelSet.size() == 0) return;
		
		List<String> labelList = Util.newList(0);
		labelList.addAll(labelSet);
		Collections.sort(labelList);
		double[] precises = new double[labelList.size()];
		double[] recalls = new double[labelList.size()];
		double[] f1s = new double[labelList.size()];
		Arrays.fill(precises, 0);
		Arrays.fill(recalls, 0);
		Arrays.fill(f1s, 0);
		for (int i = 0; i < labelList.size(); i++) {
			String labelIdx = labelList.get(i);
			int sourceCount = 0, resultCount = 0, correctCount = 0;
			if (info.sourceLabelMap.containsKey(labelIdx)) sourceCount = info.sourceLabelMap.get(labelIdx);
			if (info.resultLabelMap.containsKey(labelIdx)) resultCount = info.resultLabelMap.get(labelIdx);
			if (info.correctLabelMap.containsKey(labelIdx)) correctCount = info.correctLabelMap.get(labelIdx);
			
			if (resultCount > 0) precises[i] = (double)correctCount/(double)resultCount;
			if (sourceCount > 0) {
				recalls[i] = (double)resultCount/(double)sourceCount;
				recalls[i] = recalls[i] > 1 ? 1 : recalls[i];
			}
			f1s[i] = precises[i] + recalls[i] != 0 ? 2*precises[i]*recalls[i] / (precises[i] + recalls[i]) : 0;
		}
		
		double accuracy = (double)info.correctTotal/(double)info.N;
		double preciseMean = NeuronValueV.mean(precises);
		double preciseVar = NeuronValueV.variance(precises);
		double recallMean = NeuronValueV.mean(recalls);
		double recallVar = NeuronValueV.variance(recalls);
		double f1Mean = NeuronValueV.mean(f1s);
		double f1Var = NeuronValueV.variance(f1s);
		double labelCoverage = 0;
		for (int i = 0; i < recalls.length; i++) labelCoverage += recalls[i] != 0 ? 1 : 0;
		labelCoverage /= recalls.length;
		
		try {
			StringBuffer header = new StringBuffer();
			header.append("learning rate, iterations, batches, classes, N, accuracy, precise.mean, precise.var, recall.mean, recall.var, f1.mean, f1.var, label coverage, ");
			for (int i = 0; i < labelList.size(); i++) {
				header.append("precise$" + labelList.get(i) + ", ");
			}
			for (int i = 0; i < labelList.size(); i++) {
				header.append("recall$" + labelList.get(i) + ", ");
			}
			for (int i = 0; i < labelList.size(); i++) {
				header.append("f1$" + labelList.get(i) + ", ");
			}
			header.append("time (s), depth, parametric size, note");
			writer.write(header.toString() + "\n");
			writer.flush();
			
			StringBuffer result = new StringBuffer();
			result.append(Util.format(params.learningRate) + ", " + params.maxIteration + ", " + params.batches + ", " +
				labelList.size() + ", " + info.N + ", " + Util.format(accuracy) + ", ");
			result.append(Util.format(preciseMean) + ", " + Util.format(preciseVar) + ", ");
			result.append(Util.format(recallMean) + ", " + Util.format(recallVar) + ", ");
			result.append(Util.format(f1Mean) + ", " + Util.format(f1Var) + ", ");
			result.append(Util.format(labelCoverage) + ", ");
			
			for (int i = 0; i < labelList.size(); i++) {
				result.append(Util.format(precises[i]) + ", ");
			}
			for (int i = 0; i < labelList.size(); i++) {
				result.append(Util.format(recalls[i]) + ", ");
			}
			for (int i = 0; i < labelList.size(); i++) {
				result.append(Util.format(f1s[i]) + ", ");
			}
			
			result.append(Util.format((double)params.time/1000.0) + ", ");
			result.append(params.depth + ", ");
			result.append(params.paramSize + ", ");
			result.append("model=" + params.model + "~dataset=" + params.dataset +
				"~conv=" + params.conv +
				"~vec=" + params.vectorized +
				"~dual=" + params.dual +
				"~adjust=" + params.adjust +
				"~baseline=" + params.baseline + "\n");
			writer.write(result.toString() + "\n");
			writer.flush();
		} catch (Throwable e) {Util.trace(e);}
	}

	
	/**
	 * Saving classification information.
	 * @param writer writer.
	 * @param sources source rasters.
	 * @param results result rasters.
	 */
	public static void saveClassifyInfo(Writer writer, List<Raster> sources, List<Raster> results) {
		int n = Math.min(sources.size(), results.size());
		if (n == 0) return;
		int labelCount = 0;
		for (int i = 0; i < n; i++) {
			Raster source = sources.get(i);
			Raster result = results.get(i);
			RasterProperty sourceProperty = source.getProperty();
			RasterProperty resultProperty = result.getProperty();
			int count = Math.min(sourceProperty.getLabelCount(), resultProperty.getLabelCount());
			if (i > 0)
				labelCount = Math.min(labelCount, count);
			else
				labelCount = count;
		}
		if (labelCount == 0) return;

		try {
			writer.write("id, training classes, classified classes, correction\n");
			writer.flush();
		} catch (Throwable e) {Util.trace(e);}
		
		for (int i = 0; i < n; i++) {
			Raster source = sources.get(i);
			Raster result = results.get(i);
			RasterProperty sourceProperty = source.getProperty();
			RasterProperty resultProperty = result.getProperty();
			String sourceLabelText = "";
			String resultLabelText = "";
			int correct = 0;
			for (int l = 0; l < labelCount; l++) {
				if (l > 0) {
					sourceLabelText += "~";
					resultLabelText += "~";
				}
				int sourceLabelId = sourceProperty.getLabelId(l);
				int resultLabelId = resultProperty.getLabelId(l);
				if (sourceLabelId == resultLabelId) correct++;
				sourceLabelText += "" + sourceLabelId;
				resultLabelText += "" + resultLabelId;
			}
			
			try {
				writer.write(source.id() + ", " + sourceLabelText + ", " + resultLabelText + ", " + correct + "\n");
				writer.flush();
			} catch (Throwable e) {Util.trace(e);}
		}

	
	}
	
	
}
