/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Scanner;

/**
 * This class provides utility methods for classifier.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
class ClassifierAssoc2 extends ClassifierAssoc {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with classifier.
	 * @param classifier specified classifier.
	 */
	public ClassifierAssoc2(Classifier classifier) {
		super(classifier);
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
			String line = scanner.nextLine().trim();
			if (!line.isBlank() && !line.isEmpty()) dataset = Integer.parseInt(line);
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

	
}
