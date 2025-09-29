/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.conv.filter.ProductFilter2D;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;

/**
 * This utility class provides utility methods for matrix network layer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixLayerAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix network layer.
	 */
	protected MatrixLayerImpl layer = null;
	

	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	public MatrixLayerAssoc(MatrixLayerImpl layer) {
		this.layer = layer;
	}

	
	/**
	 * Filling matrix by specified value.
	 * @param matrix matrix.
	 * @param v value.
	 */
	private static void fill(NeuronValue[][] matrix, double v) {
		NeuronValue value = matrix[0][0].valueOf(v);
		int rows = matrix.length;
		for (int row = 0; row < rows; row++) {
			int columns = matrix[row].length;
			for (int column = 0; column < columns; column++) matrix[row][column] = value;
		}
	}

	
	/**
	 * Getting random value.
	 * @param rnd randomizer.
	 * @return random value.
	 */
	private static double r(Random rnd) {
		return rnd.nextDouble()*0.1 - 0.05;
	}
	
	
	/**
	 * Filling matrix by random values.
	 * @param matrix matrix.
	 * @param rnd randomizer.
	 */
	private static void fill(NeuronValue[][] matrix, Random rnd) {
		int rows = matrix.length;
		for (int row = 0; row < rows; row++) {
			int columns = matrix[row].length;
			for (int column = 0; column < columns; column++) {
				matrix[row][column] = matrix[row][column].valueOf(r(rnd));
			}
		}
	}

	
	/**
	 * Filling matrix by specified value.
	 * @param matrix matrix.
	 * @param v value.
	 */
	private static void fill(Matrix matrix, double v) {
		NeuronValue value = matrix.get(0, 0).valueOf(v);
		fill(matrix, value);
	}

	
	/**
	 * Filling matrix by specified value.
	 * @param matrix matrix.
	 * @param v value.
	 */
	private static void fill(Matrix matrix, NeuronValue v) {
		if (v != null) Matrix.fill(matrix, v);
	}

	
	/**
	 * Filling matrix by random value.
	 * @param matrix matrix.
	 * @param rnd randomizer.
	 */
	private static void fill(Matrix matrix, Random rnd) {
		for (int row = 0; row < matrix.rows(); row++) {
			for (int column = 0; column < matrix.columns(); column++) {
				NeuronValue value = matrix.get(row, column).valueOf(r(rnd));
				matrix.set(row, column, value);
			}
		}
	}
	
	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		if (layer.weight1 != null) fill(layer.weight1, v);
		if (layer.weight2 != null) fill(layer.weight2, v);
		if (layer.bias != null) fill(layer.bias, v);
		
		if (layer.filter != null && layer.filter instanceof ProductFilter2D) {
			ProductFilter2D filter = (ProductFilter2D)layer.filter;
			fill(filter.getKernel(), v);
			filter.setWeight(filter.getWeight().unit());
		}
		if (layer.filterBias != null) layer.filterBias = layer.filterBias.valueOf(v);
	}


//	/**
//	 * Initializing parameters
//	 */
//	private void initParams() {
//		if (layer.weight1 != null) {
//			NeuronValue unit = layer.weight1.get(0, 0).unit();
//			fill(layer.weight1, unit);
//		}
//		if (layer.weight2 != null) {
//			NeuronValue unit = layer.weight2.get(0, 0).unit();
//			fill(layer.weight2, unit);
//		}
//		if (layer.bias != null) {
//			NeuronValue zero = layer.bias.get(0, 0).zero();
//			fill(layer.bias, zero);
//		}
//		
//		if (layer.filter != null && layer.filter instanceof ProductFilter2D) {
//			ProductFilter2D filter = (ProductFilter2D)layer.filter;
//			double w = 1.0 / (double)(filter.height()*filter.width());
//			fill(filter.getKernel(), w);
//			filter.setWeight(filter.getWeight().unit());
//		}
//		if (layer.filterBias != null) layer.filterBias = layer.filterBias.zero();
//	}


	/**
	 * Initializing parameters.
	 */
	public void initParams() {
		Random rnd = new Random();
		if (layer.weight1 != null) fill(layer.weight1, rnd);
		if (layer.weight2 != null) fill(layer.weight2, rnd);
		if (layer.bias != null) fill(layer.bias, rnd);
		
		if (layer.filter != null && layer.filter instanceof ProductFilter2D) {
			ProductFilter2D filter = (ProductFilter2D)layer.filter;
			fill(filter.getKernel(), rnd);
			filter.setWeight(filter.getWeight().unit());
		}
		if (layer.filterBias != null) layer.filterBias = layer.filterBias.valueOf(r(rnd));
	}
	
	
	
	
}
