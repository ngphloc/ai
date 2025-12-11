/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.util.Random;

import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Size;

/**
 * This class represents matrix stack.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixStack implements Matrix {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal matrices.
	 */
	protected Matrix[] matrices = null;

	
	/**
	 * Constructor with matrices.
	 * @param matrices matrices.
	 */
	public MatrixStack(Matrix...matrices) {
		if (!checkValid(matrices)) throw new IllegalArgumentException();
		this.matrices = matrices;
	}

	
	/**
	 * Checking matrices.
	 * @param matrices specific matrices.
	 * @return true if matrices are valid.
	 */
	private static boolean checkValid(Matrix[] matrices) {
		if (matrices == null || matrices.length == 0) return false;
		int columns = matrices[0].columns();
		int rows = matrices[0].rows(); 
		if (columns <= 0 || rows <= 0) return false;
		for (int i = 1; i < matrices.length; i++) {
			if (matrices[i].columns() != columns || matrices[i].rows() != rows) return false;
		}
		return true;
	}

	
	@Override
	public NeuronValue newNeuronValue() {return matrices[0].newNeuronValue();}


	/**
	 * Getting depth.
	 * @return depth.
	 */
	public int depth() {return matrices.length;}
	
	
	/**
	 * Getting matrix at specified index.
	 * @param index specified index.
	 * @return matrix at specified index.
	 */
	public Matrix get(int index) {return matrices[index];}
	
	
	/**
	 * Getting first matrix.
	 * @return first matrix.
	 */
	public Matrix get() {return get(0);}
	
	
	@Override
	public int rows() {return matrices[0].rows();}

	
	@Override
	public int columns() {return matrices[0].columns();}

	
	@Override
	public NeuronValue get(int row, int column) {
		if (depth() == 1) return matrices[0].get(row, column);
		throw new RuntimeException();
	}

	
	@Override
	public void set(int row, int column, NeuronValue value) {
		if (depth() == 1) {
			matrices[0].set(row, column, value);
			return;
		}
		throw new RuntimeException();
	}

	
	@Override
	public Matrix getRow(int row) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].getRow(row);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix getColumn(int column) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].getColumn(column);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix getColumns(int column, int range) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].getColumns(column, range);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix transpose() {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].transpose();
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix negative0() {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].negative0();
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix add(Matrix other) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].add(((MatrixStack)other).matrices[i]);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix subtract(Matrix other) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].subtract(((MatrixStack)other).matrices[i]);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix multiply(Matrix other) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].multiply(((MatrixStack)other).matrices[i]);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix multiply0(NeuronValue value) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].multiply0(value);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix multiply0(double value) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].multiply0(value);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix multiplyWise(Matrix other) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].multiplyWise(((MatrixStack)other).matrices[i]);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix divide0(NeuronValue value) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].divide0(value);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix divide0(double value) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].divide0(value);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix kroneckerProductRowOf(Matrix other, int rowOfThis) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].kroneckerProductRowOf(((MatrixStack)other).matrices[i], rowOfThis);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix evaluate0(Function f) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].evaluate0(f);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix derivativeWise(Function f) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].derivativeWise(f);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix concatHorizontal(Matrix...matrices) {
		throw new RuntimeException();
	}

	@Override
	public Matrix concatVertical(Matrix...matrices) {
		throw new RuntimeException();
	}


	@Override
	public Matrix vec() {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].vec();
		}
		return new MatrixStack(result);
	}


	@Override
	public Matrix vecInverse(int rows) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].vecInverse(rows);
		}
		return new MatrixStack(result);
	}

	
	@Override
	public Matrix create(Size size) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].create(size);
		}
		return new MatrixStack(result);
	}


	@Override
	public Matrix createIdentity(int n) {
		Matrix[] result = new Matrix[this.matrices.length];
		for (int i = 0; i < this.matrices.length; i++) {
			result[i] = this.matrices[i].createIdentity(n);
		}
		return new MatrixStack(result);
	}

	
	/**
	 * Calculating sum array.
	 * @param arrays arrays.
	 * @return sum array.
	 */
	public static MatrixStack[] sum(MatrixStack[]...arrays) {
		if (arrays == null || arrays.length == 0) return null;
		MatrixStack[] sum = arrays[0];
		for (int i = 1; i < arrays.length; i++) {
			MatrixStack[] array = arrays[i];
			for (int j = 0; j < array.length; j++) {
				sum[j] = (MatrixStack)sum[j].add(array[j]);
			}
		}
		return sum;
	}

	
	/**
	 * Calculating mean array.
	 * @param arrays arrays.
	 * @return mean array.
	 */
	public static MatrixStack[] mean(MatrixStack[]...arrays) {
		MatrixStack[] mean = sum(arrays);
		if (mean == null) return null;
		for (int j = 0; j < mean.length; j++) mean[j] = (MatrixStack)mean[j].divide0(arrays.length);
		return mean;
	}


	/**
	 * Multiplying stacks by value.
	 * @param stacks stacks.
	 * @param value value.
	 * @return divided stacks.
	 */
	public static MatrixStack[] multiply(MatrixStack[] stacks, double value) {
		if (stacks == null || stacks.length == 0) return null;
		MatrixStack[] result = new MatrixStack[stacks.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (MatrixStack)stacks[i].multiply0(value);
		}
		return result;
	}

	
	/**
	 * Dividing stacks by value.
	 * @param stacks stacks.
	 * @param value value.
	 * @return divided stacks.
	 */
	public static MatrixStack[] divide(MatrixStack[] stacks, double value) {
		if (stacks == null || stacks.length == 0) return null;
		MatrixStack[] result = new MatrixStack[stacks.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (MatrixStack)stacks[i].divide0(value);
		}
		return result;
	}

	
	/**
	 * Filling matrix by specified real value.
	 * @param matrix matrix.
	 * @param value value.
	 * @return filled matrix.
	 */
	public static void fill(MatrixStack stack, NeuronValue value) {
		for (int i = 0; i < stack.depth(); i++) {
			Matrix matrix = stack.matrices[i];
			for (int j = 0; j < matrix.rows(); j++) {
				for (int k = 0; k < matrix.columns(); k++) matrix.set(j, k, value);
			}
		}
	}
	
	
	/**
	 * Filling matrix by specified real value.
	 * @param matrix matrix.
	 * @param v value.
	 * @return filled matrix.
	 */
	public static void fill(MatrixStack matrix, double v) {
		NeuronValue value = matrix.get(0, 0).valueOf(v);
		fill(matrix, value);
	}
	
	
	/**
	 * Filling matrix by random value.
	 * @param matrix matrix.
	 * @param rnd randomizer.
	 */
	public static void fill(MatrixStack stack, Random rnd) {
		for (int i = 0; i < stack.depth(); i++) {
			Matrix matrix = stack.matrices[i];
			for (int j = 0; j < matrix.rows(); j++) {
				for (int k = 0; k < matrix.columns(); k++) {
					NeuronValue value = matrix.get(j, k).valueOf(NeuronValue.r(rnd));
					matrix.set(j, k, value);
				}
			}
		}
	}

	
	/**
	 * Accumulating stacks.
	 * @param W stacks.
	 * @param dW stack biases
	 * @param factor factor.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private static void accum(MatrixStack[] W, Matrix[] dW, double factor) {
		if (W == null || dW == null) return;
		if (dW.length != W.length) throw new IllegalArgumentException();
		MatrixStack[] stacks = null;
		if (dW instanceof MatrixStack[])
			stacks = (MatrixStack[])dW;
		else {
			stacks = new MatrixStack[dW.length];
			for (int t = 0; t < stacks.length; t++) {
				stacks[t] = dW[t] instanceof MatrixStack ? (MatrixStack)dW[t] : new MatrixStack(dW[t]);
			}
		}
		for (int t = 0; t < stacks.length; t++) W[t] = (MatrixStack)W[t].add(stacks[t].multiply0(factor));
	}


}
