/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This is utility class to provide static utility methods.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Util {
	
	
	/**
	 * Decimal format.
	 */
	public static String DECIMAL_FORMAT = "%.12f";;
	
	
	/**
	 * Default date format.
	 */
	public static String  DATE_FORMAT = "yyyy-MM-dd HH-mm-ss";
	
	
	/**
	 * Static code.
	 */
	static {
		try {
			DATE_FORMAT = net.ea.ann.adapter.Util.DATE_FORMAT;
		} catch (Throwable e) {}
	}

	
	/**
	 * Creating a new array.
	 * @param <T> element type.
	 * @param tClass element type.
	 * @param length array length.
	 * @return new array
	 */
	public static <T> T[] newArray(Class<T> tClass, int length) {
		try {
		    return net.ea.ann.adapter.Util.newArray(tClass, length);
		}
		catch (Throwable e) {}
		
		@SuppressWarnings("unchecked")
		T[] array = (T[]) Array.newInstance(tClass, length);
		return array;
	}

	
	/**
	 * Creating a new list with initial capacity.
	 * @param <T> type of elements in list.
	 * @param initialCapacity initial capacity of this list.
	 * @return new list with initial capacity.
	 */
	public static <T> List<T> newList(int initialCapacity) {
		try {
		    return net.ea.ann.adapter.Util.newList(initialCapacity);
		}
		catch (Throwable e) {}
		
	    return new ArrayList<T>(initialCapacity);
	}
	
	
	/**
	 * Creating a new map.
	 * @param <K> type of key.
	 * @param <V> type of value.
	 * @param initialCapacity initial capacity of this list.
	 * @return new map.
	 */
	public static <K, V> Map<K, V> newMap(int initialCapacity) {
		try {
		    return net.ea.ann.adapter.Util.newMap(initialCapacity);
		}
		catch (Throwable e) {}
		
	    return new HashMap<K, V>(initialCapacity);
	}

	
	/**
	 * Converting the specified number into a string.
	 * @param number specified number.
	 * @return text format of number of the specified number.
	 */
	public static String format(double number) {
		try {
		    return net.ea.ann.adapter.Util.format(number);
		}
		catch (Throwable e) {}

		return String.format(DECIMAL_FORMAT, number);
	}

	
	/**
	 * Tracing error.
	 * @param e throwable error.
	 */
	public static void trace(Throwable e) {
		try {
			net.ea.ann.adapter.Util.trace(e);
		}
		catch (Throwable ex) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Multiplying two matrix.
	 * @param A first matrix.
	 * @param B second matrix.
	 * @return multiplied matrix.
	 */
	public static double[][] multiply(double[][] A, double[][] B) {
		int m = A.length;
		double[][] C = new double[m][];
		
		for (int i = 0; i < m; i++) {
			int n = A[i].length;
			C[i] = new double[n];
			
			for (int j = 0; j < n; j++) {
				C[i][j] = 0;
				for (int k = 0; k < n; k++) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
		
		return C;
	}
	
	
	/**
	 * Generate cofactor of matrix, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @param removedRow removed row.
	 * @param removedColumn removed column.
	 * @return cofactor of matrix.
	 */
	private static double[][] genCofactor(double A[][], int removedRow, int removedColumn) {
		int n = A.length;
		double[][] sub = new double[n-1][];
		for (int k = 0; k < n-1; k++) sub[k] = new double[n-1];

		int k = 0;
		for (int i = 0; i < n; i++) {
			if (i == removedRow) continue;
			
			int l = 0;
			for (int j = 0; j < n; j++){
				if(j != removedColumn) {
					sub[k][l] = A[i][j];
					l++;
				}
			}
			k++;
		}
		
		return sub;
	}
	
	
	/**
	 * Calculate determinant recursively, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @return determinant of specific matrix.
	 */
	private static double det0(double[][] A) {
		int n = A.length;
		if (n == 1) return A[0][0];
		if (n == 2) return A[0][0]*A[1][1] - A[1][0]*A[0][1];
		
		double det = 0;
		for (int j = 0; j < n; j++){
			double[][] co = genCofactor(A, 0, j);
			det += Math.pow(-1.0, j) * A[0][j] * det0(co);
		}
		
		return det;
	}
	
	
	/**
	 * Calculate determinant recursively, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @return determinant of specific matrix.
	 */
	public static double det(double[][] A) {
		if (A == null) return Double.NaN;
		int n = A.length;
		if (n == 0) return Double.NaN;
		
		return det0(A);
	}


	/**
	 * Calculating the inverse of specific matrix.
	 * @param A specific matrix.
	 * @return the inverse of specific matrix.
	 */
	public static double[][] inverse(double[][] A) {
		if (A == null) return null;
		int n = A.length;
		if (n == 0) return null;
		
		if (A.length == 1) return new double[][] {{1.0/A[0][0]}};
		
		double[][] B = new double[n][];
		for (int i = 0; i < n; i++) B[i] = new double[n];
		
		double det = det(A);
		if (det == 0) return null;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double[][] co = genCofactor(A, i, j);
				B[j][i] = Math.pow(-1.0, i+j) * det(co) / det;
			}
		}
		
		return B;
	}
	
	
}
