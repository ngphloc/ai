/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * This is utility class to provide static utility methods.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Util {
	
	
	/**
	 * Working directory.
	 */
	public static String  WORKING_DIRECTORY = "working";
	
	
	/**
	 * Decimal format.
	 */
	public static String DECIMAL_FORMAT = "%.12f";;
	
	
	/**
	 * Default date format.
	 */
	public static String  DATE_FORMAT = "yyyy-MM-dd HH-mm-ss";
	
	
	/**
	 * No name field.
	 */
	public final static String NONAME = "noname";
	
	
	/**
	 * Static code.
	 */
	static {
		try {
			WORKING_DIRECTORY = net.ea.ann.adapter.Util.WORKING_DIRECTORY;
		} catch (Throwable e) {}

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
	 * Creating a new set with initial capacity.
	 * @param <T> type of elements in set.
	 * @param initialCapacity initial capacity of this list.
	 * @return new set.
	 */
	public static <T> Set<T> newSet(int initialCapacity) {
		try {
		    return net.ea.ann.adapter.Util.newSet(initialCapacity);
		}
		catch (Throwable e) {}
		
	    return new HashSet<T>(initialCapacity);
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
		catch (Throwable ex) {e.printStackTrace();}
	}
	
	
	/**
	 * Clone object by serialization
	 * @param object specified object.
	 * @return cloned object.
	 */
	public static Object cloneBySerialize(Object object) {
		if (object == null) return null;
		try {
			ByteArrayOutputStream os = new ByteArrayOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(os);
			oos.writeObject(object);
			oos.flush();

			ByteArrayInputStream is = new ByteArrayInputStream(os.toByteArray());
			ObjectInputStream ois = new ObjectInputStream(is);
			Object cloned = ois.readObject();
			
			oos.close();
			ois.close();
			return cloned;
		}
		catch (Throwable e) {trace(e);} 
		
		return null;
	}

	
	/**
	 * Writing (serializing) object to output stream.
	 * @param object object will be serialized.
	 * @param os output stream.
	 * @return true if writing (serializing) is successful. 
	 */
	public static boolean serialize(Object object, OutputStream os) {
		try {
			if (object == null) return false;
			ObjectOutputStream output = new ObjectOutputStream(os);
			output.writeObject(object);
			output.flush();
			return true;
		}
		catch (Throwable e) {trace(e);}
		
		return false;
	}

	
	/**
	 * Reading (deserializing) object from input stream.
	 * @param is input stream.
	 * @return deserialized object. Returning null if deserializing is not successful.
	 */
	public static Object deserialize(InputStream is) {
		try {
			ObjectInputStream input = new ObjectInputStream(is);
			Object object = input.readObject();
			return object;
		}
		catch (Throwable e) {trace(e);}
		
		return null;
	}

	
	/**
	 * Converting a specified array of objects (any type) into a string in which each object is converted as a word in such string.
	 * Words in such returned string are connected by the character specified by the parameter {@code sep}. 
	 * This is template static method and so the type of object is specified by the template &lt;{@code T}&gt;.
	 * @param <T> type of each object in the specified array.
	 * @param array Specified array of objects.
	 * @param sep The character that is used to connect words in the returned string. As usual, it is a comma &quot;,&quot;.
	 * @return Text form (string) of the specified array of objects, in which each object is converted as a word in such text form.
	 */
	public static <T extends Object> String toText(T[] array, String sep) {
		StringBuffer buffer = new StringBuffer();
		
		for (int i = 0; i < array.length; i++) {
			if ( i > 0)
				buffer.append(sep + " ");

			T value = array[i];
			if (value instanceof TextParsable)
				buffer.append(((TextParsable)value).toText());
			else
				buffer.append(value);
		}
		
		return buffer.toString();
		
	}

	
	/**
	 * Randomizing Gaussian number.
	 * @param rnd specified randomizer.
	 * @return Gaussian number.
	 */
	public static double randomGaussian(Random rnd) {
		double r = rnd.nextGaussian();
//		//Three sigma rule.
//		r = Math.min(r, 3.0);
//		r = Math.max(r, -3.0);
		
//		//Squashing in [-1, 1] to reserve standard Gaussian distribution mean 0 and variance 1.
//		r = 2 / (1.0 + Math.exp(-r)) - 1;
		
		return r;
	}
	
	
//	/**
//	 * Multiplying two matrices.
//	 * @param A first matrix.
//	 * @param B second matrix.
//	 * @return multiplied matrix.
//	 */
//	public static double[][] multiply(double[][] A, double[][] B) {
//		int m = A.length;
//		double[][] C = new double[m][];
//		
//		for (int i = 0; i < m; i++) {
//			int n = A[i].length;
//			C[i] = new double[n];
//			
//			for (int j = 0; j < n; j++) {
//				C[i][j] = 0;
//				for (int k = 0; k < n; k++) {
//					C[i][j] += A[i][k] * B[k][j];
//				}
//			}
//		}
//		
//		return C;
//	}
	
	
	/**
	 * Generate cofactor of matrix, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @param removedRow removed row.
	 * @param removedColumn removed column.
	 * @return cofactor of matrix.
	 */
	private static double[][] genCofactor(double A[][], int removedRow, int removedColumn) {
		int n = A.length;
		double[][] co = new double[n-1][];
		for (int k = 0; k < n-1; k++) co[k] = new double[n-1];

		int k = 0;
		for (int i = 0; i < n; i++) {
			if (i == removedRow) continue;
			
			int l = 0;
			for (int j = 0; j < n; j++){
				if(j != removedColumn) {
					co[k][l] = A[i][j];
					l++;
				}
			}
			k++;
		}
		
		return co;
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
	public static double detNotOptimalYet(double[][] A) {
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
	public static double[][] inverseNotOptimalYet(double[][] A) {
		if (A == null) return null;
		int n = A.length;
		if (n == 0) return null;
		
		if (A.length == 1) return A[0][0] != 0 ? new double[][] {{1.0/A[0][0]}} : null;
		
		double[][] B = new double[n][];
		for (int i = 0; i < n; i++) B[i] = new double[n];
		
		double det = detNotOptimalYet(A);
		if (Double.isNaN(det) || det == 0) return null;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double[][] co = genCofactor(A, i, j);
				B[j][i] = Math.pow(-1.0, i+j) * detNotOptimalYet(co) / det;
			}
		}
		
		return B;
	}
	
	
	/**
	 * Checking if the given matrix is invertible.
	 * @param A given matrix.
	 * @return if the given matrix is invertible.
	 */
	public static boolean isInvertible(double[][] A) {
		try {
		    return net.ea.ann.adapter.Util.isInvertible(A);
		}
		catch (Throwable e) {
			if (e instanceof ClassNotFoundException) {
				double det = detNotOptimalYet(A);
				return (!Double.isNaN(det)) && (det != 0);
			}
			System.out.println("Checking if matrix is invertible causes error: " + e.getMessage());
		}

		return false;
	}

	
	/**
	 * Checking if the given matrix is invertible.
	 * @param A given matrix.
	 * @return if the given matrix is invertible.
	 */
	public static double det(double[][] A) {
		try {
		    return net.ea.ann.adapter.Util.det(A);
		}
		catch (Throwable e) {
			if (e instanceof ClassNotFoundException) return detNotOptimalYet(A);
			System.out.println("Calculating matrix determinant causes error: " + e.getMessage());
		}
		
		return Double.NaN;
	}

	
	/**
	 * Calculating inverse of the given matrix.
	 * @param A given matrix.
	 * @return inverse of the given matrix. Return null if the matrix is not invertible.
	 */
	public static double[][] inverse(double[][] A) {
		try {
		    return net.ea.ann.adapter.Util.inverse(A);
		}
		catch (Throwable e) {
			if (e instanceof ClassNotFoundException) return inverseNotOptimalYet(A);
			System.out.println("Calculating matrix inverse causes error: " + e.getMessage());
		}
		
		return null;
	}
	
	
	/**
	 * Calculating pseudo square root of the given matrix.
	 * @param A given matrix.
	 * @return pseudo square root of the given matrix.
	 */
	public static double[][] sqrtNotOptimalYet(double[][] A) {
		if (A == null || A.length == 0 || A[0].length == 0 || A.length != A[0].length) return null;
		if (A.length == 1) return A[0][0] >= 0 ? new double[][] {{Math.sqrt(A[0][0])}} : null;
		
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) if (A[i][j] < 0) return null;
		}
		
		double[][] S = new double[A.length][A.length];
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) S[i][j] = Math.sqrt(A[i][j]);
		}
		return S;
	}
	
	
	/**
	 * Calculating square root of the given matrix.
	 * @param A given matrix.
	 * @return square root of the given matrix.
	 */
	public static double[][] sqrt(double[][] A) {
		if (A == null || A.length == 0) return null;
		if (A.length == 1) return A[0][0] >= 0 ? new double[][] {{Math.sqrt(A[0][0])}} : null;
		
		boolean specialTechnique = false;
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) {
				specialTechnique = ((i != j) && (A[i][j] != 0)) || ((i == j) && (A[i][j] < 0));
				if (specialTechnique) break;
			}
		}
		
		if (specialTechnique) {
			try {
			    return net.ea.ann.adapter.Util.sqrt(A);
			}
			catch (Throwable e) {
				if (e instanceof ClassNotFoundException)
					return sqrtNotOptimalYet(A);
				System.out.println("Calculating matrix square root causes error: " + e.getMessage());
			}
			return null;
		}
		else {
			double[][] S = new double[A.length][A.length];
			for (int i = 0; i < A.length; i++) {
				for (int j = 0; j < A.length; j++) S[i][j] = i == j ? Math.sqrt(A[i][j]) : 0;
			}
			return S;
		}
	}


}
