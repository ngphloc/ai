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
	
	
}
