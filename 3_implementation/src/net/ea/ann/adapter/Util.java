/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter;

import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.core.NetworkConfig;
import net.hudup.core.Constants;
import net.hudup.core.data.Attribute.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.MathUtil;

/**
 * This is utility class to provide static utility methods. It is also adapter to other libraries.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Util {

	
	/**
	 * Default date format.
	 */
	public static String DATE_FORMAT = "yyyy-MM-dd HH-mm-ss";

	
	/**
	 * Static code.
	 */
	static {
		try {
			DATE_FORMAT = Constants.DATE_FORMAT;
		}
		catch (Throwable e) {}
	}
	
	
	/**
	 * Creating a new array.
	 * @param <T> element type.
	 * @param tClass element type.
	 * @param length array length.
	 * @return new array
	 */
	public static <T> T[] newArray(Class<T> tClass, int length) {
		return net.hudup.core.Util.newArray(tClass, length);
	}

	
	/**
	 * Creating a new list with initial capacity.
	 * @param <T> type of elements in list.
	 * @param initialCapacity initial capacity of this list.
	 * @return new list with initial capacity.
	 */
	public static <T> List<T> newList(int initialCapacity) {
	    return net.hudup.core.Util.newList(initialCapacity);
	}
	
	
	/**
	 * Creating a new map.
	 * @param <K> type of key.
	 * @param <V> type of value.
	 * @param initialCapacity initial capacity of this list.
	 * @return new map.
	 */
	public static <K, V> Map<K, V> newMap(int initialCapacity) {
	    return net.hudup.core.Util.newMap(initialCapacity);
	}

	
	/**
	 * Converting the specified number into a string. The number of decimal digits is specified by {@link Constants#DECIMAL_PRECISION}.
	 * @param number specified number.
	 * @return text format of number of the specified number.
	 */
	public static String format(double number) {
		return MathUtil.format(number);
	}

	
	/**
	 * Tracing error.
	 * @param e throwable error.
	 */
	public static void trace(Throwable e) {
		net.hudup.core.logistic.LogUtil.trace(e);
	}


	/**
	 * Converting Hudup profile into ANN profile.
	 * @param newAttRef ANN attributes.
	 * @param profile Hudup profile.
	 * @return ANN profile.
	 */
	public static net.ea.ann.core.Profile toANNProfile(net.ea.ann.core.AttributeList newAttRef, Profile profile) {
		if (newAttRef == null || profile == null) return null;
		
		net.ea.ann.core.Profile newProfile = new net.ea.ann.core.Profile(newAttRef);
		int n = Math.min(newProfile.getAttCount(), profile.getAttCount());
		for (int i = 0; i < n; i++) {
			newProfile.setValue(i, profile.getValue(i));
		}
		
		return newProfile;
	}
	
	
	/**
	 * Converting Hudup profile into ANN profile.
	 * @param profile Hudup profile.
	 * @return ANN profile.
	 */
	public static net.ea.ann.core.Profile toANNProfile(Profile profile) {
		net.ea.ann.core.AttributeList newAttRef = extractANNAttributes(profile);
		return toANNProfile(newAttRef, profile);
	}

		
	/**
	 * Extracting ANN attributes from Hudup profile.
	 * @param profile Hudup profile.
	 * @return list of ANN attributes.
	 */
	public static net.ea.ann.core.AttributeList extractANNAttributes(Profile profile) {
		if (profile == null) return new net.ea.ann.core.AttributeList();
		
		net.ea.ann.core.AttributeList newAttRef = new net.ea.ann.core.AttributeList();
		for (int i = 0; i < profile.getAttCount(); i++) {
			Type type = profile.getAtt(i).getType();
			String name = profile.getAtt(i).getName();
			net.ea.ann.core.Attribute.Type newType = net.ea.ann.core.Attribute.Type.real;
			switch (type) {
			case bit:
				newType = net.ea.ann.core.Attribute.Type.bit;
				break;
			case nominal:
				newType = net.ea.ann.core.Attribute.Type.integer;
				break;
			case integer:
				newType = net.ea.ann.core.Attribute.Type.integer;
				break;
			case real:
				newType = net.ea.ann.core.Attribute.Type.real;
				break;
			case string:
				newType = net.ea.ann.core.Attribute.Type.string;
				break;
			case date:
				newType = net.ea.ann.core.Attribute.Type.date;
				break;
			case time:
				newType = net.ea.ann.core.Attribute.Type.time;
				break;
			case object:
				newType = net.ea.ann.core.Attribute.Type.object;
				break;
			}
			
			newAttRef.add(new net.ea.ann.core.Attribute(name, newType));
		}
		
		return newAttRef;
	}


	/**
	 * Convert Hudup configuration to ANN configuration.
	 * @param config Hudup configuration.
	 * @return ANN configuration.
	 */
	public static NetworkConfig transferToANNConfig(DataConfig config) {
		if (config == null) return new NetworkConfig();
		
		NetworkConfig annConfig = new NetworkConfig();
		Set<String> keys = config.keySet();
		for (String key : keys) {
			if (annConfig.containsKey(key)) annConfig.put(key, config.get(key));
		}
		
		return annConfig;
	}

	
	/**
	 * Convert ANN configuration to Hudup configuration.
	 * @param annConfig ANN configuration.
	 * @return Hudup configuration.
	 */
	public static DataConfig toConfig(NetworkConfig annConfig) {
		if (annConfig == null) return new DataConfig();
		
		DataConfig config = new DataConfig();
		Set<String> keys = annConfig.keySet();
		for (String key : keys) {
			config.put(key, annConfig.get(key));
		}
		
		return config;
	}


}
