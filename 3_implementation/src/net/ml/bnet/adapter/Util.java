/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ml.bnet.adapter;

import java.util.List;
import java.util.Map;

import net.hudup.core.Constants;
import net.hudup.core.data.Attribute.Type;
import net.hudup.core.data.Profile;

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
	 * Tracing error.
	 * @param e throwable error.
	 */
	public static void trace(Throwable e) {
		net.hudup.core.logistic.LogUtil.trace(e);
	}


	/**
	 * Converting Hudup profile into Bayesian network profile.
	 * @param newAttRef Bayesian network attributes.
	 * @param profile Hudup profile.
	 * @return Bayesian network profile.
	 */
	public static net.ml.bnet.Profile toBnetProfile(net.ml.bnet.AttributeList newAttRef, Profile profile) {
		if (newAttRef == null || profile == null) return null;
		
		net.ml.bnet.Profile newProfile = new net.ml.bnet.Profile(newAttRef);
		int n = Math.min(newProfile.getAttCount(), profile.getAttCount());
		for (int i = 0; i < n; i++) {
			newProfile.setValue(i, profile.getValue(i));
		}
		
		return newProfile;
	}
	
	
	/**
	 * Converting Hudup profile into Bayesian network profile.
	 * @param profile Hudup profile.
	 * @return Bayesian network profile.
	 */
	public static net.ml.bnet.Profile toBnetProfile(Profile profile) {
		net.ml.bnet.AttributeList newAttRef = extractBnetAttributes(profile);
		return toBnetProfile(newAttRef, profile);
	}

		
	/**
	 * Extracting Bayesian network attributes from Hudup profile.
	 * @param profile Hudup profile.
	 * @return list of Bayesian network attributes.
	 */
	public static net.ml.bnet.AttributeList extractBnetAttributes(Profile profile) {
		if (profile == null) return new net.ml.bnet.AttributeList();
		
		net.ml.bnet.AttributeList newAttRef = new net.ml.bnet.AttributeList();
		for (int i = 0; i < profile.getAttCount(); i++) {
			Type type = profile.getAtt(i).getType();
			String name = profile.getAtt(i).getName();
			net.ml.bnet.Attribute.Type newType = net.ml.bnet.Attribute.Type.real;
			switch (type) {
			case bit:
				newType = net.ml.bnet.Attribute.Type.bit;
				break;
			case nominal:
				newType = net.ml.bnet.Attribute.Type.integer;
				break;
			case integer:
				newType = net.ml.bnet.Attribute.Type.integer;
				break;
			case real:
				newType = net.ml.bnet.Attribute.Type.real;
				break;
			case string:
				newType = net.ml.bnet.Attribute.Type.string;
				break;
			case date:
				newType = net.ml.bnet.Attribute.Type.date;
				break;
			case time:
				newType = net.ml.bnet.Attribute.Type.time;
				break;
			case object:
				newType = net.ml.bnet.Attribute.Type.object;
				break;
			}
			
			newAttRef.add(new net.ml.bnet.Attribute(name, newType));
		}
		
		return newAttRef;
	}


}
