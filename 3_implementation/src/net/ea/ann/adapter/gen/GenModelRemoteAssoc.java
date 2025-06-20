/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import java.io.Serializable;

import net.ea.ann.core.Util;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.hudup.core.data.DataConfig;

/**
 * This class is an associator of remote generative model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenModelRemoteAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal generative model.
	 */
	protected GenModelRemote gm = null;
	
	
	/**
	 * Constructor with remote generative model.
	 * @param gm remote generative model.
	 */
	public GenModelRemoteAssoc(GenModelRemote gm) {
		this.gm = gm;
	}
	

	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	public boolean isNorm() {
		DataConfig config = null;
		try {
			config = gm.queryConfig();
		} catch (Throwable e) {Util.trace(e);}
		if (config != null && config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}

	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	public int getDefaultAlpha() {
		DataConfig config = null;
		try {
			config = gm.queryConfig();
		} catch (Throwable e) {Util.trace(e);}
		if (config != null && config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}


}
