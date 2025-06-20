/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Rectangle;
import java.io.Serializable;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.Raster;

/**
 * This class is an associator of content.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ContentAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Internal content.
	 */
	protected Content content = null;
	
	
	/**
	 * Constructor with content.
	 * @param content specified content.
	 */
	public ContentAssoc(Content content) {
		this.content = content;
	}

	
	/**
	 * Getting filter of current layer.
	 * @return filter of current layer.
	 */
	public Filter getMyFilter() {
		return new ConvLayerSingleAssoc(content).getMyFilter();
	}
	
	
	/**
	 * Forwarding evaluation from current content to next content.
	 * @param nextContent next content.
	 * @param thisFilterRegion filtering region of current content.
	 * @param nextFilterRegion filtering region of next content.
	 * @param nextAffected flag to indicate whether the next content is affected
	 * @return arrays of neurons filtered.
	 */
	public NeuronRaster forward(Content nextContent, Cube thisFilterRegion, Cube nextFilterRegion, boolean nextAffected) {
		if (nextContent == null)
			return ContentImpl.forward(content, (Content)content.getNextLayer(), content.getFilter(), thisFilterRegion, nextFilterRegion, nextAffected);
		else if (nextContent instanceof ConvLayerSingle3D)
			return ContentImpl.forward(content, (ConvLayerSingle3D)nextContent, content.getFilter(), thisFilterRegion, nextFilterRegion, nextAffected);
		else
			return ConvLayer2DAbstract.forward(content, nextContent, content.getFilter(), thisFilterRegion != null ? thisFilterRegion.toRectangle() : (Rectangle)null, nextFilterRegion != null ? nextFilterRegion.toRectangle() : (Rectangle)null, nextAffected);
	}
	
	
	/**
	 * Getting raster of this content.
	 * @param isNorm flag to indicate whether raster value is normalized in [0, 1].
	 * @param defaultAlpha default alpha value.
	 * @return raster of this content.
	 */
	public Raster getRaster(boolean isNorm, int defaultAlpha) {
		return content != null ? content.createRaster(content.getData(), isNorm, defaultAlpha) : null;
	}

	
}
