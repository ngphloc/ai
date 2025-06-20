/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;

/**
 * This interface represents a filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Filter extends Serializable, Cloneable {

	
	/**
	 * Getting filter width.
	 * @return filter width.
	 */
	int width();

	
	/**
	 * Getting stride width.
	 * @return stride width.
	 */
	int getStrideWidth();
	
	
	/**
	 * Getting filter height.
	 * @return filter height.
	 */
	int height();
	
	
	/**
	 * Getting stride height.
	 * @return stride height.
	 */
	int getStrideHeight();
	
	
	/**
	 * Getting filter depth.
	 * @return filter depth.
	 */
	int depth();

	
	/**
	 * Getting stride depth.
	 * @return stride depth.
	 */
	int getStrideDepth();

	
	/**
	 * Getting filter time.
	 * @return filter time.
	 */
	int time();

	
	/**
	 * Getting stride time.
	 * @return stride time.
	 */
	int getStrideTime();

	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @return whether to move according to stride when filtering.
	 */
	boolean isMoveStride();
	
	
	/**
	 * Checking whether to move according to stride when filtering.
	 * @param moveStride flag to whether to move according to stride when filtering.
	 */
	void setMoveStride(boolean moveStride);
	
	
	/**
	 * Calculating zoom filters.
	 * @param zoom zoom.
	 * @param factory factory.
	 * @param zoomOut zoom flag.
	 * @return zoom filters.
	 */
	static Filter[] calcZoomFilters(SizeZoom zoom, FilterFactory factory, boolean zoomOut) {
		if (zoom.timeZoom <= 1) {
			if (zoom.depthZoom <= 1) {
				if (zoom.widthZoom > 1 || zoom.heightZoom > 1) {
					if (zoomOut)
						return new Filter[] {factory.zoomOut(zoom.widthZoom, zoom.heightZoom)};
					else
						return new Filter[] {factory.zoomIn(zoom.widthZoom, zoom.heightZoom)};
				}
				else
					return null;
			}
			else {
				if (zoomOut)
					return new Filter[] {factory.zoomOut(zoom.widthZoom, zoom.heightZoom, zoom.depthZoom)};
				else
					return new Filter[] {factory.zoomIn(zoom.widthZoom, zoom.heightZoom, zoom.depthZoom)};
			}
		}
		else {
			if (zoomOut)
				return new Filter[] {factory.zoomOut(zoom.widthZoom, zoom.heightZoom, zoom.depthZoom, zoom.timeZoom)};
			else
				return new Filter[] {factory.zoomIn(zoom.widthZoom, zoom.heightZoom, zoom.depthZoom, zoom.timeZoom)};
		}
	
	}


	/**
	 * Calculating the size from filters.
	 * @param outputSize output size. This is output parameter.
	 * @param filters array of filters.
	 * @return size which is the parameter output size. 
	 */
	static Size calcSize(Size outputSize, Filter...filters) {
		if (outputSize == null || filters == null || filters.length == 0) return outputSize;
		for (int i = 0; i < filters.length; i++) {
			Filter filter = filters[i];
			if (filter instanceof DeconvFilter) {
				outputSize.width *= filter.getStrideWidth();
				outputSize.height *= filter.getStrideHeight();
				outputSize.depth *= filter.getStrideDepth();
				outputSize.time *= filter.getStrideTime();
			}
			else {
				outputSize.width /= filter.getStrideWidth();
				outputSize.height /= filter.getStrideHeight();
				outputSize.depth /= filter.getStrideDepth();
				outputSize.time /= filter.getStrideTime();
			}
		}
		
		return outputSize;
	}

	
	/**
	 * Calculating length from filters.
	 * @param initialLength initial length.
	 * @param filters array of filters.
	 * @return length from filters.
	 */
	public static int calcLength(int initialLength, Filter...filters) {
		if (filters == null || filters.length == 0) return initialLength;
		int length = initialLength;
		for (Filter filter : filters) {
			if (filter instanceof DeconvFilter)
				length *= filter.getStrideWidth() * filter.getStrideHeight() * filter.getStrideDepth() * filter.getStrideTime();
			else
				length /= filter.getStrideWidth() * filter.getStrideHeight() * filter.getStrideDepth() * filter.getStrideTime();
		}
		
		return length;
	}


	/**
	 * Calculating length from arrays of filters.
	 * @param initialLength initial length.
	 * @param filterArrays arrays of filters. Filters in the same array have the same size and the same property (convolutional or deconvolutional).
	 * @return length from arrays of filters.
	 */
	static int calcLengthSimply(int initialLength, Filter[][] filterArrays) {
		if (filterArrays == null || filterArrays.length == 0) return initialLength;
		List<Filter> filterList = Util.newList(0);
		for (Filter[] filters : filterArrays) {
			if (filters != null && filters.length > 0) filterList.add(filters[0]);
		}
		
		if (filterList.size() == 0)
			return initialLength;
		else
			return calcLength(initialLength, filterList.toArray(new Filter[] {}));
	}
	
	
	/**
	 * Calculating zooming ratio of filters.
	 * @param filters specified filters.
	 * @return zooming ratio of filters.
	 */
	static SizeZoom zoomRatioOf(Filter...filters) {
		SizeZoom zoom = SizeZoom.zoom(Size.unit());
		if (filters == null || filters.length == 0) return zoom;
		
		for (Filter filter : filters) {
			zoom.widthZoom *= filter.getStrideWidth();
			zoom.heightZoom *= filter.getStrideHeight();
			zoom.depthZoom *= filter.getStrideDepth();
			zoom.timeZoom *= filter.getStrideTime();
		}
			
		return zoom;
	}

	
	/**
	 * Calculating zooming ratio of arrays of filters.
	 * @param filterArrays arrays of filters. Filters in the same array have the same size and the same property (convolutional or deconvolutional).
	 * @return zooming ratio of arrays of filters.
	 */
	static SizeZoom zoomRatioOfSimply(Filter[][] filterArrays) {
		if (filterArrays == null || filterArrays.length == 0) return SizeZoom.zoom(Size.unit());
		List<Filter> filterList = Util.newList(0);
		for (Filter[] filters : filterArrays) {
			if (filters != null && filters.length > 0) filterList.add(filters[0]);
		}
		
		if (filterList.size() == 0)
			return SizeZoom.zoom(Size.unit());
		else
			return zoomRatioOf(filterList.toArray(new Filter[] {}));
	}
	
	
	/**
	 * Calculating zooming ratio of two sizes.
	 * @param thisSize first size.
	 * @param newSize second size.
	 * @return zooming ratio of two sizes.
	 */
	static int zoomRatioOf(Size thisSize, Size newSize) {
		int zoom = Math.max(1, thisSize.width > newSize.width ? thisSize.width/newSize.width : newSize.width/thisSize.width);
		zoom = Math.max(zoom, thisSize.height > newSize.height ? thisSize.height/newSize.height : newSize.height/thisSize.height);
		zoom = Math.max(zoom, thisSize.depth > newSize.depth ? thisSize.depth/newSize.depth : newSize.depth/thisSize.depth);
		zoom = Math.max(zoom, thisSize.time > newSize.time ? thisSize.time/newSize.time : newSize.time/thisSize.time);
		return zoom;
	}


}
