/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

/**
 * This class represents multidimensional size and zoom.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SizeZoom extends Size {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Width zoom ratio.
	 */
	public int widthZoom = 0;
	
	
	/**
	 * Height zoom ratio.
	 */
	public int heightZoom = 0;
	
	
	/**
	 * Depth zoom ratio.
	 */
	public int depthZoom = 0;
	
	
	/**
	 * Time zoom ratio.
	 */
	public int timeZoom = 0;


	/**
	 * Constructor with specified width, height, depth, time, width zoom ratio, height zoom ratio, depth zoom ratio, and time zoom ratio.
	 * @param width specified width.
	 * @param height specified height.
	 * @param depth specified depth.
	 * @param time specified time.
	 * @param widthZoom specified width zoom ratio.
	 * @param heightZoom specified height zoom ratio.
	 * @param depthZoom specified depth zoom ratio.
	 * @param timeZoom specified time zoom ratio.
	 */
	public SizeZoom(int width, int height, int depth, int time, int widthZoom, int heightZoom, int depthZoom, int timeZoom) {
		super(width, height, depth, time);
		this.widthZoom = widthZoom;
		this.heightZoom = heightZoom;
		this.depthZoom = depthZoom;
		this.timeZoom = timeZoom;
	}


	/**
	 * Constructor with specified width, height, depth, width zoom ratio, height zoom ratio, and depth zoom ratio.
	 * @param width specified width.
	 * @param height specified height.
	 * @param depth specified depth.
	 * @param widthZoom specified width zoom ratio.
	 * @param heightZoom specified height zoom ratio.
	 * @param depthZoom specified depth zoom ratio.
	 */
	public SizeZoom(int width, int height, int depth, int widthZoom, int heightZoom, int depthZoom) {
		this(width, height, depth, 0, widthZoom, heightZoom, depthZoom, 0);
	}

	
	/**
	 * Constructor with specified width, height, width zoom ratio, and height zoom ratio.
	 * @param width specified width.
	 * @param height specified height.
	 * @param widthZoom specified width zoom ratio.
	 * @param heightZoom specified height zoom ratio.
	 */
	public SizeZoom(int width, int height, int widthZoom, int heightZoom) {
		this(width, height, 0, widthZoom, heightZoom, 0);
	}

	
	/**
	 * Default constructor.
	 */
	public SizeZoom() {
		this(0, 0, 0, 0);
	}

	
	/**
	 * Creating zoom.
	 * @param widthZoom specified width zoom ratio.
	 * @param heightZoom specified height zoom ratio.
	 * @param depthZoom specified depth zoom ratio.
	 * @param timeZoom specified time zoom ratio.
	 * @return zoom structure.
	 */
	public static SizeZoom zoom(int widthZoom, int heightZoom, int depthZoom, int timeZoom) {
		return new SizeZoom(0, 0, 0, 0, widthZoom, heightZoom, depthZoom, timeZoom);
	}
	
	
	/**
	 * Creating zoom from size.
	 * @param size specified size.
	 * @return zoom structure.
	 */
	public static SizeZoom zoom(Size size) {
		return size != null ? zoom(size.width, size.height, size.depth, size.time) : new SizeZoom();
	}


}
