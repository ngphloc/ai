/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

/**
 * This class represents 3D point.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Point extends java.awt.Point {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Z coordinate.
	 */
	public int z = 0;
	
	
	/**
	 * T coordinate.
	 */
	public int t = 0;

	
	/**
	 * Constructor with X coordinate, Y coordinate, Z coordinate, and T coordinate.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 * @param z Z coordinate.
	 * @param t T coordinate.
	 */
	public Point(int x, int y, int z, int t) {
		super(x, y);
		this.z = z;
		this.t = t;
	}

	
	/**
	 * Constructor with other point.
	 * @param point other point.
	 */
	public Point(Point point) {
		this(point.x, point.y, point.z, point.t);
	}

	
	/**
	 * Constructor with X coordinate, Y coordinate, and Z coordinate.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 * @param z Z coordinate.
	 */
	public Point(int x, int y, int z) {
		this(x, y, z, 0);
	}


	/**
	 * Constructor with X coordinate and Y coordinate.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 */
	public Point(int x, int y) {
		this(x, y, 0);
	}

	
	/**
	 * Constructor with X coordinate.
	 * @param x X coordinate.
	 */
	public Point(int x) {
		this(x, 0);
	}

	
	/**
	 * Default constructor.
	 */
	public Point() {
		this(0);
	}


	/**
	 * Getting row.
	 * @return row.
	 */
	public int row( ) {return y;}
	
	
	/**
	 * Getting column.
	 * @return column.
	 */
	public int column( ) {return x;}

	
	/**
	 * Creating by row and column.
	 * @param row row.
	 * @param column column.
	 * @return point.
	 */
	public static Point createByRowColumn(int row, int column) {
		return new Point(column, row);
	}
	
	
	@Override
	public boolean equals(Object obj) {
		if (obj == null) return false;
		if (!(obj instanceof Point)) return super.equals(obj);
		Point point = (Point)obj;
		return this.x == point.x && this.y == point.y && this.z == point.z && this.t == point.t;
	}

	
}
