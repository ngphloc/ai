/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.awt.Rectangle;
import java.io.Serializable;

/**
 * This class represents 4D cube.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Cube implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * X coordination.
	 */
	public int x = 0;
	
	
	/**
	 * Y coordination.
	 */
	public int y = 0;

	
	/**
	 * Z coordination.
	 */
	public int z = 0;

	
	/**
	 * T coordination.
	 */
	public int t = 0;

	
	/**
	 * Width.
	 */
	public int width = 0;
	

	/**
	 * Height.
	 */
	public int height = 0;

	
	/**
	 * Depth.
	 */
	public int depth = 0;

	
	/**
	 * Time.
	 */
	public int time = 0;

	
	/**
	 * Constructor with x, y, z, t, width, height, depth, and time.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @param z Z coordination.
	 * @param t T coordination.
	 * @param width specified width.
	 * @param height specified height.
	 * @param depth specified depth.
	 * @param time specified time.
	 */
	public Cube(int x, int y, int z, int t, int width, int height, int depth, int time) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.t = t;
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.time = time;
	}
	
	
	/**
	 * Constructor with point and size.
	 * @param point specified point.
	 * @param size specified size.
	 */
	public Cube(Point point, Size size) {
		this(point.x, point.y, point.z, point.t, size.width, size.height, size.depth, size.time);
	}
	
	
	/**
	 * Constructor with x, y, z, width, height, and depth.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @param z Z coordination.
	 * @param width specified width.
	 * @param height specified height.
	 * @param depth specified depth.
	 */
	public Cube(int x, int y, int z, int width, int height, int depth) {
		this(x, y, z, 0, width, height, depth, 0);
	}
	
	
	/**
	 * Constructor with x, y, width, and height.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @param width specified width.
	 * @param height specified height.
	 */
	public Cube(int x, int y, int width, int height) {
		this(x, y, 0, width, height, 0);
	}

	
	/**
	 * Constructor with other cube.
	 * @param cube other cube.
	 */
	public Cube(Cube cube) {
		this.x = cube.x;
		this.y = cube.y;
		this.z = cube.z;
		this.t = cube.t;
		this.width = cube.width;
		this.height = cube.height;
		this.depth = cube.depth;
		this.time = cube.time;
	}
	
	
	/**
	 * Default constructor.
	 */
	public Cube() {

	}

	
	/**
	 * Checking whether this cube contains x.
	 * @param X X coordination.
	 * @return whether this cube contain x.
	 */
    public boolean contains(int X) {
        return (this.x <= X) && (X < this.x + this.width);
    }

    
    /**
	 * Checking whether this cube contains x and y.
	 * @param X X coordination.
	 * @param Y Y coordination.
	 * @return whether this cube contain x and y.
	 */
    public boolean contains(int X, int Y) {
        return (this.x <= X) && (X < this.x + this.width) && (this.y <= Y) && (Y < this.y + this.height);
    }


	/**
	 * Checking whether this cube contains x, y, and z.
	 * @param X X coordination.
	 * @param Y Y coordination.
	 * @param Z Z coordination.
	 * @return whether this cube contain x, y, and z.
	 */
    public boolean contains(int X, int Y, int Z) {
        return (contains(X, Y)) && (this.z <= Z) && (Z < this.z + this.depth);
    }


	/**
	 * Checking whether this cube contains x, y, z, and t.
	 * @param X X coordination.
	 * @param Y Y coordination.
	 * @param Z Z coordination.
	 * @param T T coordination.
	 * @return whether this cube contain x, y, z, and t.
	 */
    public boolean contains(int X, int Y, int Z, int T) {
        return (contains(X, Y, Z)) && (this.t <= T) && (T < this.t + this.time);
    }

    
    /**
     * Checking whether this cube contains the specified point.
     * @param point specified point.
     * @return whether this cube contains the specified point.
     */
    public boolean contains(Point point) {
    	return contains(point.x, point.y, point.z, point.t);
    }
    
    
    /**
     * Converting this 3D cube to 2D rectangle.
     * @return 2D rectangle.
     */
    public Rectangle toRectangle() {
    	return new Rectangle(x, y, width, height);
    }
    
    
	/**
	 * Multiplying this dimension with factor.
	 * @param factor specified factor.
	 * @return new dimension multiplied with factor.
	 */
	public Cube multiply(int factor) {
		return new Cube(x*factor, y*factor, z*factor, t*factor, width*factor, height*factor, depth*factor, time*factor);
	}


}
