/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

/**
 * This interface represents raster property.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RasterPropertyImpl implements RasterProperty {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Label.
	 */
	protected Label label = new Label();
	
	
	/**
	 * Default constructor.
	 */
	public RasterPropertyImpl() {

	}


	/**
	 * Constructor with other property.
	 * @param property other property.
	 */
	public RasterPropertyImpl(RasterProperty property) {
		this.label = property.getLabel();
	}
	
	
	@Override
	public Label getLabel() {
		return label;
	}


	@Override
	public void setLabel(Label label) {
		this.label = label != null ? label : new Label();
	}


	@Override
	public int getLabelId() {
		return label.labelId;
	}


	@Override
	public void setLabelId(int labelId) {
		label.labelId = labelId;
	}


	@Override
	public String getLabelName() {
		return label.labelName;
	}


	@Override
	public void setLabelName(String labelName) {
		label.labelName = labelName;
	}


	@Override
	public void clearLabel() {
		label = new Label();
	}


	@Override
	public RasterProperty shallowDuplicate() {
		return new RasterPropertyImpl(this);
	}


}
