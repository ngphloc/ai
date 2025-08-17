/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.util.List;

import net.ea.ann.core.Util;

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
	 * List of labels.
	 */
	protected List<Label> labels = Util.newList(0);
	
	
	/**
	 * Default constructor.
	 */
	public RasterPropertyImpl() {
		labels.add(new Label());
	}


	/**
	 * Constructor with other property.
	 * @param property other property.
	 */
	public RasterPropertyImpl(RasterProperty property) {
		if (property == null || property.getLabelCount() == 0) throw new IllegalArgumentException("Illegal parameter");
		int n = property.getLabelCount();
		for (int i = 0; i < n; i++) this.labels.add(property.getLabel(i));
	}
	
	
	/**
	 * Constructor with array of labels.
	 * @param labels array of labels.
	 */
	public RasterPropertyImpl(Label...labels) {
		if (labels == null || labels.length == 0) throw new IllegalArgumentException("Illegal parameter");
		for (int i = 0; i < labels.length; i++) this.labels.add(labels[i]);
	}
	
	
	@Override
	public Label getLabel() {
		return labels.get(0);
	}


	@Override
	public Label getLabel(int index) {
		return labels.get(index);
	}


	@Override
	public int getLabelCount() {
		return labels.size();
	}


	@Override
	public void addLabel(Label label) {
		if (label != null) this.labels.add(label);
	}


	@Override
	public void setLabel(Label label) {
		if (label == null) this.labels.set(0, label);
	}


	@Override
	public void setLabels(Label...labels) {
		if (labels == null || labels.length == 0) return;
		this.labels.clear();
		for (int i = 0; i < labels.length; i++) this.labels.add(labels[i]);
	}


	@Override
	public void setLabel(int index, Label label) {
		if (label == null) this.labels.set(index, label);
	}


	@Override
	public int getLabelId() {
		return getLabel().labelId;
	}


	@Override
	public int getLabelId(int index) {
		return getLabel(index).labelId;
	}


	@Override
	public void setLabelId(int labelId) {
		getLabel().labelId = labelId;
	}


	@Override
	public void setLabelId(int index, int labelId) {
		getLabel(index).labelId = labelId;
	}


	@Override
	public String getLabelName() {
		return getLabel().labelName;
	}


	@Override
	public String getLabelName(int index) {
		return getLabel(index).labelName;
	}


	@Override
	public void setLabelName(String labelName) {
		getLabel().labelName = labelName;
	}


	@Override
	public void setLabelName(int index, String labelName) {
		getLabel(index).labelName = labelName;
	}


	@Override
	public void clearLabel() {
		this.labels.clear();
		this.labels.add(new Label());
	}


	@Override
	public RasterProperty shallowDuplicate() {
		return new RasterPropertyImpl(this);
	}


}
