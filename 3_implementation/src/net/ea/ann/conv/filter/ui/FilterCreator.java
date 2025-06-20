/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.filter.ui;

import java.awt.Window;

import javax.swing.JDialog;

import net.ea.ann.conv.filter.Filter;

/**
 * This class represents a dialog to create filters.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class FilterCreator extends JDialog {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Created filters
	 */
	protected Filter[] filters = null;
	
	
	/**
	 * Constructor with parent window.
	 * @param parent parent window.
	 */
	public FilterCreator(Window parent) {
		super(parent, "Creating filters", ModalityType.APPLICATION_MODAL);
	}

	
	/**
	 * Getting created filters.
	 * @return created filters.
	 */
	public Filter[] getFilters() {
		return filters;
	}
	
	
}
