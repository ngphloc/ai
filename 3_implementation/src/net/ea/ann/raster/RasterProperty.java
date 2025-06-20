/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import net.ea.ann.core.Util;

/**
 * This interface represents raster property.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface RasterProperty extends Serializable, Cloneable {

	
	/**
	 * No label.
	 */
	public final static int NOLABEL = -1;
	
	
	/**
	 * No label name.
	 */
	public final static String NOLABELNAME = Util.NONAME;

	
	/**
	 * This class represents label.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class Label implements Serializable, Cloneable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Label identifier.
		 */
		public int labelId = NOLABEL;
		
		/**
		 * Label name.
		 */
		public String labelName = NOLABELNAME;
		
		/**
		 * Constructor with label identifier and label name.
		 * @param labelId label identifier.
		 * @param labelName label name.
		 */
		public Label(int labelId, String labelName) {
			this.labelId = labelId;
			this.labelName = labelName;
		}
		
		/**
		 * Constructor with label.
		 * @param label label.
		 */
		public Label(Label label) {
			this(label.labelId, label.labelName);
		}
		
		/**
		 * Default constructor.
		 */
		public Label() {
			
		}
		
		/**
		 * Sorting label array.
		 * @param labels array.
		 * @param ascend ascending flag.
		 */
		public static void sort(List<Label> labels, boolean ascend) {
			Collections.sort(labels, new Comparator<Label>() {
				@Override
				public int compare(Label o1, Label o2) {
					if (o1.labelId < o2.labelId)
						return ascend ? -1 : 1;
					else if (o1.labelId == o2.labelId)
						return 0;
					else
						return ascend ? 1 : -1;
				}
			});
		}
		
	}

	
	/**
	 * Getting label.
	 * @return label.
	 */
	Label getLabel();
	
	
	/**
	 * Setting label.
	 * @param label specified label.
	 */
	void setLabel(Label label);

	
	/**
	 * Getting label identifier.
	 * @return label identifier.
	 */
	int getLabelId();
	
	
	/**
	 * Setting label identifier.
	 * @param labelId specified label identifier.
	 */
	void setLabelId(int labelId);
	
	
	/**
	 * Getting label name.
	 * @return label name.
	 */
	String getLabelName();
	
	
	/**
	 * Setting label name.
	 * @param labelName specified label name.
	 */
	void setLabelName(String labelName);

	
	/**
	 * Clearing label.
	 */
	void clearLabel();

	
	/**
	 * Shallow duplicating.
	 * @return duplicated property.
	 */
	RasterProperty shallowDuplicate();
	
	
}
