package net.ea.ann.adapter.ui;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.nio.file.Path;
import java.util.List;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFormattedTextField;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.KeyStroke;
import javax.swing.WindowConstants;
import javax.swing.text.NumberFormatter;

import net.ea.ann.core.Util;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;
import net.hudup.core.logistic.ui.JImageList.ImagePathList;
import net.hudup.core.logistic.ui.TextArea;
import net.hudup.core.logistic.ui.TextField;
import net.hudup.core.logistic.ui.UIUtil;

/**
 * This class is the extended image list with image directory.
 * @author Loc Nguyen
 * @version 1.0
 */
public class ImagePathListExt extends ImagePathList {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Information dialog size.
	 */
	protected final static Dimension DIALOG_INFO_SIZE = new Dimension(300, 200);

	
	/**
	 * Icon size.
	 */
	protected int iconSize = ImageListItem.ICON_MINSIZE;

	
	/**
	 * Default constructor.
	 */
	public ImagePathListExt() {
		super();
	}

	
	/**
	 * Constructor from image directory.
	 * @param imageDir image directory.
	 * @param iconSize icon size.
	 * @param storeImage flag to indicate whether to store image.
	 */
	public ImagePathListExt(Path imageDir, int iconSize, boolean storeImage) {
		super(imageDir, iconSize, storeImage);
		this.iconSize = iconSize;
	}

	
	/**
	 * Constructor from image directory.
	 * @param imageDir image directory.
	 * @param iconSize icon size.
	 */
	public ImagePathListExt(Path imageDir, int iconSize) {
		super(imageDir, iconSize);
		this.iconSize = iconSize;
	}

	
	/**
	 * Constructor from image directory.
	 * @param imageDir image directory.
	 */
	public ImagePathListExt(Path imageDir) {
		super(imageDir);
	}

	
	/**
	 * Getting this image list.
	 * @return this image list.
	 */
	protected ImagePathListExt getThisImageList() {
		return this;
	}
	
	
	/**
	 * Getting icon size.
	 * @return icon size.
	 */
	protected int getIconSize() {
		return iconSize;
	}
	
	
	/**
	 * Loading rasters.
	 * @param dirOrFile directory or file.
	 * @return list of rasters.
	 */
	protected List<Raster> loadRasters(Path dirOrFile) {
		return RasterAssoc.load(dirOrFile);
	}

	
	/**
	 * Setting list data by rasters
	 * @param commonName common name.
	 * @param rasters raster
	 */
	public void setRasters(String commonName, List<Raster> rasters) {
		List<ImageListItem<Path>> items = Util.newList(rasters.size());
		for (int i = 0; i < rasters.size(); i++) {
			Raster raster = rasters.get(i);
			java.awt.Image image = raster.getRepImage();
			String name = RasterAssoc.genDefaultName(commonName, raster.getDefaultFormat(), i+1);
			ImageListItem<Path> item = image != null ? ImagePathList.createItem(name, image, getIconSize()) : ImagePathList.createItem(name);
			if (item != null) {
				item.setTag(raster);
				items.add(item);
			}
		}
		setListData(items);
	}
	
	
	/**
	 * Adding rasters
	 * @param commonName common name.
	 * @param rasters raster
	 * @return added items.
	 */
	public List<ImageListItem<Path>> addRasters(String commonName, List<Raster> rasters) {
		List<ImageListItem<Path>> items = Util.newList(rasters.size());
		for (int i = 0; i < rasters.size(); i++) {
			Raster raster = rasters.get(i);
			java.awt.Image image = raster.getRepImage();
			String name = RasterAssoc.genDefaultName(commonName, raster.getDefaultFormat(), i+1);
			ImageListItem<Path> item = image != null ? addItem(name, image, getIconSize()) : addItem(name);
			if (item != null) {
				setItemRaster(item, raster);
				items.add(item);
			}
		}
		return items;
	}

	
	/**
	 * Getting raster of item.
	 * @param item specified item.
	 * @return raster of item.
	 */
	public Raster getItemRaster(ImageListItem<Path> item) {
    	if (item == null) return null;
    	Object tag = item.getTag();
    	return tag != null && tag instanceof Raster ? (Raster)tag : null;
	}
	
	
	/**
	 * Setting raster of item.
	 * @param item specified item.
	 * @param raster specified raster. It can be null.
	 */
	public void setItemRaster(ImageListItem<Path> item, Raster raster) {
		if (item == null) return;
		item.setTag(raster);
	}
	
	
	/**
     * Querying raster at specified index.
     * @param index specified index.
     * @return raster at specified index.
     */
    public Raster queryItemRaster(int index) {
    	ImageListItem<Path> item = getItem(index);
    	if (item == null) return null;
    	Raster raster = getItemRaster(item);
    	if (raster != null) return raster;
    	
    	Path path = item.queryPath();
    	if (path == null) return null;
    	List<Raster> rasters = loadRasters(path);
    	return rasters.size() > 0 ? rasters.get(0) : null;
    }
    
    
    /**
     * Querying all rasters.
     * @return all rasters.
     */
    public List<Raster> queryItemRasters() {
    	int n = getItemCount();
    	List<Raster> rasters = Util.newList(n);
    	for (int i = 0; i < n; i++) {
    		Raster raster = queryItemRaster(i);
    		if (raster != null) rasters.add(raster);
    	}
    	
    	return rasters;
    }

    
	@Override
	protected void addToContextMenu(JPopupMenu contextMenu) {
		super.addToContextMenu(contextMenu);
		if (contextMenu == null) return;
		
		if (contextMenu.getComponentCount() > 0) contextMenu.addSeparator();

		JMenuItem mniRaster = new JMenuItem(
			new AbstractAction("Raster") {
				
				/**
				 * Serial version UID for serializable class. 
				 */
				private static final long serialVersionUID = 1L;

				@Override
				public void actionPerformed(ActionEvent e) {
					int index = getSelectedIndex();
					if (index >= 0) rasterInfo(getItem(index), queryItemRaster(index));
				}
				
			});
		if (getSelectedIndex() >= 0) contextMenu.add(mniRaster);
	}

	
	@Override
	protected void tagUI(ImageListItem<Path> item) {
		Raster raster = getItemRaster(item);
		if (raster != null)
			rasterInfo(item, raster);
		else
			super.tagUI(item);
	}

	
	/**
	 * This class represents dialog of raster information.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	private class RasterInfo extends JDialog {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Internal raster.
		 */
		protected Raster raster = null;
		
		/**
		 * Information text area.
		 */
		protected TextArea txtInfo = null;
		
		/**
		 * Constructor with specfied raster.
		 * @param raster specfied raster.
		 */
		public RasterInfo(Raster raster) {
			super(UIUtil.getWindowForComponent(getThisImageList()), "Raster information", ModalityType.APPLICATION_MODAL);

			this.raster = raster;
			setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
			setSize(DIALOG_INFO_SIZE);
			setLocationRelativeTo(UIUtil.getWindowForComponent(getThisImageList()));
			setLayout(new BorderLayout());

			JMenuBar mnuBar = createMenuBar();
		    if (mnuBar != null) setJMenuBar(mnuBar);

		    JPanel body = new JPanel(new BorderLayout());
			add(body, BorderLayout.CENTER);

			this.txtInfo = new TextArea("");
			this.txtInfo.setEditable(false);
			body.add(new JScrollPane(this.txtInfo), BorderLayout.CENTER);
			updateRasterInfo();
			
			JPanel footer = new JPanel();
			add(footer, BorderLayout.SOUTH);
			
			JButton close = new JButton("Close");
			close.addActionListener(new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					dispose();
				}
				
			});
			footer.add(close);
		}
		
		/**
		 * Creating main menu bar.
		 * @return main menu bar.
		 */
		protected JMenuBar createMenuBar() {
			JMenuBar mnBar = new JMenuBar();
			
			JMenu mnFile = new JMenu("File");
			mnFile.setMnemonic('f');

			JMenuItem mniSetLabel = new JMenuItem(
				new AbstractAction("Set label") {
					
					/**
					 * Serial version UID for serializable class. 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public void actionPerformed(ActionEvent e) {
						setLabel();
					}
					
				});
			mniSetLabel.setMnemonic('l');
			mniSetLabel.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, InputEvent.CTRL_DOWN_MASK));
			mnFile.add(mniSetLabel);

			if (mnFile.getMenuComponentCount() > 0) mnBar.add(mnFile);
			
			return mnBar.getMenuCount() > 0 ? mnBar : null;
		}

		/**
		 * Updating raster information.
		 */
		private void updateRasterInfo() {
			StringBuffer buffer = new StringBuffer();
			try {
				RasterProperty rp = raster.getProperty();
				buffer.append("Label = " + rp.getLabelId() + " with name '" + rp.getLabelName() + "'");
			} catch (Throwable e) {Util.trace(e);}
			txtInfo.setText(buffer.toString());
		}
		
		/**
		 * Setting label.
		 */
		private void setLabel() {
			RasterProperty rp = raster.getProperty();
			NumberFormatter formatter = new NumberFormatter();
			formatter.setAllowsInvalid(false);

			JDialog dlgSetting = new JDialog(this, "Set label", true);
			dlgSetting.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
			dlgSetting.setSize(DIALOG_INFO_SIZE);
			dlgSetting.setLocationRelativeTo(this);
			dlgSetting.setLayout(new BorderLayout());
			
			JPanel header = new JPanel(new BorderLayout());
			dlgSetting.add(header, BorderLayout.NORTH);
			
			JPanel left = new JPanel(new GridLayout(0, 1));
			header.add(left, BorderLayout.WEST);
			
			left.add(new JLabel("Label:"));
			left.add(new JLabel("Label name:"));
			
			JPanel right = new JPanel(new GridLayout(0, 1));
			header.add(right, BorderLayout.CENTER);

			JFormattedTextField txtLabel = new JFormattedTextField(formatter);
			txtLabel.setValue(rp.getLabelId());
			right.add(txtLabel);

			TextField txtLabelName = new TextField(rp.getLabelName());
			right.add(txtLabelName);

			JPanel footer = new JPanel();
			dlgSetting.add(footer, BorderLayout.SOUTH);
			
			JButton ok = new JButton("OK");
			ok.addActionListener(new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					Object labelValue = txtLabel.getValue();
					if ((labelValue == null) || !(labelValue instanceof Number)) return;
					int label = ((Number)labelValue).intValue();
					if (label < 0) {
						JOptionPane.showMessageDialog(dlgSetting, "Negative label", "Negative label", JOptionPane.ERROR_MESSAGE);
						return;
					}
					
					String labelName = txtLabelName.getText();
					if (labelName != null) labelName = labelName.trim();
					if (labelName == null || labelName.isEmpty()) {
						JOptionPane.showMessageDialog(dlgSetting, "Empty label name", "Empty label name", JOptionPane.ERROR_MESSAGE);
						return;
					}
					
					rp.setLabelId(label);
					rp.setLabelName(labelName);
					updateRasterInfo();
					dlgSetting.dispose();
				}
			});
			footer.add(ok);
			
			JButton reset = new JButton("Reset");
			reset.addActionListener(new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					txtLabel.setValue(rp.getLabelId());
					txtLabelName.setText(rp.getLabelName());
				}
				
			});
			footer.add(reset);

			JButton clear = new JButton("Clear");
			clear.addActionListener(new ActionListener() {
				
				@Override
				public void actionPerformed(ActionEvent e) {
					rp.clearLabel();
					updateRasterInfo();
					dlgSetting.dispose();
				}
				
			});
			footer.add(clear);

			JButton cancel = new JButton("Cancel");
			cancel.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					dlgSetting.dispose();
				}
			});
			footer.add(cancel);

			dlgSetting.setVisible(true);
		}
		
	}
	
	
	/**
	 * Raster information.
	 * @param raster specified raster.
	 */
	private void rasterInfo(ImageListItem<Path> item, Raster raster) {
		if (item == null) return;
		if (getItemRaster(item) != raster) setItemRaster(item, raster);
		if (raster == null) return;
		new RasterInfo(raster).setVisible(true);
	}

	
}
