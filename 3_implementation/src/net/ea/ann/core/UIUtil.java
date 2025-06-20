/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.ActionListener;
import java.net.URL;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JMenuItem;

/**
 * This utility class provides utility methods relevant to user interface (UI) such as getting image from URI, creating button, and creating menu item.
 *  
 * @author Loc Nguyen
 * @version 10.0
 */
public final class UIUtil {

	
	/**
	 * The directory contains images.
	 */
	private final static String  IMAGES_PACKAGE           = "/net/ea/ann/core/resources/images/";

	
	/**
	 * Retrieving URL of an image from image name.
	 * The directory of such image is specified by constants {@link #IMAGES_PACKAGE}.
	 * @param imageName image name.
	 * @return URL of an image from image name.
	 */
	private static URL getImageUrl(String imageName) {
		if (imageName == null || imageName.isEmpty())
			return null;
		
		URL imageURL = null;
		try {
			String path = IMAGES_PACKAGE;
			if (!path.endsWith("/"))
				path = path + "/";
			path = path + imageName;
			imageURL = UIUtil.class.getResource(path);
		}
		catch (Exception e) {
			Util.trace(e);
			imageURL = null;
		}
		
		return imageURL;
	}
	
	
	/**
	 * Creating an image from image name.
	 * The directory of such image is specified by constants {@link #IMAGES_PACKAGE}.
	 * @param imageName image name.
	 * @return image from image name.
	 */
	public static Image getImage(String imageName) {
		URL url = getImageUrl(imageName);
		if (url == null)
			return null;
		
		try {
			return Toolkit.getDefaultToolkit().getImage(url);
		}
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	
	
	/**
	 * Creating an icon from icon name.
	 * The directory of such icon is specified by constants {@link #IMAGES_PACKAGE}.
	 * @param iconName icon name.
	 * @param alt alternative name.
	 * @return icon from icon name.
	 */
	public static ImageIcon getImageIcon(String iconName, String alt) {
		if (alt == null)
			return new ImageIcon(getImageUrl(iconName));
		else
			return new ImageIcon(getImageUrl(iconName), alt);
	}
	
	
	/**
	 * Creating an icon button.
	 * @param iconName icon file name.
	 * The directory of icon is specified by constants {@link #IMAGES_PACKAGE}.
	 * @param cmd action command for this button.
	 * @param tooltip tool-tip for this button.
	 * @param alt alternative text for this button.
	 * @param listener action listener for this button.
	 * @return {@link JButton} with icon.
	 */
	public static JButton makeIconButton(String iconName, String cmd, String tooltip, String alt, 
			ActionListener listener) {
		
		return makeIconButton(getImageUrl(iconName), cmd, tooltip, alt, listener);
	}
	
	
	/**
	 * Creating an icon button.
	 * @param iconURL {@link URL} of icon. URL, abbreviation of Uniform Resource Locator, locates any resource on internet.
	 * @param cmd action command for this button.
	 * @param tooltip tool-tip for this button.
	 * @param alt alternative text for this button.
	 * @param listener action listener for this button.
	 * @return {@link JButton} with icon.
	 */
	private static JButton makeIconButton(URL iconURL, String cmd, String tooltip, String alt, 
			ActionListener listener) {
		JButton button = new JButton();
	    button.setActionCommand(cmd);
	    button.setToolTipText(tooltip);
	    button.addActionListener(listener);
	
	    try {
		    if (iconURL != null) {
		        button.setIcon(new ImageIcon(iconURL, alt));
		    }
		    else {  //no image found
		        button.setText(alt);
		    }
	    }
	    catch (Exception e) {
	    	Util.trace(e);
	    }
	    
	    return button;
	}

	
	/**
	 * Creating a check-box menu item with icon.
	 * @param iconName icon file name.
	 * The directory of icon is specified by constants {@link #IMAGES_PACKAGE}.
	 * @param text text of menu item.
	 * @param listener action listener for this menu item.
	 * @return {@link JCheckBoxMenuItem} with icon.
	 */
	public static JCheckBoxMenuItem makeCheckBoxMenuItem(String iconName, String text, ActionListener listener) {
		return (JCheckBoxMenuItem)makeMenuItem(getImageUrl(iconName), text, listener, true);
	}

	
	/**
	 * Creating a menu item with icon.
	 * @param iconName icon file name.
	 * The directory of icon is specified by constants {@link #IMAGES_PACKAGE}.
	 * @param text text of menu item.
	 * @param listener action listener for this menu item.
	 * @return {@link JMenuItem} with icon.
	 */
	public static JMenuItem makeMenuItem(String iconName, String text, ActionListener listener) {
		return makeMenuItem(getImageUrl(iconName), text, listener, false);
	}
	
	
	/**
	 * Creating a menu item with icon.
	 * @param iconURL {@link URL} of icon. URL, abbreviation of Uniform Resource Locator, locates any resource on Internet.
	 * @param text text of menu item.
	 * @param listener action listener for this menu item.
	 * @param isCheckbox true if this is check-box menu item.
	 * @return {@link JMenuItem} with icon.
	 */
	private static JMenuItem makeMenuItem(URL iconURL, String text, ActionListener listener, boolean isCheckbox) {
		JMenuItem item = null;
		if (isCheckbox)
			item = new JCheckBoxMenuItem(text);
		else
			item = new JMenuItem(text);
		
		item.addActionListener(listener);
		
		try {
		    if (iconURL != null) {
		        item.setIcon(new ImageIcon(iconURL, text));
		    }
		}
		catch (Exception e) {
			Util.trace(e);
		}
		
	    return item;
	}

	
}
