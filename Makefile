# * Variables

# ** Project configuration

# Package name.
PKG_NAME=soapyrx

# Package directory.
# NOTE: This corresponds to the so-called "src" layout of Python documentation.
PKG_DIR=src

# * Recipes

# ** Installing/Uninstalling to user using PipX

# Install the project user-wide using dynamic installation method (symlinking files).
install:
	pipx install --editable .

# Uninstall the project user-wide.
uninstall: clean
	pipx uninstall $(PKG_NAME)

# ** Cleaning

clean: 
	rm -rf build
	rm -rf $(PKG_DIR)/$(PKG_NAME).egg-info
	rm -rf $(PKG_DIR)/$(PKG_NAME)/__pycache__
