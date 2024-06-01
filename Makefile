# * Project configuration

# Package name.
PKG_NAME=mypkg

# Main module name.
MOD_NAME=mymod

# Package directory.
# NOTE: This corresponds to the so-called "src" layout of Python documentation.
PKG_DIR=src

# Python interpreter.
PYSHELL=python # [python | ipython]

# * Running

# Run the package main function as a module and exit.
run-once:
	PYTHONPATH=$(PKG_DIR) $(PYSHELL) -c "from $(PKG_NAME) import $(MOD_NAME); $(MOD_NAME).main();"

# Import the package main module and gives an interactive REPL.
run-repl:
	PYTHONPATH=$(PKG_DIR) $(PYSHELL) -i -c "from $(PKG_NAME) import $(MOD_NAME);"

# Run the package main module as a script.
run-script:
	PYTHONPATH=$(PKG_DIR) $(PKG_DIR)/$(PKG_NAME)/$(MOD_NAME).py

# * Testing

# Run the library test as a script.
test-lib:
	PYTHONPATH=$(PKG_DIR) $(PKG_DIR)/$(PKG_NAME)/mylib.py

# * Building

# Build the Python distribution of our project.
build:
	python -m build

# * Installing to root

# Install the project system-wide using static installation method (copying files).
install-root-dist:
	sudo pip install --break-system-packages .

# Install the project system-wide using dynamic installation method (symlinking files).
install-root-dev:
	sudo pip install --break-system-packages --editable .

# Uninstall the project system-wide.
uninstall-root:
	sudo pip uninstall --break-system-packages $(PKG_NAME)

# * Installing to user

# Install the project user-wide using static installation method (copying files).
install-user-dist:
	pip install --break-system-packages --user .

# Install the project user-wide using dynamic installation method (symlinking files).
install-user-dev:
	pip install --break-system-packages --user --editable .

# Uninstall the project user-wide.
uninstall-user:
	pip uninstall --break-system-packages $(PKG_NAME)

# * Cleaning

clean:
	sudo rm -rf build
	sudo rm -rf dist
	sudo rm -rf $(PKG_DIR)/$(PKG_NAME).egg-info
	sudo rm -rf $(PKG_DIR)/$(PKG_NAME)/__pycache__
