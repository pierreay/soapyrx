#!/usr/bin/env python3

"""My library module."""

# Message to print from our library. Needs to be initialized.
MSG = None

# Initialize the library of our package.
def hello_init():
    global MSG
    MSG="Hello world from 'mylib.py'!"

# Library function of our package.
def hello_world():
    assert MSG is not None, "Package has not been initialized!"
    print(MSG)

# Test the library.
if __name__ == "__main__":
    print("Call hello_init() from 'mylib.py'...")
    hello_init()
    print("Call hello_world() from 'mylib.py'...")
    hello_world()
