#!/usr/bin/python3

# * Importation

# Standard import.
import time
from os import path
from functools import partial

# Compatibility import.
try:
    import tomllib
# NOTE: For Python <= 3.11:
except ModuleNotFoundError as e:
    import tomli as tomllib

# External import.
import numpy as np
from matplotlib import pyplot as plt
import click

# Internal import.
from soapyrx import logger as l
from soapyrx import helpers
from soapyrx import plotters
from soapyrx import core

# * Global variables

CONFIG = None

# * Command-line interface

@click.group(context_settings={'show_default': True})
@click.option("--config", type=click.Path(), default="", help="Path of a TOML configuration file.")
@click.option("--log/--no-log", default=True, help="Enable or disable logging.")
@click.option("--loglevel", default="INFO", help="Set the logging level.")
def cli(config, log, loglevel):
    """Signal recording tool.

    CONFIG is the configuation file.

    """
    global CONFIG
    l.configure(log, loglevel)
    # Load the configuration file.
    if config != "" and path.exists(config):
        with open(config, "rb") as f:
            CONFIG = tomllib.load(f)
    elif config != "":
        l.LOGGER.error("Configuration file not found: {}".format(path.abspath(config)))

@cli.command()
def discover():
    """Discover SDRs.

    Discover connected SDRs and print capabilities.

    """
    helpers.discover()
    
@cli.command()
@click.argument("freq", type=float)
@click.argument("samp", type=float)
@click.option("--duration", type=float, default=1, help="Duration of the recording [s].")
@click.option("--gain", type=int, default=0, help="Gain for the SDR [dB].")
@click.option("--save-sig", default="", help="If set to a file path, save the recorded signal as .npy file.")
@click.option("--save-plot", default="", help="If set to a file path, save the plot to this path.")
@click.option("--plot/--no-plot", "plot_flag", default=True, help="Plot the recorded signal.")
@click.option("--cut/--no-cut", "cut_flag", default=True, help="Cut the recorded signal.")
@click.option("--live/--no-live", "live_flag", default=False, help="Live display instead of one shot recording.")
def record(freq, samp, duration, gain, save_sig, save_plot, plot_flag, cut_flag, live_flag):
    """Record a signal.

    It will automatically use the first found radio. FREQ is the center
    frequency (e.g., 2.4e9). SAMP is the sampling rate (e.g., 4e6).

    """
    if live_flag is False:
        sig = helpers.record(freq=freq, samp=samp, duration=duration, gain=gain)
        helpers.plot(sig, samp=samp, freq=freq, cut_flag=cut_flag, plot_flag=plot_flag, save_sig=save_sig, save_plot=save_plot, title=save_sig)
    elif live_flag is True:
        plotters.SignalQuadPlot(None, sigfunc=partial(helpers.record, freq=freq, samp=samp, duration=duration, gain=gain), sr=samp, fc=freq).plot(save=False, title=save_sig)
        

@cli.command()
@click.argument("file", type=click.Path())
@click.option("--cut/--no-cut", "cut_flag", default=False, help="Cut the plotted signal.")
@click.option("--save-sig", default="", help="If set to a file path, save the plotted signal as .npy file.")
@click.option("--save-plot", default="", help="If set to a file path, save the plot to this path.")
@click.option("--freq", type=float, default=None, help="Center frequency of the recording (plot feature).")
@click.option("--samp", type=float, default=None, help="Sampling rate of the recording (plot feature).")
def plot(file, cut_flag, save_sig, save_plot, freq, samp):
    """Plot a signal.

    FILE is the path to the file.

    """
    # Signal loading block.
    try:
        sig = np.load(file)
    except Exception as e:
        l.LOGGER.critical("Cannot load signal from disk: {}".format(e)); exit(1)
    helpers.plot(sig, samp=samp, freq=freq, cut_flag=cut_flag, plot_flag=True, save_sig=save_sig, save_plot=save_plot, title=file)

@cli.command()
@click.argument("idx", type=int)
@click.argument("freq", type=float)
@click.argument("samp_rate", type=float)
@click.option("--duration", type=float, default=1, help="Duration of the recording [s].")
@click.option("--gain", type=int, default=0, help="Gain for the SDR [dB].")
def server_start(idx, freq, samp_rate, duration, gain):
    """Start a radio server.

    IDX is the radio index (see discover()). FREQ is the center frequency
    (e.g., 2.4e9). SAMP is the sampling rate (e.g., 4e6).
    
    Start the radio in server mode, listening for commands from another process
    to performs recordings. This process will not go in background
    automatically, hence, use Bash to launch it in the background.

    """
    helpers.server_start(idx, freq, samp_rate, duration, gain)

@cli.command()
def server_stop():
    """Stop a radio server.

    This command is used to properly quit the radio server instead of killing
    it, possibly letting the SDR driver in a bad state.

    """
    core.SoapyClient().stop()
    
@cli.command()
@click.option("--save-sig", default="", help="If set to a file path, save the recorded signal as .npy file.")
@click.option("--save-plot", default="", help="If set to a file path, save the plot to this path.")
@click.option("--plot/--no-plot", "plot_flag", default=True, help="Plot the recorded signal.")
@click.option("--cut/--no-cut", "cut_flag", default=True, help="Cut the recorded signal.")
@click.option("--live/--no-live", "live_flag", default=False, help="Live display instead of one shot recording.")
def client(save_sig, save_plot, plot_flag, cut_flag, live_flag):
    """Record a signal from a radio server."""    
    if live_flag is False:
        sig = helpers.client()
        helpers.plot(sig, cut_flag=cut_flag, plot_flag=plot_flag, save_sig=save_sig, save_plot=save_plot, title=save_sig)
    elif live_flag is True:
        plotters.SignalQuadPlot(None, sigfunc=partial(helpers.client)).plot(save=False, title=save_sig)
    
if __name__ == "__main__":
    cli()
