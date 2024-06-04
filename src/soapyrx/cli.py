#!/usr/bin/python3

# Standard import.
import time
from os import path

# Compatibility import.
try:
    import tomllib
# NOTE: For Python <= 3.11:
except ModuleNotFoundError as e:
    import tomli as tomllib

# External import.
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import click
import SoapySDR

# Internal imports.
from soapyrx import lib as soapysdr_lib
from soapyrx import analyze
from soapyrx import log as l
from soapyrx import plot as libplot

# * Global variables

CONFIG = None
DIR = None

# * Functions from old project

def load_raw_trace(dir, rad_idx, rec_idx, log=False):
    trace = None
    fp = path.join(dir, "raw_{}_{}.npy".format(rad_idx, rec_idx))
    if path.exists(fp):
        l.LOGGER.info("Load RAW trace from {}".format(fp))
        try:
            trace = np.load(fp)
        except Exception as e:
            print(e)
    else:
        l.LOGGER.warning("No loaded raw trace for radio index #{}!".format(rad_idx))
    return trace

# * Command-line interface

@click.group(context_settings={'show_default': True})
@click.option("--config", type=click.Path(), default="", help="Path of a TOML configuration file.")
@click.option("--dir", type=click.Path(), default="/tmp", help="Temporary directory used to hold raw recording.")
@click.option("--log/--no-log", default=True, help="Enable or disable logging.")
@click.option("--loglevel", default="INFO", help="Set the logging level.")
def cli(config, dir, log, loglevel):
    """Signal recording utility.

    CONFIG is the configuation file.

    """
    global CONFIG, DIR
    l.configure(log, loglevel)
    # Set the temporary directory.
    DIR = path.expanduser(dir)
    # Load the configuration file.
    if config != "" and path.exists(config):
        with open(config, "rb") as f:
            CONFIG = tomllib.load(f)
    elif config != "":
        l.LOGGER.error("Configuration file not found: {}".format(path.abspath(config)))

@cli.command()
@click.argument("freq", type=float)
@click.argument("samp_rate", type=float)
@click.option("--duration", type=float, default=0.5, help="Duration of the recording.")
@click.option("--id", default=-1, help="Enable radio index.")
@click.option("--gain", type=int, default=76, help="Gain for the SDR.")
def listen(freq, samp_rate, duration, id, gain):
    """Initialize the radio and listen for commands.
    
    This commands will put our radio module in server mode, where the radio is
    listening for commands from another process through a pipe to perform
    recordings. This process will not go in background automatically, hence,
    use Bash to launch it in the background.

    """
    # Initialize the radio as requested.
    with soapysdr_lib.SoapyServer() as rad:
        # Initialize the radios individually.
        try:
            if id != -1:
                rad_id = soapysdr_lib.SoapyRadio(samp_rate, freq, id, duration=duration, dir=DIR, gain=gain)
                rad.register(rad_id)
        except Exception as e:
            l.log_n_exit("Error during radio initialization", 1, e)
        if rad.get_nb() <= 0:
            l.LOGGER.error("we need at least one radio index to record!")
            exit(1)
        # Initialize the driver
        rad.open()
        # Listen for commands from another process.
        rad.listen()

@cli.command()
def quit():
    """Send a quit message to the listening radio server.

    This command is used to properly quit the radio server instead of killing
    it, possibly letting the SDR driver in a bad state.

    """
    soapysdr_lib.SoapyClient().quit()

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
    # Simple visualization and processing block.
    try:
        # Cut the signal as requested.
        if cut_flag is True:
            pltshrk = libplot.PlotShrink(sig, sr=samp, fc=freq)
            pltshrk.plot()
            sig = pltshrk.get_signal_from(sig)
        # Plot the signal.
        libplot.SignalQuadPlot(sig, sr=samp, fc=freq).plot(save=save_plot, title=file)
        # Save the signal as requested.
        if save_sig != "":
            l.LOGGER.info("Save recording: {}".format(save_sig))
            np.save(save_sig, sig)
    except Exception as e:
        l.LOGGER.critical("Error during signal processing!")
        raise e

@cli.command()
@click.argument("freq", type=float)
@click.argument("samp_rate", type=float)
@click.option("--duration", type=float, default=1, help="Duration of the recording [s].")
@click.option("--gain", type=int, default=0, help="Gain for the SDR [dB].")
@click.option("--save", "save_path", default="", help="If set to a file path, save the recorded signal as .npy file.")
@click.option("--plot/--no-plot", "plot_flag", default=True, help="Plot the recorded signal.")
@click.option("--cut/--no-cut", "cut_flag", default=True, help="Cut the recorded signal.")
def record(freq, samp_rate, duration, gain, save_path, plot_flag, cut_flag):
    """Record a signal.

    It will automatically use the first found radio. FREQ is the center
    frequency (e.g., 2.4e9). SAMP_RATE is the sampling rate (e.g., 4e6).

    """
    # Radio block.
    try:
        with soapysdr_lib.SoapyRadio(fs=samp_rate, freq=freq, idx=0, duration=duration, dir=DIR, gain=gain) as rad:
            # Initialize the driver.
            rad.open()
            # Perform the recording.
            rad.record()
            # Save the radio capture in temporary buffer.
            rad.accept()
            # Get the radio capture.
            sig = rad.get_signal()
    except Exception as e:
        l.LOGGER.critical("Error during radio recording!")
        raise e
    # Simple visualization and processing block.
    try:
        # Cut the signal as requested.
        if cut_flag is True:
            pltshrk = libplot.PlotShrink(sig, sr=samp_rate, fc=freq)
            pltshrk.plot()
            sig = pltshrk.get_signal_from(sig)
        # Plot the signal as requested.
        if plot_flag:
            libplot.SignalQuadPlot(sig, sr=samp_rate, fc=freq).plot()
        # Save the signal as requested.
        if save_path != "":
            l.LOGGER.info("Save recording: {}".format(save_path))
            np.save(save_path, sig)
    except Exception as e:
        l.LOGGER.critical("Error during signal processing!")
        raise e

@cli.command()
@click.option("--save", default="", help="If set to a file path, save the recorded signal as .npy file.")
@click.option("--norm/--no-norm", default=False, help="Normalize the recording before saving.")
@click.option("--amplitude/--no-amplitude", default=False, help="Extract only the amplitude of the signal.")
@click.option("--phase/--no-phase", default=False, help="Extract only the phase of the signal.")
@click.option("--plot/--no-plot", "plot_flag", default=True, help="Plot the recorded signal.")
def client(save, norm, amplitude, phase, plot_flag):
    """Record a signal by connecting to the running and configured SDR server.

    It will automatically use the first found radio with ID 0.

    """
    rad_id=0
    rad = soapysdr_lib.SoapyClient()
    # Record and save the signal.
    rad.record()
    rad.accept()
    rad.save()
    # NOTE: SoapyClient.save() is not synchronous, then wait enough for signal to be saved.
    time.sleep(1)
    # NOTE: Following code duplicated from `record()'.
    # Save the radio capture outside the radio for an additional save or plot.
    # NOTE: Not especially efficient since we use the disk as buffer here,
    # but the SDR client cannot receive data from the SDR server currently.
    sig = load_raw_trace(dir=DIR, rad_idx=rad_id, rec_idx=0, log=False)
    # Plot the signal as requested [amplitude by default].
    comp = analyze.CompType.PHASE if phase is True else analyze.CompType.AMPLITUDE
    libplot.plot_time_spec_sync_axis([sig], comp=comp, cond=plot_flag, xtime=False)
    # Save the signal as requested.
    if save != "":
         np.save(save, sig)

@cli.command()
def discover():
    """Discover SDRs.

    Discover connected SDRs and print capabilities.

    """
    results = SoapySDR.Device.enumerate()
    for idx, result in enumerate(results):
        l.LOGGER.info("{}".format(result))
        l.LOGGER.info("Index: {}".format(idx))
        sdr = SoapySDR.Device(result)
        # Query device info.
        l.LOGGER.info("Antennas: {}".format(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)))
        l.LOGGER.info("Gains: {}".format(sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)))
        freqRanges = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
        for freqRange in freqRanges:
            l.LOGGER.info("Frenquency range: {}".format(freqRange))
    if not results:
        l.LOGGER.error("No detected SDR!"); exit(1)
    
if __name__ == "__main__":
    cli()
