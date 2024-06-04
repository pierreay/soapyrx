#!/usr/bin/python3

import time
from os import path
try:
    import tomllib
# NOTE: For Python <= 3.11:
except ModuleNotFoundError as e:
    import tomli as tomllib

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import click

from soapyrx import lib as soapysdr_lib
from soapyrx import analyze
from soapyrx import log as l
from soapyrx import plot as libplot

# * Helper functions

def exit_on_cond(cond, ret=1):
    if cond is True:
        exit(ret)

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
@click.argument("freq_nf", type=float)
@click.argument("freq_ff", type=float)
@click.argument("samp_rate", type=float)
@click.option("--duration", type=float, default=0.5, help="Duration of the recording.")
@click.option("--nf-id", default=-1, help="Enable and associate radio index to near-field (NF) recording.")
@click.option("--ff-id", default=-1, help="Enable and associate radio index to far-field (FF) recording.")
@click.option("--gain", type=int, default=76, help="Gain for the SDR.")
def listen(freq_nf, freq_ff, samp_rate, duration, nf_id, ff_id, gain):
    """Initialize the radio and listen for commands.
    
    This commands will put our radio module in server mode, where the radio is
    listening for commands from another process through a pipe to perform
    recordings. This process will not go in background automatically, hence,
    use Bash to launch it in the background.

    """
    # Initialize the radio as requested.
    with soapysdr_lib.MySoapySDRs() as rad:
        # Initialize the radios individually.
        try:
            if nf_id != -1:
                rad_nf = soapysdr_lib.MySoapySDR(samp_rate, freq_nf, nf_id, duration=duration, dir=DIR, gain=gain)
                rad.register(rad_nf)
            if ff_id != -1:
                rad_ff = soapysdr_lib.MySoapySDR(samp_rate, freq_ff, ff_id, duration=duration, dir=DIR, gain=gain)
                rad.register(rad_ff)
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
    soapysdr_lib.MySoapySDRsClient().quit()

@cli.command()
@click.argument("samp_rate", type=float)
@click.option("--amplitude/--no-amplitude", default=True, help="Plot the amplitude of the traces.")
@click.option("--phase/--no-phase", default=False, help="Plot the phase of the traces.")
@click.option("--nf-id", default=-1, help="Enable and associate radio index to near-field (NF) recording.")
@click.option("--ff-id", default=-1, help="Enable and associate radio index to far-field (FF) recording.")
@click.option("--fast/--no-fast", default=False, help="Decimate the signal to speed-up plotting.")
def plot(samp_rate, amplitude, phase, nf_id, ff_id, fast):
    """Plot RAW traces from DIR.

    SAMP_RATE is the sampling rate used for both recording.

    """
    s_arr = []
    nf_arr = None
    ff_arr = None
    # Load the traces and quit with an error if nothing is choosen.
    if nf_id != -1:
        nf_arr = load_raw_trace(DIR, nf_id, 0, log=True)
        s_arr.append(nf_arr)
    if ff_id != -1:
        ff_arr = load_raw_trace(DIR, ff_id, 0, log=True)
        s_arr.append(ff_arr)
    if nf_arr is None and ff_arr is None:
        l.LOGGER.error("we need at least one trace index to plot!")
        exit(1)
    # Truncate the traces to the exact size for plotting using synchronized axis.
    s_arr = np.asarray(analyze.truncate_min(s_arr))
    # Plot the result.
    if amplitude is True:
        libplot.plot_time_spec_sync_axis(s_arr, samp_rate, comp=analyze.CompType.AMPLITUDE, fast=fast)
    if phase is True:
        libplot.plot_time_spec_sync_axis(s_arr, samp_rate, comp=analyze.CompType.PHASE, fast=fast)

@cli.command()
@click.argument("samp_rate", type=float)
@click.argument("file", type=click.Path())
@click.option("--cut/--no-cut", "cut_flag", default=False, help="Cut the recorded signal.")
@click.option("--save", default="", help="If set to a file path, save the recorded signal as .npy file.")
@click.option("--save-plot", default="", help="If set to a file path, save the plot to this path.")
@click.option("--freq", type=float, default=None, help="Set the center frequency for the spectrogram.")
def plot_file(samp_rate, file, cut_flag, save, save_plot, freq):
    """Plot a trace from FILE.

    SAMP_RATE is the sampling rate used for the recording.

    """
    sig = np.load(file)
    if cut_flag is True:
        pltshrk = libplot.PlotShrink(sig)
        pltshrk.plot()
        sig = pltshrk.get_signal_from(sig)
    libplot.SignalQuadPlot(sig, sr=samp_rate, fc=freq).plot(save=save_plot, title=file)
    # Save the signal as requested.
    if save != "":
        l.LOGGER.info("Additional save of plotted signal to: {}".format(save))
        np.save(save, sig)

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
        with soapysdr_lib.MySoapySDR(fs=samp_rate, freq=freq, idx=0, duration=duration, dir=DIR, gain=gain) as rad:
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
    rad = soapysdr_lib.MySoapySDRsClient()
    # Record and save the signal.
    rad.record()
    rad.accept()
    rad.save()
    # NOTE: MySoapySDRsClient.save() is not synchronous, then wait enough for signal to be saved.
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

if __name__ == "__main__":
    cli()
