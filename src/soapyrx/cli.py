#!/usr/bin/python3

from os import path
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import click
import time

try:
    import tomllib
# NOTE: For Python <= 3.11:
except ModuleNotFoundError as e:
    import tomli as tomllib

import lib.dataset as dataset
import lib.debug
import lib.analyze as analyze
import lib.load as load
import lib.device as device
import lib.log as l
import lib.plot as libplot
import lib.filters as filters
import lib.triggers as triggers
import lib.soapysdr as soapysdr
import lib.complex as complex
import lib.utils as utils

# * Helper functions

def exit_on_cond(cond, ret=1):
    if cond is True:
        exit(ret)

# * Global variables

CONFIG = None
DIR = None

# * Command-line interface

@click.group(context_settings={'show_default': True})
@click.option("--config", type=click.Path(), default="config.toml", help="Path of the TOML configuration file.")
@click.option("--dir", type=click.Path(), default="/tmp", help="Temporary directory used to hold raw recording.")
@click.option("--log/--no-log", default=True, help="Enable or disable logging.")
@click.option("--loglevel", default="DEBUG", help="Set the logging level.")
def cli(config, dir, log, loglevel):
    """Signal recording utility.

    CONFIG is the configuation file.

    """
    global CONFIG, DIR
    # Load the configuration file.
    with open(config, "rb") as f:
        CONFIG = tomllib.load(f)
    # Set the temporary directory.
    l.configure(log, loglevel)
    DIR = path.expanduser(dir)

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
    with soapysdr.MySoapySDRs() as rad:
        # Initialize the radios individually.
        try:
            if nf_id != -1:
                rad_nf = soapysdr.MySoapySDR(samp_rate, freq_nf, nf_id, duration=duration, dir=DIR, gain=gain)
                rad.register(rad_nf)
            if ff_id != -1:
                rad_ff = soapysdr.MySoapySDR(samp_rate, freq_ff, ff_id, duration=duration, dir=DIR, gain=gain)
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
    soapysdr.MySoapySDRsClient().quit()

@cli.command()
@click.argument("indir", type=click.Path())
@click.argument("subset", type=str)
@click.argument("bd_addr_src")
@click.argument("bd_addr_dest")
@click.argument("ser_port")
@click.option("--radio/--no-radio", default=True, help="Enable or disable the radio recording (instrument only).")
@click.option("--idx", default=0, help="Current recording index to get correct dataset's inputs.")
@click.option("--config", default="example", help="Select the device configuration based on device.NAME in the configuration file.")
def instrument(indir, subset, bd_addr_src, bd_addr_dest, ser_port, radio, idx, config):
    """Instrument the device and record RAW traces to DIR.

    Trigger the target device and store the RAW recording of the communication
    in DIR.

    INDIR is the path to the dataset. It is used for collection parameters
    (e.g. sample rate) and input values (e.g. key and plaintext).

    SUBSET corresponds to the subset's name where the inputs comes from.

    BD_ADDR_SRC the Bluetooth address of the spoofed legitimate device.

    BD_ADDR_DEST is the Bluetooth address of the target device to connect to.

    SER_PORT is the serial port of the target device to connect to.

    """    
    # Load the dataset.
    dset = dataset.Dataset.pickle_load(indir, quit_on_error=True)
    sset = dset.get_subset(subset)

    # Perform optionally sanity-check about user request.
    try:
        dset.is_able_to_instrument(sset, idx)
    except Exception as e:
        l.log_n_exit("Can't instrument current index, exit!", code=1, e=e)

    # Initialize the radio client.
    rad = soapysdr.MySoapySDRsClient(enabled=radio)

    # Initalize the device.
    with device.Device(cfg=CONFIG["device"][config], ser_port=ser_port, baud=115200, bd_addr_src=bd_addr_src, bd_addr_dest=bd_addr_dest, radio=rad, dset=dset, sset=sset) as dev:
        # Configure everything related to current trace index.
        dev.configure(idx)
        # Perform the instrumentation and the recording.
        try:
            dev.execute()
        except Exception as e:
            l.log_n_exit("error during whad instrumentation", 3, e, traceback=True)
        # Save the inputs used by device (NOTE: Only useful for runtime
        # generated using pairing).
        dev.save(idx)
    # Save the radio capture after success.
    rad.save()
    # NOTE: Here we save the entire dataset. In fact, it is useful for saving
    # new inputs and pairing parameters generated during run time when
    # dset.sset.input_gen == InputGeneration.RUN_TIME.
    dset.pickle_dump(force=True)

# NOTE: Skeleton for a new instrumentation code.
# @cli.command()
# @click.argument("indir", type=click.Path())
# @click.argument("subset", type=str)
# @click.argument("ser_port")
# @click.option("--radio/--no-radio", default=True, help="Enable or disable the radio recording (instrument only).")
# @click.option("--idx", default=0, help="Current recording index to get correct dataset's inputs.")
# def instrument_custom(indir, subset, ser_port, radio, idx):
#     """Instrument a device using custom firwmare.

#     Trigger the target device and store the RAW recording of the communication
#     in DIR.

#     INDIR is the path to the dataset. It is used for collection parameters
#     (e.g. sample rate) and input values (e.g. key and plaintext).

#     SUBSET corresponds to the subset's name where the inputs comes from.

#     SER_PORT is the serial port of the target device to connect to.

#     """    
#     # Load the dataset.
#     dset = dataset.Dataset.pickle_load(indir, quit_on_error=True)
#     sset = dset.get_subset(subset)

#     # Perform optionally sanity-check about user request.
#     try:
#         dset.is_able_to_instrument(sset, idx)
#     except Exception as e:
#         l.log_n_exit("Can't instrument current index, exit!", code=1, e=e)

#     # Initialize the radio client.
#     rad = soapysdr.MySoapySDRsClient(enabled=radio)

#     # Initalize the device.
#     with device.DeviceCustom(ser_port=ser_port, baud=115200, radio=rad, dset=dset, sset=sset) as dev:
#         # Configure everything related to current trace index.
#         dev.configure(idx)
#         # Perform the instrumentation and the recording.
#         try:
#             dev.execute()
#         except Exception as e:
#             l.log_n_exit("error during custom instrumentation", 3, e, traceback=True)
#     # Save the radio capture after success.
#     rad.save()

@cli.command()
@click.argument("freq", type=float)
@click.argument("samp_rate", type=float)
@click.argument("id_ref", type=int)
@click.option("--plot/--no-plot", default=True, help="Plot a summary of the processing.")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite the original RAW traces with extracted window.")
@click.option("--id", type=int, multiple=True, help="Radio indexes on which apply trace extraction (in addition to ID_REF). Can be specified multiple time.")
@click.option("--exit-on-error/--no-exit-on-error", default=False, help="If true, exit with error on bad AES detection instead of saving a bad trace.")
@click.option("--config", default="1_aes", help="Select the extractor configuration based on radio.extract.NAME in the configuration file.")
@click.option("--save", default="", type=click.Path(), help="If set to a file path, save the ID_REF extracted signal as .npy file without custom dtype. Ignored if --overwrite is set to False.")
@click.option("--corr", default="", type=click.Path(), help="If set to a file path, cross-correlate the results against the signal.")
@click.option("--save-plot", default="", help="If set to a file path (without extension), save the plot to this path.")
def extract(freq, samp_rate, id_ref, plot, overwrite, id, exit_on_error, config, save, corr, save_plot):
    """Extract RAW traces from DIR.

    Extract a rough window around interesting signals from just-recorded RAW
    traces. It uses one raw trace as reference to find the interesting signal
    and extract the rough window for all specified raw traces.

    FREQ is the center frequency of the recording.
    SAMP_RATE is the sampling rate used for both recording.
    
    ID_REF is the radio index to use for extraction reference. Set it to -1 to
    ignore the extraction.

    """
    if id_ref == -1:
        exit(0)
    else:
        l.LOGGER.info("Extract RAW trace using ID #{}".format(id_ref))

    # Get the configuration for AES extraction.
    window              = CONFIG["radio"]["extract"][config]["window"]
    offset              = CONFIG["radio"]["extract"][config]["offset"]
    trg_bp_low          = CONFIG["radio"]["extract"][config]["trg_bp_low"]
    trg_bp_high         = CONFIG["radio"]["extract"][config]["trg_bp_high"]
    trg_lp              = CONFIG["radio"]["extract"][config]["trg_lp"]
    trg_peak_prominence = CONFIG["radio"]["extract"][config]["trg_peak_prominence"]
    l.LOGGER.debug("Configuration: {}".format(config))
    l.LOGGER.debug("Peak search prominence={}".format(trg_peak_prominence))

    # * Loading.
    # sig_raw_ref = analyze.normalize(complex.get_amplitude(load.load_raw_trace(DIR, id_ref, 0)))
    # NOTE: I didn't use the normalized version anymore because it helps to
    # distinguish between real signal / noise by looking at amplitude value,
    # while it doesn't decrease extraction accuracy.
    sig_raw_ref = complex.get_amplitude(load.load_raw_trace(DIR, id_ref, 0))

    # * Triggering.
    assert(len(trg_bp_low) == len(trg_bp_high))
    TRG_NB = len(trg_bp_low)
    # ** Trigger signals.
    nf_triggers = triggers.Triggers()
    for idx in list(range(TRG_NB)):
        nf_triggers.add(triggers.Trigger(sig_raw_ref, trg_bp_low[idx], trg_bp_high[idx], trg_lp, samp_rate))
        l.LOGGER.debug(nf_triggers.get(idx))
    nf_triggers.reduce_add()
    # ** Trigger indexes.
    peaks = signal.find_peaks(nf_triggers.get(0).signal, distance=samp_rate/10, prominence=trg_peak_prominence)
    peaks = peaks[0]
    l.LOGGER.info("Number of detected peaks={}".format(len(peaks)))

    # Compute cross-correlation if specified.
    if corr != "":
        for idx, peak in enumerate(peaks):
            corr_ref = np.abs(np.load(corr))
            corr_ref = analyze.normalize(corr_ref)
            corr_test = analyze.extract_time_window(sig_raw_ref, samp_rate, peak, window, offset=offset)
            corr_test = analyze.normalize(corr_test)
            # NOTE: DEBUG
            # libplot.plot_time_spec_sync_axis([corr_ref], samp_rate=samp_rate, title="corr_ref")
            # libplot.plot_time_spec_sync_axis([corr_test], samp_rate=samp_rate, title="corr_test")
            corr_output = signal.correlate(corr_ref, corr_test)
            l.LOGGER.info("Correlation for peak #{}: min={};max={};median={};mean={}".format(idx, int(np.min(corr_output)), int(np.max(corr_output)), int(np.median(corr_output)), int(np.mean(corr_output))))
            
    # * Logging based on results.
    peak_detect_ok = len(peaks) == 1
    if peak_detect_ok is False:
        l.LOGGER.error("Signal localization confusion because no or multiple peaks are detected!")
    else:
        peak_position = (peaks[0] / len(sig_raw_ref)) * 100
        snr = utils.snr(sig_raw_ref, samp_rate, peaks[0])
        accept_snr_min = CONFIG["radio"]["extract"][config]["accept_snr_min"]
        accept_snr_max = CONFIG["radio"]["extract"][config]["accept_snr_max"]
        avg = int(utils.avg_window(sig_raw_ref, samp_rate, peaks[0]))
        l.LOGGER.info("Peak: Position={:.2f}% ; SNR={:.2} ; Avg={}".format(peak_position, snr, avg))
        if peak_position < 15 or peak_position > 85:
            l.LOGGER.warning("Peak is located at recording's boundaries, bad connection event?")
        if snr < accept_snr_min:
            l.LOGGER.error("Current SNR of {:.2f} is inferior to validated minimum SNR of {:.2f}".format(snr, accept_snr_min))
            peak_detect_ok = False
        if snr > accept_snr_max:
            l.LOGGER.error("Current SNR of {:.2f} is superior to validated maximum SNR of {:.2f}".format(snr, accept_snr_max))
            peak_detect_ok = False
        if avg < 1300:
            l.LOGGER.warning("Peak value is inferior to 1300, noise?")

    # Plotting based on results.
    if len(peaks) > 0 and plot is True:
        for idx, peak in enumerate(peaks):
            # libplot.plot_time_spec_sync_axis([analyze.extract_time_window(sig_raw_ref, samp_rate, peak, window, offset=offset)], samp_rate=samp_rate, cond=plot, title="Peak #{}".format(idx))
            sigtoplot = analyze.extract_time_window(load.load_raw_trace(DIR, rad_idx=0, rec_idx=0), samp_rate, peak, window, offset=offset)
            libplot.SignalQuadPlot(sigtoplot, sr=samp_rate, fc=freq).plot(save="{}_1.pdf".format(save_plot) if save_plot != "" else "", show=plot)
    libplot.plot_time_spec_sync_axis([sig_raw_ref], samp_rate=samp_rate, peaks=peaks, triggers=nf_triggers, cond=plot, save="{}_2.pdf".format(save_plot) if save_plot != "" else "")

    # Exit based on results.
    if peak_detect_ok is False:
        exit_on_cond(exit_on_error)

    # * Extraction.
    if overwrite:
        id = tuple(set(id + (id_ref,))) # Add id_ref and only get unique values of radio indexes.
        for idx in id:
            sig_raw = load.load_raw_trace(DIR, idx, 0)
            if peak_detect_ok is False:
                l.LOGGER.warning("replace recording by a bad trace")
                sig_raw = analyze.get_bad_trace(sig_raw)
                # Extract at middle of trace.
                peak = int(len(sig_raw) / 2)
            else:
                peak = peaks[0]
            sig_raw = analyze.extract_time_window(sig_raw, samp_rate, peak, window, offset=offset)
            l.LOGGER.info("overwrite extracted signal #{} in {}".format(idx, DIR))
            load.save_raw_trace(sig_raw, DIR, idx, 0)
            # * Handle --save switch.
            if save != "" and idx == id_ref:
                l.LOGGER.info("Save extracted I/Q from reference signal to {}".format(save))
                np.save(save, sig_raw)
    else:
        l.LOGGER.info("ignore overwrite for extracted signal(s)")

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
        nf_arr = load.load_raw_trace(DIR, nf_id, 0, log=True)
        s_arr.append(nf_arr)
    if ff_id != -1:
        ff_arr = load.load_raw_trace(DIR, ff_id, 0, log=True)
        s_arr.append(ff_arr)
    if nf_arr is None and ff_arr is None:
        l.LOGGER.error("we need at least one trace index to plot!")
        exit(1)
    # Truncate the traces to the exact size for plotting using synchronized axis.
    s_arr = np.asarray(load.truncate_min(s_arr))
    # Plot the result.
    if amplitude is True:
        libplot.plot_time_spec_sync_axis(s_arr, samp_rate, comp=complex.CompType.AMPLITUDE, fast=fast)
    if phase is True:
        libplot.plot_time_spec_sync_axis(s_arr, samp_rate, comp=complex.CompType.PHASE, fast=fast)

@cli.command()
@click.argument("samp_rate", type=float)
@click.argument("file", type=click.Path())
@click.option("--npy/--no-npy", type=bool, default="False", help="If set to true, assume FILE is a regular Numpy array instead of a custom dtype one produced by soapysdr.py")
@click.option("--cut/--no-cut", "cut_flag", default=False, help="Cut the recorded signal.")
@click.option("--save", default="", help="If set to a file path, save the recorded signal as .npy file without custom dtype.")
@click.option("--save-plot", default="", help="If set to a file path, save the plot to this path.")
@click.option("--freq", type=float, default=None, help="Set the center frequency for the spectrogram.")
def plot_file(samp_rate, file, npy, cut_flag, save, save_plot, freq):
    """Plot a trace from FILE.

    SAMP_RATE is the sampling rate used for the recording.

    """
    sig = soapysdr.MySoapySDR.numpy_load(file) if npy is False else np.load(file)
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
@click.option("--duration", type=float, default=0.5, help="Duration of the recording.")
@click.option("--save", default="", help="If set to a file path, save the recorded signal as .npy file without custom dtype.")
@click.option("--norm/--no-norm", default=False, help="Normalize the recording before saving.")
@click.option("--amplitude/--no-amplitude", default=False, help="Extract only the amplitude of the signal.")
@click.option("--phase/--no-phase", default=False, help="Extract only the phase of the signal.")
@click.option("--plot/--no-plot", "plot_flag", default=True, help="Plot the recorded signal.")
@click.option("--cut/--no-cut", "cut_flag", default=True, help="Cut the recorded signal.")
@click.option("--gain", type=int, default=76, help="Gain for the SDR.")
def record(freq, samp_rate, duration, save, norm, amplitude, phase, plot_flag, cut_flag, gain):
    """Record a trace without any instrumentation.

    It will automatically use the first found radio with ID 0.

    FREQ is the center frequency of the recording.
    SAMP_RATE is the sampling rate used for the recording.

    """
    rad_id=0
    # Initialize the radio as requested.
    try:
        with soapysdr.MySoapySDR(fs=samp_rate, freq=freq, idx=rad_id, duration=duration, dir=DIR, gain=gain) as rad:
            # Initialize the driver.
            rad.open()
            # Perform the recording.
            rad.record()
            # Save the radio capture on disk.
            rad.accept()
            rad.save(reinit=False)
            # Save the radio capture outside the radio for an additional save or plot.
            sig = rad.get_signal()
    except Exception as e:
        l.log_n_exit("Error during radio instrumentation", 1, e)
    # Cut the signal as requested.
    if cut_flag is True:
        pltshrk = libplot.PlotShrink(sig)
        pltshrk.plot()
        sig = pltshrk.get_signal_from(sig)
    # Plot the signal as requested.
    if plot_flag:
        libplot.SignalQuadPlot(sig, sr=samp_rate, fc=freq).plot()
    # Save the signal as requested.
    if save != "":
        sig = analyze.process_iq(sig, amplitude=amplitude, phase=phase, norm=norm, log=True)
        l.LOGGER.info("Additional save of recorded signal to: {}".format(save))
        np.save(save, sig)

@cli.command()
@click.option("--save", default="", help="If set to a file path, save the recorded signal as .npy file without custom dtype.")
@click.option("--norm/--no-norm", default=False, help="Normalize the recording before saving.")
@click.option("--amplitude/--no-amplitude", default=False, help="Extract only the amplitude of the signal.")
@click.option("--phase/--no-phase", default=False, help="Extract only the phase of the signal.")
@click.option("--plot/--no-plot", "plot_flag", default=True, help="Plot the recorded signal.")
def client(save, norm, amplitude, phase, plot_flag):
    """Record a signal by connecting to the running and configured SDR server.

    It will automatically use the first found radio with ID 0.

    """
    rad_id=0
    rad = soapysdr.MySoapySDRsClient()
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
    sig = load.load_raw_trace(dir=DIR, rad_idx=rad_id, rec_idx=0, log=False)
    # Plot the signal as requested [amplitude by default].
    comp = complex.CompType.PHASE if phase is True else complex.CompType.AMPLITUDE
    libplot.plot_time_spec_sync_axis([sig], comp=comp, cond=plot_flag, xtime=False)
    # Save the signal as requested.
    if save != "":
        sig = analyze.process_iq(sig, amplitude=amplitude, phase=phase, norm=norm, log=True)
        l.LOGGER.info("Additional save of recorded signal to: {}".format(save))
        np.save(save, sig)

@cli.command()
def debug():
    """Debug currently recorded radio signals."""
    sig = load.load_raw_trace(DIR, 0, 0, log=True)
    from IPython import embed; embed()
        
@cli.command()
@click.argument("file")
def to_numpy(file):
    """Store a signal recorded in our custom dtype to Numpy format in FILE."""
    np.save(file, load.load_raw_trace(DIR, 0, 0, log=True))
    l.LOGGER.info("Signal saved in {}".format(file))

if __name__ == "__main__":
    cli()
