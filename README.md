# SoapyRX

An SDR receiver built on [SoapySDR](https://github.com/pothosware/SoapySDR/wiki).

The scope of this tool is to handle signal acquisition from an SDR.
Using SoapySDR as backend, it supports every SDR supported by the latter -- as long as the corresponding module is installed.
It is not meant to process datasets of signals (*e.g.*, conversion, alignment, normalization).

- **Features:**
    - One-shot record-plot-cut-save from the command-line.
    - Server mode controlled from a client (local socket).
    - Multiple synchronized SDRs controlled in parallel.
    - Display basic signal visualization (amplitude, phase rotation, time-domain, frequency-domain).
    - Perform single-signal operations (*e.g.*, interactive cutting).
    - Use complex numbers (I/Q) for storage (`numpy.complex64`).
    - Live display (buffered recording, like an oscilloscope, not like a spectrum analyzer).
-   **Supported devices:**
    - [RTL-SDR](https://www.rtl-sdr.com/)
    - [HackRF](https://greatscottgadgets.com/hackrf/one/)
    - [USRP](https://www.ettus.com/product-categories/usrp-bus-series/)
    - [SDRPlay](https://www.sdrplay.com/)
    - [AirSpy](https://airspy.com/)

**References**

SoapyRX can be used independently, but has been initially developed for the PhaseSCA project, see the main project repository at **[phasesca](https://github.com/pierreay/phasesca.git)**.

# Installation

1. Install [SoapySDR](https://github.com/pothosware/SoapySDR/wiki) and the module corresponding to the desired SDR using packages (refer to your distribution) or sources (refer to [Installation of SoapySDR from sources](#installation-of-soapysdr-from-sources)).

2.  Clone SoapyRX repository and install it using using [Pip](https://pypi.org/project/pip/):

```bash
git clone https://github.com/pierreay/soapyrx.git
cd soapyrx && pip install --user .
```

# Installation of SoapySDR from sources

1. Install the required dependencies using your distribution package manager:
    - `cmake`
    - `g++`
    - `python3`
    - `numpy`
    - `dbg`
    - `swig`
    - `boost`
    - `boost-thread`

2. Get the source code and build:

```bash
cd /tmp && git clone https://github.com/pothosware/SoapySDR.git
mkdir SoapySDR/build build && cd SoapySDR/build
cmake .. && make -j4
```

3. Install and test if SoapySDR is installed correctly:

```bash
sudo make install && sudo ldconfig
SoapySDRUtil --info
```

4. We support the following modules:
    - [SoapyRTLSDR](https://github.com/pothosware/SoapyRTLSDR)
    - [SoapyHackRF](https://github.com/pothosware/SoapyHackRF.git)
    - [SoapyUHD](https://github.com/pothosware/SoapyUHD.git)
    - [SoapySDRPlay3](https://github.com/pothosware/SoapySDRPlay3)
    - [SoapyAirspy](https://github.com/pothosware/SoapyAirspy)

5. Select the module according to your SDR and ensure that the SDR driver is correctly installed on your system, independently of SoapySDR.

6. Then, get the source code and build, replacing `$MODULE_URL` and `$MODULE_NAME` accordingly:

```bash
cd /tmp && git clone $MODULE_URL
mkdir $MODULE_NAME/build && cd $MODULE_NAME/build
cmake .. && make
```

7. Install and test if SoapySDR is able to detect the USRP, replacing `$MODULE_DRIVER` by:
    - `rtlsdr` for RTL-SDR.
    - `hackrf` for SoapyHackRF.
    - `uhd` for SoapyUHD.
    - `sdrplay` for SoapySDRPlay3.
    - `airspy` for AirSpy.
    
```bash
sudo make install
SoapySDRUtil --probe="driver=$MODULE_DRIVER"
```
