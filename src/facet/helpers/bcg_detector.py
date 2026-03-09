"""
BCG Detector - QRS Peak Detection Implementation

This module provides tools for detecting QRS peaks from ECG data using
combined adaptive thresholding techniques based on [Christov04, Niazy06].

The main function detects QRS peaks by:
1. Computing the Teager Energy Operator (TEO) [Kim04]
2. Computing an adaptive threshold for each sampling point
3. Detecting and correcting QRS peaks

References:
-----------
[Niazy06] R.K. Niazy, C.F. Beckmann, G.D. Iannetti, J.M. Brady, and
 S.M. Smith (2005) Removal of FMRI environment artifacts from EEG data
 using optimal basis sets. NeuroImage 28 (3), pages 720-737.

[Christov04] Real time electrocardiogram QRS detection using combined
 adaptive threshold, Ivaylo I. Christov. Biomedical Engineering Online,
 BioMed Central (2004).

[Kim04] Improved ballistocardiac artifact removal from the
 electroencephalogram recored in FMRI, KH Kim, HW Yoon, HW Park.
 J NeouroSience Methods 135 (2004) 193-203.

This is a Python implementation based on the original MATLAB code by Rami Niazy.
"""

import logging

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def prcorr2(a, b):
    """
    Correlation coefficient between two matrices/vectors.
    Python implementation of the MATLAB prcorr2 function.

    Parameters
    ----------
    a, b : array_like
        Arrays to be correlated. Must be the same size.

    Returns
    -------
    r : float
        Correlation coefficient.
    """
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same size")

    # Compute the correlation coefficient
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_centered = a - a_mean
    b_centered = b - b_mean

    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))

    if denominator == 0:
        return 0

    return numerator / denominator


def qrscorrect(Peaks, Ecg, fs):
    """
    Correct for false positive and negative QRS peaks and align peaks for maximum correlation.

    Parameters
    ----------
    Peaks : array-like
        Indices of located QRS peaks
    Ecg : array-like
        Vector of ECG data
    fs : float
        Sampling frequency

    Returns
    -------
    Peaks : array-like
        Corrected peak indices
    """
    # Convert to numpy array if not already
    Peaks = np.array(Peaks, dtype=int)
    Ecg = np.array(Ecg, dtype=float)

    # Initialize
    P = np.zeros(len(Ecg), dtype=int)
    P[Peaks] = 1

    if len(Peaks) < 2:
        logger.warning("Not enough peaks for correction, returning original peaks")
        return Peaks

    dP_std = np.std(np.diff(Peaks))
    dP_med = np.median(np.diff(Peaks))
    pad = int(round(5 * fs))
    secL = int(round(10 * fs))
    sections = int(np.floor(len(P) / secL)) + 1

    if len(P[(sections - 1) * secL + 1 :]) < pad:
        sections -= 1

    # Progress display variables
    barth = 2
    barth_step = barth

    # Stage 2: Correct for False Positives (FP)
    print("Stage 2 of 5: Correcting for False Positive detection.")
    Flag25, Flag50, Flag75 = False, False, False

    for s in range(1, sections + 1):
        # Progress reporting
        percentdone = int(s * 100 / sections)
        if int(percentdone) >= barth:
            if percentdone >= 25 and not Flag25:
                print("25% ", end="", flush=True)
                Flag25 = True
            elif percentdone >= 50 and not Flag50:
                print("50% ", end="", flush=True)
                Flag50 = True
            elif percentdone >= 75 and not Flag75:
                print("75% ", end="", flush=True)
                Flag75 = True
            elif percentdone == 100:
                print("100%")
            else:
                print(".", end="", flush=True)

            while barth <= percentdone:
                barth += barth_step

            if barth > 100:
                barth = 100

        # Get section of P
        f_flag = False
        if s == 1:
            sec = P[: secL + pad].copy()
        elif s == sections:
            sec = P[(s - 1) * secL + 1 - pad :].copy()
        else:
            sec = P[(s - 1) * secL + 1 - pad : s * secL + pad].copy()

        sec_P = np.where(sec == 1)[0]

        if len(sec_P) < 2:
            continue

        sec_dP = np.diff(sec_P)

        # Remove peaks that are too close (false positives)
        false_p = [1]  # Dummy initial value
        while len(false_p) > 0:
            false_p = np.where(sec_dP < (dP_med - 3 * dP_std))[0]
            if len(false_p) > 0:
                f_flag = True
                sec[sec_P[false_p[0] + 1]] = 0
                sec_P = np.where(sec == 1)[0]

                if len(sec_P) < 2:
                    break

                sec_dP = np.diff(sec_P)

        # Update P with corrected section
        if f_flag:
            if s == 1:
                P[: secL + pad] = sec
            elif s == sections:
                P[(s - 1) * secL + 1 - pad :] = sec
            else:
                P[(s - 1) * secL + 1 - pad : s * secL + pad] = sec

    # Check edge peaks
    Peaks = np.where(P == 1)[0]

    if len(Peaks) < 2:
        logger.warning("Not enough peaks after false positive correction, returning original peaks")
        return Peaks

    pc1 = 2
    pc2 = 1

    while pc1 > 1 or pc2 > 0:
        dP = np.diff(Peaks)
        if len(dP) == 0:  # Handle case with only one peak left
            return Peaks

        hwinl = int(round(np.mean(dP) / 2))
        searchw = int(round(0.33 * np.mean(dP)))

        pc1 = 1
        if len(Peaks) > 5:
            for p in range(5):
                try:
                    _ = Ecg[Peaks[p] - hwinl - searchw : Peaks[p] + hwinl]
                    break
                except Exception:
                    pc1 += 1

        if pc1 > 1:
            Peaks = Peaks[pc1 - 1 :]

        if len(Peaks) < 2:  # Handle case with too few peaks left
            return Peaks

        pc2 = 0
        if len(Peaks) > 5:
            for p in range(len(Peaks) - 1, len(Peaks) - 6, -1):
                if p < 0:  # Handle case with fewer than 5 peaks
                    break
                try:
                    _ = Ecg[Peaks[p] - hwinl : Peaks[p] + hwinl + searchw]
                    break
                except Exception:
                    pc2 += 1

        if pc2 > 0:
            Peaks = Peaks[:-pc2]

    if len(Peaks) < 2:  # Handle case with too few peaks left
        return Peaks

    # Stage 3: Finding corrected peaks by alignment
    # First calculate the average QRS template
    peaksL = len(Peaks)
    qrs = np.zeros((hwinl * 2 + 1, peaksL))

    # Build template of average heartbeat
    for p in range(peaksL):
        if Peaks[p] - hwinl >= 0 and Peaks[p] + hwinl < len(Ecg):
            qrs[:, p] = Ecg[Peaks[p] - hwinl : Peaks[p] + hwinl + 1]

    m_qrs = np.mean(qrs, axis=1)

    # Stage 3: Align Peaks (first pass)
    print("Stage 3 of 5: Aligning QRS Peaks (1)")
    Flag25, Flag50, Flag75 = False, False, False
    barth = 5

    for p in range(peaksL):
        # Progress reporting
        percentdone = int(p * 100 / peaksL)
        if int(percentdone) >= barth:
            if percentdone >= 25 and not Flag25:
                print("25% ", end="", flush=True)
                Flag25 = True
            elif percentdone >= 50 and not Flag50:
                print("50% ", end="", flush=True)
                Flag50 = True
            elif percentdone >= 75 and not Flag75:
                print("75% ", end="", flush=True)
                Flag75 = True
            elif percentdone == 100:
                print("100%")
            else:
                print(".", end="", flush=True)

            while barth <= percentdone:
                barth += barth_step

            if barth > 100:
                barth = 100

        # Skip if too close to edge
        if Peaks[p] - hwinl - searchw < 0 or Peaks[p] + hwinl + searchw >= len(Ecg):
            continue

        # Find best correlation with template
        C = np.zeros(2 * searchw + 1)
        for i, B in enumerate(range(Peaks[p] - searchw, Peaks[p] + searchw + 1)):
            if B - hwinl >= 0 and B + hwinl < len(Ecg):
                C[i] = prcorr2(Ecg[B - hwinl : B + hwinl + 1], m_qrs)

        # Get position with maximum correlation
        CP = np.argmax(C)
        Beta = CP - searchw
        Peaks[p] = Peaks[p] + Beta

    # Update P with aligned peaks
    P = np.zeros(len(Ecg), dtype=int)
    P[Peaks] = 1

    # Stage 4: Correct for False Negatives (FN)
    print("Stage 4 of 5: Correcting for False Negative detection.")
    Flag25, Flag50, Flag75 = False, False, False
    barth = 5

    for s in range(1, sections + 1):
        # Progress reporting
        percentdone = int(s * 100 / sections)
        if int(percentdone) >= barth:
            if percentdone >= 25 and not Flag25:
                print("25% ", end="", flush=True)
                Flag25 = True
            elif percentdone >= 50 and not Flag50:
                print("50% ", end="", flush=True)
                Flag50 = True
            elif percentdone >= 75 and not Flag75:
                print("75% ", end="", flush=True)
                Flag75 = True
            elif percentdone == 100:
                print("100%")
            else:
                print(".", end="", flush=True)

            while barth <= percentdone:
                barth += barth_step

            if barth > 100:
                barth = 100

        # Get section of P
        f_flag = False
        if s == 1:
            sec = P[: secL + pad].copy()
        elif s == sections:
            sec = P[(s - 1) * secL + 1 - pad :].copy()
        else:
            sec = P[(s - 1) * secL + 1 - pad : s * secL + pad].copy()

        sec_P = np.where(sec == 1)[0]

        if len(sec_P) < 2:
            continue

        sec_dP = np.diff(sec_P)
        dP_med = np.median(sec_dP)

        # Add peaks where there's a gap (false negatives)
        false_n = [1]  # Dummy initial value
        while len(false_n) > 0:
            false_n = np.where(sec_dP > (1.5 * dP_med))[0]
            if len(false_n) > 0:
                f_flag = True
                new_peak_pos = sec_P[false_n[0]] + int(round(dP_med))
                if new_peak_pos < len(sec):
                    sec[new_peak_pos] = 1
                sec_P = np.where(sec == 1)[0]

                if len(sec_P) < 2:
                    break

                sec_dP = np.diff(sec_P)

        # Update P with corrected section
        if f_flag:
            if s == 1:
                P[: secL + pad] = sec
            elif s == sections:
                P[(s - 1) * secL + 1 - pad :] = sec
            else:
                P[(s - 1) * secL + 1 - pad : s * secL + pad] = sec

    # Stage 5: Final alignment of peaks
    Peaks = np.where(P == 1)[0]
    peaksL = len(Peaks)

    if peaksL < 3:  # Need at least 3 peaks for second alignment (for p in range(1, peaksL-1))
        return Peaks

    print("Stage 5 of 5: Aligning QRS Peaks (2)")
    Flag25, Flag50, Flag75 = False, False, False
    barth = 5

    for p in range(1, peaksL - 1):  # Skip first and last peak
        # Progress reporting
        percentdone = int((p - 1) * 100 / (peaksL - 2))
        if int(percentdone) >= barth:
            if percentdone >= 25 and not Flag25:
                print("25% ", end="", flush=True)
                Flag25 = True
            elif percentdone >= 50 and not Flag50:
                print("50% ", end="", flush=True)
                Flag50 = True
            elif percentdone >= 75 and not Flag75:
                print("75% ", end="", flush=True)
                Flag75 = True
            elif percentdone == 100:
                print("100%")
            else:
                print(".", end="", flush=True)

            while barth <= percentdone:
                barth += barth_step

            if barth > 100:
                barth = 100

        # Skip if too close to edge
        if Peaks[p] - hwinl - searchw < 0 or Peaks[p] + hwinl + searchw >= len(Ecg):
            continue

        # Find best correlation with template
        C = np.zeros(2 * searchw + 1)
        for i, B in enumerate(range(Peaks[p] - searchw, Peaks[p] + searchw + 1)):
            if B - hwinl >= 0 and B + hwinl < len(Ecg):
                C[i] = prcorr2(Ecg[B - hwinl : B + hwinl + 1], m_qrs)

        # Get position with maximum correlation
        CP = np.argmax(C)
        Beta = CP - searchw
        Peaks[p] = Peaks[p] + Beta

    return Peaks


def fmrib_qrsdetect(raw, ecg_channel=None):
    """
    Detect QRS peaks from ECG channel using combined adaptive thresholding.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing the EEG/ECG data
    ecg_channel : str or int, optional
        The name or index of the ECG channel. If None, tries to find it automatically.

    Returns
    -------
    peaks : ndarray
        Array of QRS peak indices in samples
    """

    # Find ECG channel if not specified
    if ecg_channel is None:
        ecg_picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=True)
        if len(ecg_picks) == 0:
            # Try to find channels with ECG in the name
            ecg_picks = [i for i, ch in enumerate(raw.ch_names) if "ECG" in ch.upper() or "EKG" in ch.upper()]

        if len(ecg_picks) == 0:
            raise ValueError("No ECG channels found in the data")

        ecg_channel = ecg_picks[0]

    # Get ECG data and sampling rate
    if isinstance(ecg_channel, str):
        ecg_data = raw.get_data(picks=ecg_channel)[0]
    else:
        ecg_data = raw.get_data(picks=[ecg_channel])[0]

    ofs = raw.info["sfreq"]

    # Determine decimation factor
    dL = 1
    dLFlag = True
    if np.mod(ofs, 128) == 0:
        dL = int(ofs / 128)
    elif np.mod(ofs, 100) == 0:
        dL = int(ofs / 100)
    elif np.mod(ofs, 125) == 0:
        dL = int(ofs / 125)
    else:
        dL = int(np.round(ofs / 100))
        if ofs / dL < 100:
            dLFlag = False
            dL = 1

    # Decimate signal
    if dLFlag and dL > 1:
        try:
            if dL > 4:
                if np.mod(dL, 2) == 0:
                    Ecg = signal.decimate(ecg_data, int(dL / 2))
                    Ecg = signal.decimate(Ecg, 2)
                elif np.mod(dL, 3) == 0:
                    Ecg = signal.decimate(ecg_data, int(dL / 3))
                    Ecg = signal.decimate(Ecg, 3)
                elif np.mod(dL, 5) == 0:
                    Ecg = signal.decimate(ecg_data, int(dL / 5))
                    Ecg = signal.decimate(Ecg, 5)
                elif np.mod(dL, 7) == 0:
                    Ecg = signal.decimate(ecg_data, int(dL / 7))
                    Ecg = signal.decimate(Ecg, 7)
                elif np.mod(dL, 9) == 0:
                    Ecg = signal.decimate(ecg_data, int(dL / 9))
                    Ecg = signal.decimate(Ecg, 9)
                else:
                    Ecg = signal.decimate(ecg_data, int(dL))
            else:
                Ecg = signal.decimate(ecg_data, int(dL))
        except Exception as e:
            logger.warning(f"Decimation failed: {e}. Using original signal.")
            Ecg = ecg_data
            dL = 1
            dLFlag = False
    else:
        Ecg = ecg_data

    fs = ofs / dL

    if np.any(np.isnan(Ecg)):
        raise ValueError("Decimation failed. Downsample the data first and try again.")

    # MFR Settings and initialization
    L = len(Ecg)
    msWait = int(0.55 * fs)
    ms1200 = int(1.2 * fs)
    ms350 = int(0.35 * fs)
    ms300 = int(0.3 * fs)
    ms50 = int(0.05 * fs)
    Mc = 0.45
    s5 = int(5 * fs)
    DetectFlag = False
    timer1 = 0
    Peaks = []
    peakc = 0
    firstdetect = True
    Ecg = Ecg.reshape(-1, 1).flatten()

    # Allocate memory
    M = np.zeros(L)
    R = np.zeros(L)
    F = np.zeros(L)
    MFR = np.zeros(L)
    Y = np.zeros(L)
    M5 = np.ones(5)
    R5 = np.ones(5)
    np.zeros(ms350)

    # Pre-processing filtering
    fL = int(np.round(fs / 50))
    b = np.ones(fL) / fL
    Ecg = signal.filtfilt(b, 1, Ecg)

    fL = int(np.round(fs / 35))
    b = np.ones(fL) / fL
    Ecg = signal.filtfilt(b, 1, Ecg)

    # Estimate initial R and k
    FFTp = int(np.round(100 * fs))
    P2 = int(np.ceil(np.log(FFTp) / np.log(2)))
    NFFT = 2**P2

    Fecg = np.fft.fft(signal.detrend(Ecg[: int(5 * fs)]) * np.hanning(len(Ecg[: int(5 * fs)])), NFFT)
    Pecg = Fecg * np.conj(Fecg) / NFFT
    ML = np.argmax(np.abs(Pecg))

    R5 = R5 * np.round(NFFT / ML)
    k = int(np.round(fs * fs * np.pi / (2 * 2 * np.pi * 10 * R5[0])))

    # Construct complex lead Y using TEO
    f = np.array([0, 7 / (fs / 2), 9 / (fs / 2), 40 / (fs / 2), 42 / (fs / 2), 1])
    a = np.array([0, 0, 1, 1, 0, 0])
    wts = signal.firls(101, f, a)
    ecgF = signal.filtfilt(wts, 1, Ecg)

    for n in range(k + 1, L - k):
        Y[n] = ecgF[n] ** 2 - ecgF[n - k] * ecgF[n + k]

    Y[L - 1] = 0
    fL = int(np.round(fs / 25))
    b = np.ones(fL) / fL
    Y = signal.filtfilt(b, 1, Y)
    Y[Y < 0] = 0

    # Initialize M and F
    M5 = Mc * np.max(Y[int(fs) : int(fs + s5)]) * M5
    M[:s5] = np.mean(M5)
    newM5 = np.mean(M5)
    F[:ms350] = np.mean(Y[int(fs) : int(fs + ms350)])
    F2 = np.copy(F)

    # Detect QRS
    print("Stage 1 of 5: Adaptive threshold peak detection.")
    barth = 5
    barth_step = barth
    Flag25, Flag50, Flag75 = False, False, False

    for n in range(L):
        timer1 += 1

        if len(Peaks) >= 2:
            if DetectFlag:
                DetectFlag = False
                M[n] = np.mean(M5)
                Mdec = (M[n] - M[n] * Mc) / (ms1200 - msWait)
                Rdec = Mdec / 1.4
            elif not DetectFlag and (timer1 <= msWait or timer1 > ms1200):
                M[n] = M[n - 1]
            elif not DetectFlag and timer1 == msWait + 1:
                M[n] = M[n - 1] - Mdec
                newM5 = Mc * np.max(Y[n - msWait : n])
                if newM5 > 1.5 * M5[4]:
                    newM5 = 1.5 * M5[4]
                M5 = np.append(M5[1:], newM5)
            elif not DetectFlag and timer1 > msWait + 1 and timer1 <= ms1200:
                M[n] = M[n - 1] - Mdec

        if n > ms350:
            F[n] = F[n - 1] + (np.max(Y[n - ms50 + 1 : n + 1]) - np.max(Y[n - ms350 + 1 : n - ms300 + 1])) / 150
            F2[n] = F[n] - np.mean(Y[int(fs) : int(fs + ms350)]) + newM5

        Rm = np.mean(R5)
        R0int = int(np.round(2 * Rm / 3))

        if timer1 <= R0int:
            R[n] = 0
        elif len(Peaks) >= 2:
            R[n] = R[n - 1] - Rdec

        MFR[n] = M[n] + F2[n] + R[n]

        if (Y[n] >= MFR[n] and timer1 > msWait) or (Y[n] >= MFR[n] and firstdetect):
            if firstdetect:
                firstdetect = False

            Peaks.append(n)
            if len(Peaks) > 1:
                R5 = np.append(R5[1:], Peaks[-1] - Peaks[-2])

            peakc += 1
            DetectFlag = True
            timer1 = -1

        # Progress reporting
        percentdone = int(n * 100 / L)
        if int(percentdone) >= barth:
            if percentdone >= 25 and not Flag25:
                print("25% ", end="", flush=True)
                Flag25 = True
            elif percentdone >= 50 and not Flag50:
                print("50% ", end="", flush=True)
                Flag50 = True
            elif percentdone >= 75 and not Flag75:
                print("75% ", end="", flush=True)
                Flag75 = True
            elif percentdone == 100:
                print("100%")
            else:
                print(".", end="", flush=True)

            while barth <= percentdone:
                barth += barth_step

            if barth > 100:
                barth = 100

    # Correct QRS peaks
    Peaks = np.array(Peaks)
    Peaks = qrscorrect(Peaks, Ecg, fs)
    Peaks = Peaks * dL

    return Peaks


def plot_qrs_detection(raw, peaks, ecg_channel=None, start_time=0, duration=10):
    """
    Plot the ECG signal with detected QRS peaks.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing the EEG/ECG data
    peaks : array-like
        Indices of the detected QRS peaks
    ecg_channel : str or int, optional
        Name or index of the ECG channel. If None, tries to find it automatically.
    start_time : float, optional
        Start time in seconds for the plot
    duration : float, optional
        Duration in seconds for the plot
    """
    if ecg_channel is None:
        ecg_picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=True)
        if len(ecg_picks) == 0:
            ecg_picks = [i for i, ch in enumerate(raw.ch_names) if "ECG" in ch.upper() or "EKG" in ch.upper()]

        if len(ecg_picks) == 0:
            raise ValueError("No ECG channels found in the data")

        ecg_channel = ecg_picks[0]

    # Get ECG data
    if isinstance(ecg_channel, str):
        ecg_data = raw.get_data(picks=ecg_channel)[0]
    else:
        ecg_data = raw.get_data(picks=[ecg_channel])[0]

    # Get time array
    times = raw.times

    # Calculate sample indices for the plot window
    start_idx = int(start_time * raw.info["sfreq"])
    end_idx = int((start_time + duration) * raw.info["sfreq"])

    # Select peaks within the plot window
    plot_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]

    # Create the plot
    plt.figure(figsize=(15, 5))
    plt.plot(times[start_idx:end_idx], ecg_data[start_idx:end_idx])
    plt.plot(times[plot_peaks], ecg_data[plot_peaks], "ro")
    plt.title("ECG with Detected QRS Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    return plt
