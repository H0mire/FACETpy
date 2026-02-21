"""
MATLAB Reference Implementation — Sub-sample Alignment

This module is a direct port of the MATLAB RAAlignSubSample routine kept as a
reference for the algorithm.  It is **not** used by the v2.0 pipeline; see
``facet.preprocessing.alignment.SubsampleAligner`` for the production processor.
"""

import numpy as np


class MatlabSubSampleAlignment:
    """Reference port of the MATLAB sub-sample alignment algorithm.

    This class relies on attributes (``TriggersUp``, ``SplitVector``, etc.)
    that only existed in the pre-v2.0 monolithic class and is therefore not
    directly usable in the current architecture.  It is retained solely for
    algorithmic reference.
    """

    def AlignSubSample(self):
        # maximum distance between triggers
        MaxTrigDist = np.max(np.diff(self.TriggersUp))
        NumSamples = MaxTrigDist + 20  # Note: this is different from self.NumSamples

        # phase shift (-1/2 .. +1/2, below it is multiplied by 2*pi -> -pi..pi)
        ShiftAngles = (
            np.arange(1, NumSamples + 1) - np.floor(NumSamples / 2) + 1
        ) / NumSamples

        # if we already have values from a previous run
        if self.SubSampleAlignment is None or len(self.SubSampleAlignment) == 0:
            if self.SSAHPFrequency is not None and self.SSAHPFrequency > 0:
                self.ProfileStart()
                nyq = 0.5 * self.SamplingFrequency
                f = [
                    0,
                    (self.SSAHPFrequency * 0.9) / (nyq * self.Upsample),
                    (self.SSAHPFrequency * 1.1) / (nyq * self.Upsample),
                    1,
                ]
                a = [0, 0, 1, 1]
                fw = np.firls(100, f, a)

                HPEEG = np.fft.ifft(
                    np.fft.ifftshift(np.fft.fftshift(np.fft.fft(self.RAEEGAcq)) * fw)
                ).real
                HPEEG = np.concatenate((HPEEG[101:], np.zeros(100)))  # undo the shift
                self.ProfileStop("SSA-Filter")
            else:
                HPEEG = self.RAEEGAcq

            # Split vector into 2D matrix
            EEGMatrix = self.SplitVector(
                HPEEG, self.TriggersUp - self.PreTrig - 10, NumSamples
            )
            EEG_Ref = EEGMatrix[self.AlignSlicesReference, :]

            self.SubSampleAlignment = np.zeros(self.NumTriggers)

            Corrs = np.zeros((self.NumTriggers, 20))
            Shifts = np.zeros((self.NumTriggers, 20))

            self.ProfileStart()
            for Epoch in set(range(1, self.NumTriggers + 1)) - {
                self.AlignSlicesReference
            }:
                EEG_M = EEGMatrix[Epoch - 1, :]
                FFT_M = np.fft.fftshift(np.fft.fft(EEG_M))
                Shift_L = -1
                Shift_M = 0
                Shift_R = 1
                FFT_L = FFT_M * np.exp(-1j * 2 * np.pi * ShiftAngles * Shift_L)
                FFT_R = FFT_M * np.exp(-1j * 2 * np.pi * ShiftAngles * Shift_R)
                EEG_L = np.fft.ifft(np.fft.ifftshift(FFT_L)).real
                EEG_R = np.fft.ifft(np.fft.ifftshift(FFT_R)).real
                Corr_L = self.Compare(EEG_Ref, EEG_L)
                Corr_M = self.Compare(EEG_Ref, EEG_M)
                Corr_R = self.Compare(EEG_Ref, EEG_R)

                FFT_Ori = FFT_M

                for Iteration in range(15):
                    Corrs[Epoch - 1, Iteration] = Corr_M
                    Shifts[Epoch - 1, Iteration] = Shift_M

                    if Corr_L > Corr_R:
                        Corr_R = Corr_M
                        EEG_R = EEG_M
                        FFT_R = FFT_M
                        Shift_R = Shift_M
                    else:
                        Corr_L = Corr_M
                        EEG_L = EEG_M
                        FFT_L = FFT_M
                        Shift_L = Shift_M

                    Shift_M = (Shift_L + Shift_R) / 2
                    FFT_M = FFT_Ori * np.exp(-1j * 2 * np.pi * ShiftAngles * Shift_M)
                    EEG_M = np.fft.ifft(np.fft.ifftshift(FFT_M)).real
                    Corr_M = self.Compare(EEG_Ref, EEG_M)

                self.SubSampleAlignment[Epoch - 1] = Shift_M
                EEGMatrix[Epoch - 1, :] = EEG_M

            self.ProfileStop("SSA-Iterate")

            # Further processing not shown due to length. Follow the same structure for the rest of
            self.ProfileStart()
            # Loop over every epoch for final alignment
            for Epoch in set(range(1, self.NumTriggers + 1)) - {
                self.AlignSlicesReference
            }:
                EEG = EEGMatrix[Epoch - 1, :]
                FFT = np.fft.fftshift(np.fft.fft(EEG))
                FFT *= np.exp(
                    -1j * 2 * np.pi * ShiftAngles * self.SubSampleAlignment[Epoch - 1]
                )
                EEG = np.fft.ifft(np.fft.ifftshift(FFT)).real
                EEGMatrix[Epoch - 1, :] = EEG
            self.ProfileStop("SSA-Shift")

            # Join epochs
            for s in range(self.NumTriggers):
                start_index = self.TriggersUp[s] - self.PreTrig
                end_index = start_index + self.ArtLength
                self.RAEEGAcq[start_index:end_index] = EEGMatrix[
                    s, 10 : 10 + self.ArtLength
                ]

    def Compare(self, Ref, Arg):
        # Implementation of the Compare function
        # result = -sum((Ref-Arg)**2)
        return -np.sum((Ref - Arg) ** 2)
