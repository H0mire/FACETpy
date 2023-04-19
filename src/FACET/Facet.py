import numpy as np
from scipy.signal import find_peaks


class Facet:

	def __init__(self, message):
		self._message = message
		self._eeg = None
        # EEGLAB EEG-Datenstruktur
		self.exclude_channels = []
        # Trigger vom MR-Gerät
		self.slice_trigger = False
		self.rel_trig_pos = 0.03
		self.auto_pre_trig = True
		self.volume_gaps = None
		self.interpolate_volume_gaps = True
		self.align_slices_reference = 1
		# Vorfilterung der EEG-Daten
		self.pre_filter_lp_frequency = None
		self.pre_filter_hp_frequency = None
		self.pre_filter_h_frequencies = None
		self.pre_filter_h_amplitudes = None
		self.pre_filter_gauss_hp_frequency = None
		# Artefaktentfernungsschritte
		self.ra_sequence = ['Cut', 'UpSample', 'AlignSlices', 'AlignSubSample',
		                  'RemoveVolumeArt', 'CalcAvgArt', 'PCA', 'DownSample',
		                  'Paste', 'LowPass', 'ANC']
		# Interpolation
		self.upsample = None
		self.upsample_cutoff = 0.5
		# Sub-Sample-Ausrichtung
		self.ssa_hp_frequency = 300
		# Mittelwertbildung von Artefakten
		self.avg_window = 30
		self.avg_matrix = None
		self.avg_matrix_updater_data = None
		# Optimaler Basissatz (OBS) von PCA
		self.obs_num_pcs = 'auto'
		self.obs_exclude_channels = []
		self.obs_hp_frequency = 70
		# Tiefpass nach Artefaktentfernung (aber vor ANC)
		self.final_lp_frequency = None
		# Adaptive Rauschunterdrückung (ANC)
		self.do_anc = False
		# Verarbeitung von Daten außerhalb der fMRT-Akquisition
		self.dont_touch_non_artifact = True
		# Ausführungsoptionen
		self.profiling = False
		# Konstanten
		self.anc_default_lp = 70
		self.th_slope = 2
		self.th_cumvar = 80
		self.th_varexp = 5
		# Nur-Lese-Eigenschaften
		self.num_channels = None
		self.num_samples = None
		self.process_channels = None
		self.sampling_frequency = None
		self.triggers = None
		self.triggers_up = None
		self.triggers_aligned = None
		self.num_triggers = None
		self.half_window = None
		self.pre_trig = None
		self.post_trig = None
		self.art_length = None
		self.acq_start = None
		self.acq_end = None
		self.search_window = None
		self.run_time = None
		self.run_cpu_time = None
		self.pre_filter_h_freqs = None
		self.pre_filter_h_ampls = None
		self.pre_filter_gauss_hp_freqs = None
		self._ra_eeg_all = None
		self._ra_eeg_acq = None
		self._ra_noise_all = None
		self._ra_noise_acq = None
		self.sub_sample_alignment = None
		self.obs_hp_filter_weights = None
		self.final_lp_filter_weights = None
		self.anc_hp_frequency = None
		self.anc_hp_filter_weights = None
		self.anc_filter_order = None

	def find_triggers(self, EventType, TriggerOffset):
		self.Triggers = self.find_triggers(self._eeg.event, EventType, TriggerOffset)
		# Check if triggers are sorted
		if self.issorted(self.Triggers):
			error('FACET:unsortedTrigs', 'Triggers are not sorted')
		# store number of triggers
		self.NumTriggers = length(self.Triggers)

	def AnalyzeData(self, ChannelDetails=False):
    # set values of optional parameters
		if ChannelDetails is None:
			ChannelDetails = False

		# Complain if EEG was not set
		if self._eeg is None:
			raise ValueError(
			    'FACET:needEEG', 'No data in EEG base structure. Did you forget to assign EEG data?')

		# print information about the EEG data
		print(f'Samples:       {self._eeg.pnts}')
		print(f'Sampling Rate: {self._eeg.srate}')
		print(
		    f'Duration:      {self._eeg.pnts/self._eeg.srate:.4f}s ({FACET.SecToString(self._eeg.pnts/self._eeg.srate)})')
		# Channels
		print(f'Channels:      {self._eeg.nbchan}')
		if ChannelDetails:
			print('Num.  Name            min         max      mean    quant.')
			tmin = []
			tmax = []
			tmean = 0
			for i in range(1, self._eeg.nbchan+1):
				cmin = np.min(self._eeg.data[i-1, :])
				tmin = np.min([tmin, cmin])
				cmax = np.max(self._eeg.data[i-1, :])
				tmax = np.max([tmax, cmax])
				cmean = np.mean(self._eeg.data[i-1, :])
				tmean = tmean+cmean
				cstep = mode(diff(unique(self._eeg.data[i-1, :])))
				print(
				    f'  {i:2d}:  {self._eeg.chanlocs[i-1].labels:6s}  {cmin:10.2f}  {cmax:10.2f}  {cmean:8.2f}    {cstep:6.4f}')
			tmean = tmean/self._eeg.nbchan
			print(f'  Total:       {tmin:10.2f}  {tmax:10.2f}  {tmean:8.2f}')
		else:
			for i in range(1, self._eeg.nbchan+1):
				if i % 10 == 1:
					print(f'  {i:2d}-{min(i+9, self._eeg.nbchan)}:', end=' ')
				print(f'{self._eeg.chanlocs[i-1].labels}', end=' ')
				if i % 10 == 0:
					print()
			if i % 10 != 0:
				print()
		# Events
		print('Events:')
		for i in np.unique([event['type'] for event in self._eeg.event]):
			n = len(FACET.FindTriggers(self._eeg.event, i))
			print(f'  {i:10s} {n:5d}x')
		amin, amax = self.FindAcquisition()
		adur = amax-amin+1
		print('Acquisition')
		print(f'  approx. begin:    {amin:7d}  ({amin/self._eeg.srate:7.2f}s)')
		print(f'  approx. end:      {amax:7d}  ({amax/self._eeg.srate:7.2f}s)')
		print(f'  approx. duration: {adur:7d}  ({adur/self._eeg.srate:7.2f}s)')

		if len(self.Triggers) == 0:
			print('No triggers setup. Use FindTriggers() and re-run this function\n  to get more information.\n')
			return

		# Analyze triggers
		TrigsDiff = np.diff(self.Triggers)   # [ T2-T1  T3-T2  T4-T3 ... ]
		# show histogram of trigger distances
		TrigsDiffHist, TrigsDiffVal = np.histogram(
		    TrigsDiff, bins=np.arange(min(TrigsDiff), max(TrigsDiff)+1))
		print('Trigger Distances Histogram')
		for i in range(len(TrigsDiffHist)):
			if TrigsDiffHist[i] != 0:
				print(f'  {TrigsDiffVal[i]:4d} ({TrigsDiffVal[i]/self._eeg.srate*1000:.2f}ms): {TrigsDiffHist[i]:5d} {"#"*int(round(60*TrigsDiffHist[i]/len(TrigsDiff)))}')
		# Check if there are big holes
		MeanTrigsDiff = np.mean(TrigsDiff)
		if TrigsDiffVal[-1] > 1.8 * MeanTrigsDiff:
			print('WARNING: Maximum triggers distance is more than 1.8x the mean distance. Some\n  triggers might be missing. Try FindMissingTriggers().\n')
		# Check if we have enough triggers
		if round(adur/MeanTrigsDiff) > self.NumTriggers:
			print(
			    f'WARNING: Comparing mean triggers difference with total acquisition duration\n  suggests that triggers are missing (have: {self.NumTriggers}, approx. required: {adur/MeanTrigsDiff:.2f})\n  Try FindMissingTriggers() and AddTriggers.\n')
		# determine whether these are volume or slice triggers
		MeanTrigsDur = MeanTrigsDiff/self.eeg.srate
		if MeanTrigsDur > 1.0:
			print(
			    f'Mean trigger distances {MeanTrigsDur:.2f}s are larger than 1.0s, assuming VOLUME triggers.\n')
			SliceTrigs = False
		else:
			print(
			    f'Mean trigger distances {MeanTrigsDur*1000:.2f}ms are less than 1.0s, assuming SLICE triggers.\n')
			SliceTrigs = True
		# determine if we have volume gaps

		if SliceTrigs:
			if TrigsDiffVal[-1] - TrigsDiffVal[0] > 3:
				HistEndSlice = int(np.floor(np.mean([1, len(TrigsDiffVal)])))
				HistBeginVol = int(np.ceil(np.mean([1, len(TrigsDiffVal)])))
				HistBeginVol = max(HistEndSlice + 1, HistBeginVol)
				SliceDist = np.sum(TrigsDiffHist[:HistEndSlice] *
				                   TrigsDiffVal[:HistEndSlice]) / np.sum(TrigsDiffHist[:HistEndSlice])
				VolDist = np.sum(TrigsDiffHist[HistBeginVol:] *
				                 TrigsDiffVal[HistBeginVol:]) / np.sum(TrigsDiffHist[HistBeginVol:])
				SliceDistCount = np.sum(TrigsDiffHist[:HistEndSlice])
				VolDistCount = np.sum(TrigsDiffHist[HistBeginVol:])
				VolCount = VolDistCount + 1
				SliceCount = SliceDistCount + VolCount
				print(f'Found {SliceDistCount} smaller (slice) distances of {SliceDist/self._eeg.srate*1000:.2f}ms and {VolDistCount} larger (volume)\n  distances of {VolDist/self._eeg.srate*1000:.2f}ms, with a volume gap of {(VolDist-SliceDist)/self._eeg.srate*1000:.2f}ms.')
				if SliceCount % VolCount == 0:
					print(
					    f'This most probably shows an fMRI acquisition of {VolCount} volumes with {SliceCount//VolCount} slices\n  each, {SliceCount} in total.')
				else:
					print(
					    f'WARNING: Total slice count {SliceCount} is not an integer multiple of the estimated\n  volume count {VolCount}.')
			else:
				print('Small variation of trigger distances. No volume gaps assumed.')

		else:
			VolumeIdx = round(len(self.Triggers) / 2)
			Volume = np.sum(
			    self.eeg.data[:, self.Triggers[VolumeIdx]:self.Triggers[VolumeIdx+1]], axis=1)
			corr = np.correlate(Volume, Volume, mode='full')[len(Volume) - 1:]
			self = corr[0]
			corr[corr < 0.8 * self] = 0
			peaks, _ = find_peaks(corr)
			dist = np.diff([1] + list(peaks))
			SliceDistAvg = np.mean(dist)
			SliceDistStd = np.std(dist)
			SlicePerVolume = MeanTrigsDiff / SliceDistAvg
			print(f'Found slice periode of {SliceDistAvg:.2f} samples (std.dev. {SliceDistStd:.2f}).\n  Slice time = {SliceDistAvg/self._eeg.srate*1000:.2f}ms, slice frequeny = {self.eeg.srate/SliceDistAvg:.2f}Hz\n  {SlicePerVolume:.2f} slices per volume')
			if SliceDistStd > 0.02 * SliceDistAvg:
				print(
				    f'WARNING: High std.dev indicates unreliable results. Distances between\n  maxima are {str(dist)}')

	def FindMissingTriggers(self, Volumes, Slices):
    # Return if we have the correct number of triggers
		if len(self.Triggers) == Volumes*Slices:
			# nothing to do
			return
		notify(self, 'EventCorrectTriggers')
		# Reconstruct missing triggers
		self.Triggers = self.CompleteTriggers(self.Triggers, Volumes, Slices)
		# store new number of triggers
		self.NumTriggers = len(self.Triggers)

	def AddTriggers(self, Where):
		# ADDTRIGGERS  Add triggers outside of defined trigger range
		#
		# Use this method if some triggers before the first or after the last
		# are missing.
		#
		# Where
		#   Sample indices into EEG.data to add to the trigger array.
		if min(Where) < 1 or max(Where) > self.FEEG.pnts:
			raise ValueError(
			    f"Triggers ({min(Where)} - {max(Where)}) out of range 1-{self.FEEG.pnts}")
		common = set(self.Triggers).intersection(Where)
		if common:
			raise ValueError(f"Trigger(s) {common} are duplicate")
		# add new trigger locations
		self.Triggers = sorted(self.Triggers + Where)
		# store number of triggers
		self.NumTriggers = len(self.Triggers)

	def GenerateSliceTriggers(self, Slices, Duration, RelPos):
		SliceTrigs = np.round(((np.arange(Slices) - RelPos)) * Duration).astype(int)
		NewTrigs = np.zeros(Slices * self.NumTriggers, dtype=int)
		for i in range(self.NumTriggers):
			NewTrigs[i*Slices:(i+1)*Slices] = self.Triggers[i] + SliceTrigs
		self.Triggers = NewTrigs
		# store number of triggers
		self.NumTriggers = len(self.Triggers)
		# change setting
		self.SliceTrigger = True

	def CheckData(self):
		# CHECKDATA  Check EEG dataset and your setup
		#
		# Check EEG dataset and your setup for any notable or problematic
		# conditions.
		#
		# Complain if EEG was not set
		if self.FEEG is None:
			raise ValueError(
			    "No data in EEG base structure. Did you forget to assign EEG data?")
		# Complain if FindTriggers was not yet called
		if self.Triggers is None:
			raise ValueError("No triggers setup. Did you forget to use FindTriggers()?")
		# TODO: check more, especially the setup

	def Finish(self):
		self.RunTime = time.perf_counter() - self.StartTime
		self.RunCpuTime = time.process_time() - self.StartCpuTime
		notify(self, 'Finished')


	def PreFilter(self):
		if not self.PreFilterHFreqs and not self.PreFilterGaussHPFreqs:
			return
		self.ProfileStart()
		notify(self, 'StartPreFilter')
		# calculate borders between artifact and non-artifact periods
		ArtStart = self.AcqStart + \
		    np.round((self.TriggersUp[0] - self.PreTrig) / self.Upsample).astype(int)
		ArtEnd = self.AcqStart + \
		    np.round((self.TriggersUp[-1] + self.PostTrig) /
		             self.Upsample).astype(int)
		End = self.FEEG.data.shape[1]
		for Channel in self.ProcessChannels:
			# before acquisition
			self.PreFilterRun(Channel, 1, ArtStart)
			# during acquisition
			self.PreFilterRunPadded(Channel, (ArtStart + 1), ArtEnd)
			# after acquisition
			self.PreFilterRun(Channel, (ArtEnd + 1), End)
		# profiling information
		self.ProfileStop()

	def PreFilterRun(self, Channel, From, To):
		data = self.FEEG.data[Channel, From:To]
		if self.PreFilterHFreqs and self.PreFilterHFreqs[Channel]:
			f = np.concatenate(self.PreFilterHFreqs[Channel])
			a = np.concatenate(self.PreFilterHAmpls[Channel])
			data = FACET.fftfilt(data, f, a)
			# TODO: undo shift
		if self.PreFilterGaussHPFreqs:
			HPf = self.PreFilterGaussHPFreqs[Channel]
			data = FACET.fftgausshp(data, HPf, self.FEEG.srate)
		self.FEEG.data[Channel, From:To] = data

	def PreFilterRunPadded(self, Channel, From, To):
		data = self.FEEG.data[Channel, From:To]
		# prepend data with enough copies of the first artifact to let the filter taps settle
		ArtLen = self.Triggers[1] - self.Triggers[0]
		ArtS = data[:ArtLen]
		Num = np.ceil(1 * self.SamplingFrequency / ArtLen).astype(int) + \
		              1   # prequel should be longer than the filter 1/f_c
		# append data for the reverse filter
		ArtLen = self.Triggers[-1] - self.Triggers[-2]
		ArtE = data[-ArtLen-1:]
		# add data
		data = np.concatenate([np.tile(ArtS, Num), data, np.tile(ArtE, Num)])
		# filter (don't use one common function for this filter stuff here,
		# because this would require a function call with parameters, for
		# which the potentially large "data" array would have to be
		# duplicated in memory)
		if self.PreFilterHFreqs and self.PreFilterHFreqs[Channel]:
			f = np.concatenate(self.PreFilterHFreqs[Channel])
			a = np.concatenate(self.PreFilterHAmpls[Channel])
			data = FACET.fftfilt(data, f, a)
			# TODO: undo shift
		if self.PreFilterGaussHPFreqs:
			HPf = self.PreFilterGaussHPFreqs[Channel]
			data = FACET.fftgausshp(data, HPf, self.FEEG.srate)
		# cut out interesting part
		data = data[(len(ArtS) * Num + 1):(len(data) - len(ArtE) * Num)]
		# store
		self.FEEG.data[Channel, From:To] = data

	def RemoveArtifacts(self):
		notify(self, 'StartRemoveArtifacts')

		# iterate over all channels
		for Channel in self.ProcessChannels:

			# profiling start
			self.ProfileStart()
			# notify
			notify(self, 'EventRAChannelStart', FACET.EventDataOneParam(Channel))

			# prepare data
			# cast EEG to double, which previously was implicitely done by
			# interp()
			self.RAEEGAll = np.double(self.FEEG.data[Channel, :])
			self.RANoiseAll = np.zeros_like(self.RAEEGAll)

			# execute RASequence
			for i, Step in enumerate(self.RASequence):
				# profiling
				self.ProfileStart()
				# function handle?
				if isinstance(Step, types.FunctionType):
					# execute function handle
					Step(self, i, Channel)
					# done profiling
					self.ProfileStop(str(Step))
					continue
				# send event to listeneners of coming processing step
				notify(self, 'EventRA' + Step, FACET.EventDataOneParam(Channel))
				# perform processing step
				if Step == 'Cut':
					self.RACut()
				elif Step == 'UpSample':
					self.RAUpSample()
				elif Step == 'AlignSlices':
					self.RAAlignSlices(Channel)
				elif Step == 'AlignSubSample':
					self.RAAlignSubSample()
				elif Step == 'RemoveVolumeArt':
					self.RARemoveVolumeArtifact()
				elif Step == 'CalcAvgArt':
					self.RACalcAvgArt(Channel)
				elif Step == 'PCA':
					self.RAPCA(Channel)
				elif Step == 'DownSample':
					self.RADownSample()
				elif Step == 'Paste':
					self.RAPaste()
				elif Step == 'LowPass':
					self.RALowPass()
				elif Step == 'ANC':
					self.RAANC()
				else:
					error('FACET:RemoveArtifact:InvalidStep', f'Invalid artifact removal step "{Step}"')
				# done profiling
				self.ProfileStop(Step)

			# store data
			if self.DontTouchNonArtifact:
				# keep data outside of acquisition untouched
				self.FEEG.data[Channel, self.AcqStart:self.AcqEnd] = self.RAEEGAll[self.AcqStart:self.AcqEnd]
			else:
				# use full data filtered by the algorithm above
				self.FEEG.data[Channel, :] = self.RAEEGAll

			# notify
			notify(self, 'EventRAChannelDone')
			# done profiling
			self.ProfileStop(f'Channel {Channel}')


	def Prepare(self):
		# Prepare: start of the algorithm
		self.DeriveVars()

	def issorted(obj):
		return obj == True
	def printEEG(self):
		print("EEG")

	def printMessage(self):
		print(self._message)

	def _get_message(self):
			return self._message
	

	message = property(fget=_get_message)
