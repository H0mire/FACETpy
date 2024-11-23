from facet.modules.common.module import AbstractModule


class EventAnalysisModule(AbstractModule):
    def __init__(self):
        self._required = []
        self._facet = None

    def derive_values(self, facet):
        self._facet = facet
        self._check_volume_gaps()
        self._derive_art_length()
        self._derive_times()
        self._derive_tmin_tmax()

    def find_triggers(self, pattern):
        pass

    def find_missing_triggers(self, add_sub_periodic_artifacts=False):
        pass
