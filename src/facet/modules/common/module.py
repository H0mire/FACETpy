from abc import ABC, abstractmethod

class AbstractModule(ABC):

    @property
    def required(self):
        """
        Returns:
            A list of required modules.
        """
        return self._required

    @abstractmethod
    def derive_values(self, facet):
        """
        Parameters:
            facet: The facet with eeg data to analyze.

        Derives needed values from the data
        """
        pass
    
    