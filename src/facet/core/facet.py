from typing import Optional, Any, Dict, Type
import mne
from loguru import logger
from pathlib import Path

from .metadata import MetadataHandler
from ..io.loader import EEGLoader
from ..io.saver import EEGSaver
from .plugin import PluginManager, FACETPlugin

class FACET:
    """
    Main FACET class that coordinates EEG processing modules.
    Acts as a facade to provide a simplified interface to the underlying modules.
    """
    
    def __init__(self):
        """Initialize FACET instance and its processors."""
        self.raw: Optional[mne.io.Raw] = None
        self.raw_orig: Optional[mne.io.Raw] = None

        from ..processing.triggers import TriggerProcessor
        from ..processing.alignment import AlignmentProcessor
        from ..processing.artifacts import ArtifactProcessor
        from ..processing.filtering import FilterProcessor
        from ..analysis.evaluation import EvaluationProcessor
        from ..visualization.plots import Plotter
        
        # Initialize processors
        self._metadata: MetadataHandler = MetadataHandler()
        self.triggers: TriggerProcessor = TriggerProcessor(self)
        self.alignment: AlignmentProcessor = AlignmentProcessor(self)
        self.artifacts: ArtifactProcessor = ArtifactProcessor(self)
        self.filter: FilterProcessor = FilterProcessor(self)
        self.evaluation: EvaluationProcessor = EvaluationProcessor(self)
        self.plot: Plotter = Plotter(self)
        
        self._plugin_manager: PluginManager = PluginManager()
        self._plugins: Dict[str, FACETPlugin] = {}
        
    @property
    def data(self) -> Optional[mne.io.Raw]:
        """Get current EEG data."""
        return self.raw
        
    def load(self, path: str, **kwargs) -> 'FACET':
        """Load EEG data and return self for method chaining."""
        self.raw = EEGLoader.load(path, **kwargs)
        self.raw_orig = self.raw.copy() if kwargs.get('preload', True) else self.raw
        
        if kwargs.get('format') == 'bids':
            self._metadata.load_from_bids(self.raw, path)
            
        return self
        
    def save(self, path: str, **kwargs) -> 'FACET':
        """Save EEG data and return self for method chaining."""
        if self.raw is None:
            raise ValueError("No EEG data loaded")
            
        EEGSaver.save(self.raw, path, **kwargs)
        
        if kwargs.get('format') == 'bids':
            self._metadata.save_to_bids(self.raw, path)
            
        return self
    
    def get_metadata(self, key: str) -> Any:
        """Get metadata value."""
        if self.raw is None:
            raise ValueError("No EEG data loaded")
        return self._metadata.get_metadata(self.raw).get(key)
        
    def set_metadata(self, key: str, value: Any) -> 'FACET':
        """Set metadata value and return self for method chaining."""
        if self.raw is None:
            raise ValueError("No EEG data loaded")
        self._metadata.set_metadata(self.raw, key, value)
        return self
    
    def load_plugins(self, plugin_dir: Optional[str] = None) -> 'FACET':
        """Load plugins from directory."""
        if plugin_dir is None:
            # Load built-in plugins
            plugin_dir = Path(__file__).parent.parent / "plugins"
            
        self._plugin_manager.load_plugins(plugin_dir)
        return self
        
    def use_plugin(self, name: str, **kwargs) -> Any:
        """
        Returns Any to allow for different plugin interfaces
        while still maintaining basic plugin structure
        """
        if name not in self._plugins:
            plugin_class = self._plugin_manager.get_plugin(name)
            if plugin_class is None:
                raise ValueError(f"Plugin {name} not found")
            
            self._plugins[name] = plugin_class(self, **kwargs)
            
        return self._plugins[name] 