from typing import Dict, Type, Any
import importlib
import inspect
from pathlib import Path
from abc import ABC, abstractmethod

class FACETPlugin(ABC):
    """Base class for all FACET plugins."""
    
    def __init__(self, facet):
        self._facet = facet
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
        
    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"
        
    @property
    def description(self) -> str:
        """Plugin description."""
        return ""

class PluginManager:
    """Manages FACET plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, Type[FACETPlugin]] = {}
        
    def register_plugin(self, plugin_class: Type[FACETPlugin]) -> None:
        """Register a plugin class."""
        if not issubclass(plugin_class, FACETPlugin):
            raise TypeError("Plugin must inherit from FACETPlugin")
            
        plugin_name = plugin_class.name.lower()
        self._plugins[plugin_name] = plugin_class
        
    def load_plugins(self, plugin_dir: str) -> None:
        """Load plugins from directory."""
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return
            
        for file in plugin_path.glob("*.py"):
            if file.name.startswith("_"):
                continue
                
            module_name = f"facet.plugins.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                
                # Find plugin classes in module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, FACETPlugin) and 
                        obj != FACETPlugin):
                        self.register_plugin(obj)
                        
            except Exception as e:
                logger.warning(f"Failed to load plugin {file}: {e}")
                
    def get_plugin(self, name: str) -> Type[FACETPlugin]:
        """Get plugin class by name."""
        return self._plugins.get(name.lower()) 