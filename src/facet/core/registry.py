"""
Plugin Registry Module

This module provides a registry system for discovering and registering
processors as plugins.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Dict, Type, Optional, List
from loguru import logger
from .processor import Processor


class ProcessorRegistry:
    """
    Registry for processor plugins.

    This singleton class maintains a registry of all available processors,
    allowing dynamic discovery and instantiation.

    Example::

        # Register a processor
        @register_processor
        class MyProcessor(Processor):
            name = "my_processor"
            ...

        # Or manually
        registry = ProcessorRegistry.get_instance()
        registry.register("my_processor", MyProcessor)

        # Get registered processor
        processor_class = registry.get("my_processor")
        processor = processor_class(param1=value1)

        # List all processors
        all_processors = registry.list_all()
    """

    _instance: Optional['ProcessorRegistry'] = None
    _registry: Dict[str, Type[Processor]] = {}

    def __init__(self):
        """Initialize registry (use get_instance() instead)."""
        if ProcessorRegistry._instance is not None:
            raise RuntimeError("Use ProcessorRegistry.get_instance() instead")

    @classmethod
    def get_instance(cls) -> 'ProcessorRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._registry = {}
        return cls._instance

    def register(
        self,
        name: str,
        processor_class: Type[Processor],
        force: bool = False
    ) -> None:
        """
        Register a processor.

        Args:
            name: Processor name (unique identifier)
            processor_class: Processor class
            force: If True, override existing registration

        Raises:
            ValueError: If name already registered and force=False
        """
        if name in self._registry and not force:
            raise ValueError(
                f"Processor '{name}' already registered. "
                f"Use force=True to override."
            )

        if not issubclass(processor_class, Processor):
            raise TypeError(
                f"Processor class must inherit from Processor, "
                f"got {processor_class}"
            )

        self._registry[name] = processor_class
        logger.debug(f"Registered processor: {name} -> {processor_class.__name__}")

    def unregister(self, name: str) -> None:
        """
        Unregister a processor.

        Args:
            name: Processor name

        Raises:
            KeyError: If name not registered
        """
        if name not in self._registry:
            raise KeyError(f"Processor '{name}' not registered")

        del self._registry[name]
        logger.debug(f"Unregistered processor: {name}")

    def get(self, name: str) -> Type[Processor]:
        """
        Get processor class by name.

        Args:
            name: Processor name

        Returns:
            Processor class

        Raises:
            KeyError: If name not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"Processor '{name}' not registered. "
                f"Available: {self.list_names()}"
            )
        return self._registry[name]

    def has(self, name: str) -> bool:
        """Check if processor is registered."""
        return name in self._registry

    def list_names(self) -> List[str]:
        """List all registered processor names."""
        return list(self._registry.keys())

    def list_all(self) -> Dict[str, Type[Processor]]:
        """Get dictionary of all registered processors."""
        return self._registry.copy()

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._registry.clear()
        logger.debug("Cleared processor registry")

    def get_by_category(self, category: str) -> Dict[str, Type[Processor]]:
        """
        Get processors by category.

        Categories are determined by the module path.

        Args:
            category: Category name (e.g., "preprocessing", "correction")

        Returns:
            Dictionary of matching processors
        """
        matching = {}
        for name, proc_class in self._registry.items():
            module = proc_class.__module__
            if category in module:
                matching[name] = proc_class
        return matching


def register_processor(
    processor_class: Optional[Type[Processor]] = None,
    name: Optional[str] = None,
    force: bool = False
):
    """
    Decorator to register a processor.

    Can be used with or without arguments.

    Example::

        @register_processor
        class MyProcessor(Processor):
            name = "my_processor"
            ...

        @register_processor(name="custom_name")
        class MyProcessor(Processor):
            ...

    Args:
        processor_class: Processor class (when used without arguments)
        name: Custom name (overrides class.name)
        force: Force registration even if name exists

    Returns:
        Decorator function or processor class
    """
    def decorator(cls: Type[Processor]) -> Type[Processor]:
        # Determine name
        proc_name = name if name is not None else getattr(cls, 'name', cls.__name__)

        # Register
        registry = ProcessorRegistry.get_instance()
        registry.register(proc_name, cls, force=force)

        return cls

    # Handle usage without arguments
    if processor_class is not None:
        return decorator(processor_class)

    # Handle usage with arguments
    return decorator


def get_processor(name: str) -> Type[Processor]:
    """
    Get processor class by name (convenience function).

    Args:
        name: Processor name

    Returns:
        Processor class
    """
    registry = ProcessorRegistry.get_instance()
    return registry.get(name)


def list_processors(category: Optional[str] = None) -> Dict[str, Type[Processor]]:
    """
    List all registered processors (convenience function).

    Args:
        category: Optional category filter

    Returns:
        Dictionary of processors
    """
    registry = ProcessorRegistry.get_instance()
    if category:
        return registry.get_by_category(category)
    return registry.list_all()
