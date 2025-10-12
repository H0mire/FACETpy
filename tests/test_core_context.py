"""
Tests for ProcessingContext and ProcessingMetadata.
"""

import pytest
import numpy as np
import mne

from facet.core import ProcessingContext, ProcessingMetadata


@pytest.mark.unit
class TestProcessingMetadata:
    """Tests for ProcessingMetadata class."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = ProcessingMetadata()

        assert metadata.triggers is None
        assert metadata.artifact_length is None
        assert metadata.upsampling_factor == 10
        assert metadata.artifact_to_trigger_offset == 0.0
        assert metadata.volume_gaps is False
        assert isinstance(metadata.custom, dict)

    def test_copy(self):
        """Test metadata copying."""
        metadata = ProcessingMetadata()
        metadata.triggers = np.array([100, 200, 300])
        metadata.artifact_length = 50
        metadata.custom['test'] = 'value'

        # Copy
        metadata_copy = metadata.copy()

        # Check it's a different object
        assert metadata_copy is not metadata

        # Check values are equal
        assert np.array_equal(metadata_copy.triggers, metadata.triggers)
        assert metadata_copy.artifact_length == metadata.artifact_length
        assert metadata_copy.custom['test'] == 'value'

        # Modify copy shouldn't affect original
        metadata_copy.artifact_length = 100
        assert metadata.artifact_length == 50

    def test_to_dict(self):
        """Test serialization to dict."""
        metadata = ProcessingMetadata()
        metadata.triggers = np.array([100, 200])
        metadata.artifact_length = 50

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert 'triggers' in data
        assert 'artifact_length' in data
        assert data['artifact_length'] == 50

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            'triggers': [100, 200, 300],
            'artifact_length': 50,
            'upsampling_factor': 5,
            'custom': {'key': 'value'}
        }

        metadata = ProcessingMetadata.from_dict(data)

        assert np.array_equal(metadata.triggers, np.array([100, 200, 300]))
        assert metadata.artifact_length == 50
        assert metadata.upsampling_factor == 5
        assert metadata.custom['key'] == 'value'


@pytest.mark.unit
class TestProcessingContext:
    """Tests for ProcessingContext class."""

    def test_initialization(self, sample_raw):
        """Test context initialization."""
        context = ProcessingContext(raw=sample_raw)

        assert context.get_raw() is sample_raw
        assert context.get_raw_original() is sample_raw
        assert context.metadata is not None

    def test_initialization_with_metadata(self, sample_raw):
        """Test context initialization with metadata."""
        metadata = ProcessingMetadata()
        metadata.triggers = np.array([100, 200])

        context = ProcessingContext(raw=sample_raw, metadata=metadata)

        assert context.metadata is metadata
        assert np.array_equal(context.get_triggers(), np.array([100, 200]))

    def test_with_raw(self, sample_raw):
        """Test creating new context with different raw."""
        context1 = ProcessingContext(raw=sample_raw)

        # Create modified raw
        raw2 = sample_raw.copy()
        raw2._data *= 2

        # Create new context
        context2 = context1.with_raw(raw2)

        # Check context1 unchanged
        assert context1.get_raw() is sample_raw

        # Check context2 has new raw
        assert context2.get_raw() is raw2
        assert not np.array_equal(
            context1.get_raw()._data,
            context2.get_raw()._data
        )

    def test_with_metadata(self, sample_context):
        """Test creating new context with different metadata."""
        new_metadata = sample_context.metadata.copy()
        new_metadata.artifact_length = 100

        context2 = sample_context.with_metadata(new_metadata)

        # Check original unchanged
        assert sample_context.metadata.artifact_length != 100

        # Check new context has new metadata
        assert context2.metadata.artifact_length == 100

    def test_get_triggers(self, sample_context):
        """Test getting triggers."""
        triggers = sample_context.get_triggers()
        assert triggers is not None
        assert len(triggers) > 0

    def test_has_triggers(self, sample_context):
        """Test checking for triggers."""
        assert sample_context.has_triggers()

        # Context without triggers
        context_no_triggers = ProcessingContext(raw=sample_context.get_raw())
        assert not context_no_triggers.has_triggers()

    def test_get_artifact_length(self, sample_context):
        """Test getting artifact length."""
        length = sample_context.get_artifact_length()
        assert length is not None
        assert length > 0

    def test_estimated_noise(self, sample_context):
        """Test estimated noise handling."""
        # Initially no noise
        assert not sample_context.has_estimated_noise()

        # Set noise
        noise = np.random.randn(*sample_context.get_raw()._data.shape)
        sample_context.set_estimated_noise(noise)

        # Check it's stored
        assert sample_context.has_estimated_noise()
        retrieved_noise = sample_context.get_estimated_noise()
        assert np.array_equal(retrieved_noise, noise)

    def test_processing_history(self, sample_context):
        """Test processing history tracking."""
        # Initially empty
        history = sample_context.get_history()
        assert len(history) == 0

        # Add entry
        sample_context.add_history_entry(
            processor_name="test_processor",
            parameters={'param1': 'value1'}
        )

        # Check it's recorded
        history = sample_context.get_history()
        assert len(history) == 1
        assert history[0]['processor'] == "test_processor"
        assert history[0]['parameters'] == {'param1': 'value1'}
        assert 'timestamp' in history[0]

    def test_to_dict(self, sample_context):
        """Test serialization to dict."""
        data = sample_context.to_dict()

        assert isinstance(data, dict)
        assert 'raw' in data
        assert 'metadata' in data
        assert 'history' in data

    def test_from_dict(self, sample_context):
        """Test deserialization from dict."""
        # Serialize
        data = sample_context.to_dict()

        # Deserialize
        context2 = ProcessingContext.from_dict(data)

        # Check equality
        assert context2.get_raw().info['sfreq'] == sample_context.get_raw().info['sfreq']
        assert len(context2.get_history()) == len(sample_context.get_history())

    def test_copy_preserves_original_raw(self, sample_raw):
        """Test that with_raw preserves raw_original."""
        context1 = ProcessingContext(raw=sample_raw)

        # Modify raw
        raw2 = sample_raw.copy()
        raw2._data *= 2

        context2 = context1.with_raw(raw2)

        # Check original is preserved
        assert context2.get_raw_original() is sample_raw
        assert context2.get_raw() is raw2

    def test_immutability(self, sample_context):
        """Test that context is effectively immutable."""
        original_raw = sample_context.get_raw()
        original_triggers = sample_context.get_triggers().copy()

        # Create new context with modifications
        new_raw = original_raw.copy()
        new_raw._data *= 2
        context2 = sample_context.with_raw(new_raw)

        # Original should be unchanged
        assert sample_context.get_raw() is original_raw
        assert np.array_equal(sample_context.get_triggers(), original_triggers)
