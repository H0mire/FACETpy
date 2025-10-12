"""
New API Demo

This example demonstrates the new pipeline-based API for FACETpy.

Author: FACETpy Team
Date: 2025-01-12
"""

# This is a demonstration of the new API
# Note: Most processors are not yet implemented, this shows the intended usage

from facet.core import Pipeline, ProcessingContext
from facet.io import EDFLoader, EDFExporter

# Example 1: Simple pipeline (only loaders/exporters implemented so far)
def example_simple():
    """Simple example showing basic pipeline usage."""
    pipeline = Pipeline([
        EDFLoader(
            path="./examples/datasets/NiazyFMRI.edf",
            bad_channels=["EKG", "EMG", "EOG", "ECG"],
            artifact_to_trigger_offset=-0.005,
            upsampling_factor=10
        ),
        # More processors will be added here as they are implemented
        # TriggerDetector(regex=r"\b1\b"),
        # HighPassFilter(freq=1.0),
        # UpSample(factor=10),
        # AASCorrection(),
        # etc...
        EDFExporter(
            path="output_demo.edf",
            overwrite=True
        )
    ])

    # Execute pipeline
    result = pipeline.run()

    if result.was_successful():
        print(f"Pipeline completed successfully in {result.execution_time:.2f}s")
        print(f"\nProcessing history:")
        for step in result.get_history():
            print(f"  - {step.name}")
    else:
        print(f"Pipeline failed: {result.error}")


# Example 2: Using the fluent Pipeline Builder (future API)
def example_fluent():
    """Example showing fluent API (not yet fully implemented)."""
    from facet.core import PipelineBuilder

    pipeline = (PipelineBuilder(name="EEG Processing")
        # .add(EDFLoader("data.edf", bad_channels=["EKG"]))
        # .add(HighPassFilter(freq=1.0))
        # .add(UpSample(factor=10))
        # .add_if(
        #     condition=True,  # Could be dynamic
        #     processor=TriggerDetector(regex=r"\btrigger\b")
        # )
        # .add(AASCorrection(window_size=30))
        # .add(EDFExporter("output.edf"))
        .build())

    # result = pipeline.run()
    print("Fluent API builder created (processors not yet implemented)")


# Example 3: Plugin system demonstration
def example_plugins():
    """Example showing custom processor plugin."""
    from facet.core import Processor, register_processor, list_processors

    @register_processor
    class CustomFilter(Processor):
        """Custom filtering processor."""

        name = "custom_filter"
        description = "My custom filter"

        def __init__(self, cutoff: float = 50.0):
            self.cutoff = cutoff
            super().__init__()

        def process(self, context: ProcessingContext) -> ProcessingContext:
            # Custom processing logic here
            print(f"Applying custom filter with cutoff={self.cutoff}Hz")
            return context

    # List all registered processors
    all_processors = list_processors()
    print("\nRegistered processors:")
    for name, proc_class in all_processors.items():
        if hasattr(proc_class, 'description'):
            print(f"  - {name}: {proc_class.description}")
        else:
            print(f"  - {name}")


# Example 4: Conditional processing
def example_conditional():
    """Example showing conditional processor execution."""
    from facet.core import ConditionalProcessor, NoOpProcessor

    # This will be used once we have more processors implemented
    # pipeline = Pipeline([
    #     EDFLoader("data.edf"),
    #     ConditionalProcessor(
    #         condition=lambda ctx: ctx.metadata.custom.get("needs_upsampling", False),
    #         processor=UpSample(factor=10),
    #         else_processor=NoOpProcessor()
    #     ),
    #     # ... more processors
    # ])
    print("Conditional processing example (processors not yet fully implemented)")


if __name__ == "__main__":
    print("=" * 60)
    print("FACETpy New API Demo")
    print("=" * 60)

    print("\n1. Simple Pipeline Example")
    print("-" * 60)
    try:
        example_simple()
    except Exception as e:
        print(f"Example failed (expected - data file may not exist): {e}")

    print("\n2. Fluent API Example")
    print("-" * 60)
    example_fluent()

    print("\n3. Plugin System Example")
    print("-" * 60)
    example_plugins()

    print("\n4. Conditional Processing Example")
    print("-" * 60)
    example_conditional()

    print("\n" + "=" * 60)
    print("Demo completed")
    print("=" * 60)
