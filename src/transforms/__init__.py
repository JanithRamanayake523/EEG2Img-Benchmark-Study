"""
Time-series to image transformation modules.

This package provides various transformations for converting EEG time-series
to 2D images suitable for computer vision models.

Available transformations:
- GAF: Gramian Angular Field (GASF/GADF)
- MTF: Markov Transition Field
- Recurrence Plot: Phase space recurrence analysis
- Spectrogram: STFT time-frequency representation
- Scalogram: CWT wavelet time-frequency representation
- Topographic: Spatial electrode mapping (SSFI)
"""

from .gaf import GAFTransformer
from .mtf import MTFTransformer
from .recurrence import RecurrencePlotTransformer
from .spectrogram import SpectrogramTransformer
from .scalogram import ScalogramTransformer
from .topographic import TopographicTransformer

# Transform registry for easy instantiation
TRANSFORM_REGISTRY = {
    # GAF variants
    'gaf_summation': lambda **kwargs: GAFTransformer(method='summation', **kwargs),
    'gaf_difference': lambda **kwargs: GAFTransformer(method='difference', **kwargs),
    'gasf': lambda **kwargs: GAFTransformer(method='summation', **kwargs),
    'gadf': lambda **kwargs: GAFTransformer(method='difference', **kwargs),

    # MTF with different quantization levels
    'mtf_q8': lambda **kwargs: MTFTransformer(n_bins=8, **kwargs),
    'mtf_q16': lambda **kwargs: MTFTransformer(n_bins=16, **kwargs),
    'mtf': lambda **kwargs: MTFTransformer(**kwargs),

    # Recurrence Plot
    'recurrence': lambda **kwargs: RecurrencePlotTransformer(**kwargs),
    'rp': lambda **kwargs: RecurrencePlotTransformer(**kwargs),

    # Spectrogram
    'spectrogram': lambda **kwargs: SpectrogramTransformer(**kwargs),
    'stft': lambda **kwargs: SpectrogramTransformer(**kwargs),

    # Scalogram with different wavelets
    'scalogram': lambda **kwargs: ScalogramTransformer(**kwargs),
    'cwt': lambda **kwargs: ScalogramTransformer(**kwargs),
    'cwt_morlet': lambda **kwargs: ScalogramTransformer(wavelet='morl', **kwargs),
    'cwt_mexh': lambda **kwargs: ScalogramTransformer(wavelet='mexh', **kwargs),

    # Topographic
    'topographic': lambda **kwargs: TopographicTransformer(**kwargs),
    'ssfi': lambda **kwargs: TopographicTransformer(**kwargs),
    'topo': lambda **kwargs: TopographicTransformer(**kwargs),
}


def get_transformer(name: str, **kwargs):
    """
    Get transformer by name from registry.

    Args:
        name: Transformer name (e.g., 'gaf_summation', 'mtf_q8', 'spectrogram')
        **kwargs: Additional arguments to pass to transformer constructor

    Returns:
        Transformer instance

    Example:
        >>> transformer = get_transformer('gaf_summation', image_size=128)
        >>> # Or with custom parameters
        >>> transformer = get_transformer('mtf', image_size=64, n_bins=16)

    Raises:
        ValueError: If transformer name not found in registry
    """
    if name not in TRANSFORM_REGISTRY:
        available = ', '.join(sorted(TRANSFORM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown transformer: {name}. "
            f"Available transformers: {available}"
        )

    return TRANSFORM_REGISTRY[name](**kwargs)


def list_transformers():
    """
    List all available transformers.

    Returns:
        List of transformer names

    Example:
        >>> transformers = list_transformers()
        >>> print(transformers)
    """
    return sorted(TRANSFORM_REGISTRY.keys())


__all__ = [
    'GAFTransformer',
    'MTFTransformer',
    'RecurrencePlotTransformer',
    'SpectrogramTransformer',
    'ScalogramTransformer',
    'TopographicTransformer',
    'TRANSFORM_REGISTRY',
    'get_transformer',
    'list_transformers',
]
