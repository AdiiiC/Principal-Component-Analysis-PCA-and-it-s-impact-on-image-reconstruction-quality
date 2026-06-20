"""
PCA Image Reconstruction - Experiments Module
==============================================
All experiment implementations: component sweep, denoising, face recognition,
anomaly detection, compression comparison, information theory, cross-dataset.
"""

from .component_sweep import run_component_sweep
from .denoising import run_denoising_experiment
from .face_recognition import run_face_recognition_experiment
from .anomaly_detection import run_anomaly_detection
from .compression_comparison import run_compression_comparison
from .information_theory import run_information_theory_analysis
from .cross_dataset import run_cross_dataset_test
from .method_comparison import run_method_comparison
from .incremental_streaming import run_incremental_experiment
from .face_arithmetic import run_face_arithmetic

__all__ = [
    "run_component_sweep",
    "run_denoising_experiment",
    "run_face_recognition_experiment",
    "run_anomaly_detection",
    "run_compression_comparison",
    "run_information_theory_analysis",
    "run_cross_dataset_test",
    "run_method_comparison",
    "run_incremental_experiment",
    "run_face_arithmetic",
]
