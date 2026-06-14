"""
Value injection module for adding synthetic biological features to MEDS data.

This module generates synthetic biological proxies (epigenetic clocks, protein scores, embeddings)
and injects them into patient MEDS records as numerical values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


class SyntheticBiologicalFeatureGenerator:
    """Generate synthetic biological features for value injection."""

    def __init__(self, n_samples: int, random_state: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)
        self.features = {}

    def generate_clocks(self, n_clocks: int = 10) -> Dict[str, np.ndarray]:
        """Generate synthetic epigenetic clock predictions."""
        clocks = {}
        base_age = self.rng.uniform(20, 80, self.n_samples)

        for i in range(n_clocks):
            # Add realistic variation around base age
            noise = self.rng.normal(0, 5, self.n_samples)
            clock_bias = self.rng.uniform(-10, 10)
            clocks[f"clock_{i}"] = np.clip(base_age + noise + clock_bias, 0, 120)

        self.features["clocks"] = clocks
        return clocks

    def generate_episcore_proteins(self, n_proteins: int = 20) -> Dict[str, np.ndarray]:
        """Generate synthetic EpiScore protein predictions."""
        proteins = {}

        for i in range(n_proteins):
            # Lognormal-like distribution (proteins typically log-normal)
            log_values = self.rng.normal(0, 1.5, self.n_samples)
            proteins[f"protein_{i}"] = np.exp(log_values)

        self.features["proteins"] = proteins
        return proteins

    def generate_maple_embeddings(self, embedding_dim: int = 32) -> np.ndarray:
        """Generate synthetic MAPLE embeddings."""
        embeddings = self.rng.normal(0, 1, (self.n_samples, embedding_dim))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
        self.features["maple_embeddings"] = embeddings
        return embeddings

    def generate_methylgpt_embeddings(self, embedding_dim: int = 64) -> np.ndarray:
        """Generate synthetic MethylGPT embeddings."""
        embeddings = self.rng.normal(0, 1, (self.n_samples, embedding_dim))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
        self.features["methylgpt_embeddings"] = embeddings
        return embeddings

    def generate_all_features(self) -> Dict:
        """Generate all synthetic biological features."""
        return {
            "clocks": self.generate_clocks(10),
            "proteins": self.generate_episcore_proteins(20),
            "maple_embeddings": self.generate_maple_embeddings(32),
            "methylgpt_embeddings": self.generate_methylgpt_embeddings(64),
        }


def inject_values_into_meds(
    meds_data: pd.DataFrame,
    biological_features: Dict,
    patient_ids: List,
    feature_concepts: Dict = None,
) -> pd.DataFrame:
    """
    Inject synthetic biological features into MEDS data.

    Parameters
    ----------
    meds_data : pd.DataFrame
        MEDS data with columns: [subject_id, time, concept_code, value]
    biological_features : Dict
        Dictionary of feature arrays indexed by patient
    patient_ids : List
        List of patient IDs in the biological features
    feature_concepts : Dict
        Optional mapping of feature names to concept codes

    Returns
    -------
    pd.DataFrame
        MEDS data with injected biological features
    """
    if feature_concepts is None:
        feature_concepts = {}

    # Flatten biological features into separate rows
    injected_rows = []

    for feature_name, feature_array in biological_features.items():
        if isinstance(feature_array, dict):
            # Handle dict of features (e.g., clocks)
            for sub_feature_name, values in feature_array.items():
                concept_code = feature_concepts.get(
                    f"{feature_name}/{sub_feature_name}",
                    f"BIO_{feature_name.upper()}_{sub_feature_name.upper()}",
                )

                for patient_idx, patient_id in enumerate(patient_ids):
                    value = float(values[patient_idx])
                    injected_rows.append(
                        {
                            "subject_id": patient_id,
                            "time": 0,  # Add at beginning of sequence
                            "concept_code": concept_code,
                            "value": value,
                        }
                    )

        else:
            # Handle array features (embeddings)
            if len(feature_array.shape) == 1:
                # 1D array
                concept_code = feature_concepts.get(
                    feature_name, f"BIO_{feature_name.upper()}"
                )

                for patient_idx, patient_id in enumerate(patient_ids):
                    value = float(feature_array[patient_idx])
                    injected_rows.append(
                        {
                            "subject_id": patient_id,
                            "time": 0,
                            "concept_code": concept_code,
                            "value": value,
                        }
                    )

            else:
                # 2D array (embeddings) - create one concept per dimension
                for dim in range(feature_array.shape[1]):
                    concept_code = feature_concepts.get(
                        f"{feature_name}_dim{dim}", f"BIO_{feature_name.upper()}_DIM{dim}"
                    )

                    for patient_idx, patient_id in enumerate(patient_ids):
                        value = float(feature_array[patient_idx, dim])
                        injected_rows.append(
                            {
                                "subject_id": patient_id,
                                "time": 0,
                                "concept_code": concept_code,
                                "value": value,
                            }
                        )

    # Combine with original MEDS data
    injected_df = pd.DataFrame(injected_rows)
    return pd.concat([meds_data, injected_df], ignore_index=True)


def create_feature_concept_mapping(biological_features: Dict) -> Dict[str, str]:
    """Create mapping from feature names to concept codes."""
    mapping = {}

    for feature_name, feature_data in biological_features.items():
        if isinstance(feature_data, dict):
            for sub_feature_name in feature_data.keys():
                key = f"{feature_name}/{sub_feature_name}"
                mapping[key] = f"BIO_{feature_name.upper()}_{sub_feature_name.upper()}"
        else:
            if len(feature_data.shape) == 2:
                for dim in range(feature_data.shape[1]):
                    key = f"{feature_name}_dim{dim}"
                    mapping[key] = f"BIO_{feature_name.upper()}_DIM{dim}"
            else:
                mapping[feature_name] = f"BIO_{feature_name.upper()}"

    return mapping
