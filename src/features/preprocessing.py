"""
Production preprocessing workflow for the credit-risk platform.

The preprocessing pipeline always starts from the original accepted-loans CSV.
EDA artifacts and notebooks are not dependencies of the production workflow.

Responsibilities:
    1. Load the original raw CSV.
    2. Construct the observed-outcome modeling population.
    3. Select the finalized modeling features.
    4. Split the population into train/validation/test sets.
    5. Fit preprocessing transformations on training data only.
    6. Transform all three populations consistently.
    7. Save canonical processed datasets and the fitted preprocessor.

Phase 5 deliberately leaves four approved features for Feature Engineering:
    - emp_title
    - title
    - zip_code
    - earliest_cr_line
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config.paths import ProjectPaths
from src.config.settings import DataConfig
from src.data.loader import DatasetLoader
from src.data.target import TargetBuilder
from src.features.registry import (
    CATEGORICAL_FEATURES,
    EVENT_HISTORY_FEATURES,
    HIGH_CARDINALITY_FEATURES,
    STANDARD_NUMERICAL_FEATURES,
    STRUCTURAL_NUMERICAL_FEATURES,
)
from src.features.transformers import (
    CategoricalMissingHandler,
    EventHistoryImputer,
    MissingIndicatorImputer,
    StructuralMissingnessImputer,
)

JOINT_NUMERICAL_FEATURES = (
    "annual_inc_joint",
    "dti_joint",
)

JOINT_CATEGORICAL_FEATURES = ("verification_status_joint",)

DEFERRED_DATETIME_FEATURES = ("earliest_cr_line",)


def build_preprocessor() -> ColumnTransformer:
    """
    Build the Phase 5 preprocessing transformer.

    The transformer handles the 81 features finalized for Phase 5.
    High-cardinality and datetime features are intentionally excluded from
    this transformer and deferred to Feature Engineering.
    """

    standard_numerical_pipeline = Pipeline(
        steps=[
            ("imputer", MissingIndicatorImputer()),
        ]
    )

    event_history_pipeline = Pipeline(
        steps=[
            ("imputer", EventHistoryImputer()),
        ]
    )

    structural_pipeline = Pipeline(
        steps=[
            ("imputer", StructuralMissingnessImputer()),
        ]
    )

    joint_numerical_pipeline = Pipeline(
        steps=[
            ("imputer", MissingIndicatorImputer()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("missing", CategoricalMissingHandler()),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "standard_numerical",
                standard_numerical_pipeline,
                STANDARD_NUMERICAL_FEATURES,
            ),
            (
                "event_history",
                event_history_pipeline,
                EVENT_HISTORY_FEATURES,
            ),
            (
                "structural",
                structural_pipeline,
                STRUCTURAL_NUMERICAL_FEATURES,
            ),
            (
                "joint_numerical",
                joint_numerical_pipeline,
                JOINT_NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES + JOINT_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def load_modeling_data(
    input_path: Path | str = ProjectPaths.ACCEPTED_LOANS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the original accepted-loans CSV and construct the modeling data.

    The EDA parquet is deliberately not used.

    Args:
        input_path: Path to the original accepted-loans CSV.

    Returns:
        A tuple containing:
            X: Modeling features before preprocessing.
            y: Binary default target.
    """

    loader = DatasetLoader(DataConfig())
    df = loader.load(Path(input_path))

    target_builder = TargetBuilder()
    model_df = target_builder.build(df)

    feature_columns = (
        STANDARD_NUMERICAL_FEATURES
        + EVENT_HISTORY_FEATURES
        + STRUCTURAL_NUMERICAL_FEATURES
        + JOINT_NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + JOINT_CATEGORICAL_FEATURES
        + HIGH_CARDINALITY_FEATURES
        + DEFERRED_DATETIME_FEATURES
    )

    X = model_df[list(feature_columns)].copy()
    y = model_df["default"].copy()

    return X, y


def split_modeling_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Create a stratified 70/15/15 train/validation/test split.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    )


def transform_modeling_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[ColumnTransformer, object, object, object]:
    """
    Fit preprocessing on training data and transform all populations.

    Validation and test data are transformed using parameters learned from
    the training population only.
    """

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    return (
        preprocessor,
        X_train_processed,
        X_valid_processed,
        X_test_processed,
    )


def _build_processed_dataframe(
    X_processed,
    y: pd.Series,
    feature_names,
) -> pd.DataFrame:
    """
    Convert a transformed feature matrix into a labeled DataFrame.

    The binary default target is appended as the final column.
    """

    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=y.index,
    )

    processed_df["default"] = y.to_numpy()

    return processed_df


def save_preprocessing_artifacts(
    preprocessor: ColumnTransformer,
    X_train_processed,
    X_valid_processed,
    X_test_processed,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Save canonical Phase 5 datasets and the fitted preprocessor.

    Outputs:
        data/processed/train.parquet
        data/processed/validation.parquet
        data/processed/test.parquet
        models/preprocessing/preprocessor.joblib
    """

    ProjectPaths.PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    ProjectPaths.PREPROCESSING_MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_names = preprocessor.get_feature_names_out()

    train_df = _build_processed_dataframe(
        X_train_processed,
        y_train,
        feature_names,
    )

    validation_df = _build_processed_dataframe(
        X_valid_processed,
        y_valid,
        feature_names,
    )

    test_df = _build_processed_dataframe(
        X_test_processed,
        y_test,
        feature_names,
    )

    train_df.to_parquet(
        ProjectPaths.TRAIN_PROCESSED,
        index=False,
    )

    validation_df.to_parquet(
        ProjectPaths.VALIDATION_PROCESSED,
        index=False,
    )

    test_df.to_parquet(
        ProjectPaths.TEST_PROCESSED,
        index=False,
    )

    joblib.dump(
        preprocessor,
        ProjectPaths.PREPROCESSOR,
    )


def prepare_preprocessed_data(
    input_path: Path | str = ProjectPaths.ACCEPTED_LOANS,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> dict[str, object]:
    """
    Execute the complete Phase 5 preprocessing workflow.

    The workflow always begins with the original raw CSV and does not depend
    on the EDA notebook or EDA parquet artifact.

    Args:
        input_path: Path to the original accepted-loans CSV.
        random_state: Random seed for reproducible splitting.
        save_artifacts: Whether to persist the processed datasets and fitted
            preprocessor.

    Returns:
        Dictionary containing the fitted preprocessor, processed feature
        matrices, and target vectors.
    """

    X, y = load_modeling_data(input_path)

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = split_modeling_data(
        X,
        y,
        random_state=random_state,
    )

    (
        preprocessor,
        X_train_processed,
        X_valid_processed,
        X_test_processed,
    ) = transform_modeling_data(
        X_train,
        X_valid,
        X_test,
    )

    if save_artifacts:
        save_preprocessing_artifacts(
            preprocessor=preprocessor,
            X_train_processed=X_train_processed,
            X_valid_processed=X_valid_processed,
            X_test_processed=X_test_processed,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
        )

    return {
        "preprocessor": preprocessor,
        "X_train": X_train_processed,
        "X_valid": X_valid_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }
