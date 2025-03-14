"""Main module for data preprocessing operations."""

import datetime
import os
import traceback

from scipy import constants

from .categorical.encoders import (
    encode_condrank_binary,
    encode_condrank_multi,
    encode_condrank_regression,
    encode_rank_binary,
    encode_rank_multi,
    encode_rank_regression,
)
from .categorical.gpu_encoders import (
    encode_condrank_binary_cudf,
    encode_condrank_multi_cudf,
    encode_condrank_regression_cudf,
    encode_rank_binary_cudf,
    encode_rank_multi_cudf,
    encode_rank_regression_cudf,
)

# from .utils.categorical_mapper import *

from .dask_utils.dask_tools import get_dask_client
from .imputation.gpu_imputer import perform_gpu_imputation
from .imputation.imputer import perform_imputation
from .utils.io import read_data_file, read_header_file, save_processed_data, save_header_file
from .utils.logger import get_logger
from .utils.validators import is_cudf_df, validate_dataframe, validate_header
from .utils import constants

logger = get_logger(__name__)


def get_encoding_meta(df, dtype_dict):
    """
    Create metadata for Dask to understand the output type of categorical encoding.

    Args:
        df: Input DataFrame
        dtype_dict: Dictionary of data types

    Returns:
        DataFrame: Empty DataFrame with correct structure for metadata
    """
    import pandas as pd
    # Create an empty DataFrame with same columns and dtypes
    meta_df = pd.DataFrame({col: pd.Series(dtype=dtype)
                           for col, dtype in dtype_dict.items()})
    # Ensure all columns from original DataFrame are included
    for col in df.columns:
        if col not in meta_df.columns:
            meta_df[col] = pd.Series(dtype='float64')
    return meta_df


def process_data(
    data_file=None,
    header_file=None,
    data_format="csv",
    header_format="json",
    output_dir=None,
    output_prefix="processed",
    model_type="classification",
    categorical_encoding_method="rank",
    prior_for_conditional_rank="event_rate",
    sample_size_for_conditional_rank=30,
    use_gpu=False,
    is_distributed=False,
    df=None,
    header=None,
    output_format="parquet",
    workers_to_use=None,
    worker_memory=None,
    worker_cores=None,
):
    """Process data by applying missing value imputation and categorical encoding."""

    # Setup logger
    logger.info("Starting data preprocessing")

    # Initialize Dask client if distributed processing is requested
    dask_client = None
    if is_distributed:
        try:
            logger.info("Setting up distributed processing with Dask")
            logger.info(f"Gpu based execution : {use_gpu}")
            logger.info(f"============ Dask configuration =================")
            logger.info(f"")
            logger.info(f"workers_to_use : {workers_to_use}")
            logger.info(f"worker_memory = {worker_memory}")
            logger.info(f"worker_cores = {worker_cores}")

            # Set default values for GPU distributed processing if not provided
            if use_gpu:
                workers_to_use = workers_to_use or 2
                worker_memory = worker_memory or "4GiB"
                worker_cores = worker_cores or 1

            dask_client = get_dask_client("preprocessing", logger,
                                          n_workers=workers_to_use,
                                          worker_memory=worker_memory,
                                          worker_cores=worker_cores)

            logger.info(f"Using Dask client: {dask_client.client}")
        except Exception as e:
            logger.warning(f"Failed to initialize Dask client: {str(e)}")
            logger.warning("Falling back to non-distributed processing")
            is_distributed = False

    # Validate required files
    if df is None and data_file is None:
        raise ValueError("Either df or data_file must be provided")

    if header is None and header_file is None:
        raise ValueError("Either header or header_file must be provided")

    # Log info about data source
    if data_file:
        logger.info(f"Reading data from {data_file}")

    # Load header if not provided
    dtype_dict = None
    if header is None:
        logger.info(f"Reading header from {header_file}")
        header, dtype_dict = read_header_file(
            header_file, header_format=header_format)

        # Validate header
        header = validate_header(header)

    # Read data with proper header
    if df is None:
        df = read_data_file(
            data_file=data_file,
            header=header,
            data_format=data_format,
            use_gpu=use_gpu,
            use_dask=is_distributed,
            dtype_dict=dtype_dict
        )

    # Validate dataframe
    validate_dataframe(df)

    # Handle GPU processing
    is_gpu_df = is_cudf_df(df)
    use_gpu = use_gpu or is_gpu_df

    if use_gpu and not is_gpu_df and not is_distributed:
        # Only convert to cuDF if not using Dask (for Dask we use dask_cudf)
        try:
            import cudf
            logger.info("Converting pandas DataFrame to cuDF DataFrame")
            df = cudf.DataFrame.from_pandas(df)
            is_gpu_df = True
        except ImportError:
            logger.warning(
                "cuDF not available, falling back to CPU processing")
            use_gpu = False

    # Perform missing value imputation
    logger.info("Performing missing value imputation")
    if is_distributed:
        if use_gpu:
            try:
                # Import dask_cudf for GPU-accelerated distributed processing
                import dask_cudf  # noqa: F401 - imported but unused

                # Implement distributed GPU imputation if needed
                logger.warning(
                    "Distributed GPU imputation not fully implemented, using CPU version")
                df = df.map_partitions(lambda part: perform_gpu_imputation(
                    part, header, logger) if hasattr(part, 'copy') else part)
            except ImportError:
                logger.warning("dask_cudf not available, using CPU imputation")
                df = df.map_partitions(lambda part: perform_imputation(
                    part, header, logger) if hasattr(part, 'copy') else part)
        else:
            # Distributed CPU imputation
            df = df.map_partitions(lambda part: perform_imputation(
                part, header, logger) if hasattr(part, 'copy') else part)
    else:
        # Non-distributed imputation
        if use_gpu:
            df = perform_gpu_imputation(df, header, logger)
        else:
            df = perform_imputation(df, header, logger)

    # Find label column
    label_col_idx = header[constants.FEATURE_TYPES_KEY].index(
        constants.LABEL_FEATURE_TYPE) if constants.LABEL_FEATURE_TYPE in header[constants.FEATURE_TYPES_KEY] else None
    if label_col_idx is None:
        logger.warning(
            "No label column (L) found in header, assuming regression")
        model_type = constants.REGRESSION

    # Determine number of classes for classification
    # Default for regression and binary classification
    num_class = constants.CLASSIFICATION_NUM_CLASSES_1
    if model_type == constants.CLASSIFICATION and label_col_idx is not None:
        label_col = header[constants.FEATURE_NAMES_KEY][label_col_idx]
        if label_col in df.columns:
            unique_values = []  # Initialize to avoid unbound variable error
            if is_distributed:
                # For Dask DataFrames, we need to compute
                try:
                    # Try to get a sample instead of computing on the entire dataset
                    sample_df = df.compute().head(constants.DEFAULT_SAMPLE_SIZE) if hasattr(
                        df, 'compute') else df.head(constants.DEFAULT_SAMPLE_SIZE)
                    unique_values = sample_df[label_col].unique()
                    num_class = len(unique_values)
                except Exception as e:
                    logger.warning(f"Error computing unique values: {str(e)}")
                    # Default to binary classification if we can't determine
                    num_class = constants.CLASSIFICATION_NUM_CLASSES_2

                    # Alternative approach if head() fails
                    try:
                        # Just assume binary classification for distributed mode
                        logger.info(
                            "Assuming binary classification for distributed processing")
                        num_class = constants.CLASSIFICATION_NUM_CLASSES_2
                    except Exception as e2:
                        logger.warning(
                            f"Alternative approach also failed: {str(e2)}")
            else:
                unique_values = df[label_col].unique()

            if is_gpu_df and not is_distributed:
                unique_values = unique_values.to_arrow().to_pylist()

            if not is_distributed or len(unique_values) > 0:
                num_class = len(unique_values)

    logger.info(f"Detected {num_class} classes for {model_type} task")

    # Perform categorical encoding
    logger.info(
        f"Performing categorical encoding using {categorical_encoding_method} method")

    # Define a helper function for distributed categorical encoding
    def apply_categorical_encoding(partition_df, header, method, model_type, num_class,
                                   use_gpu, logger, prior=None, sample=None):
        if method == constants.RANK_ENCODING:
            if model_type == constants.CLASSIFICATION:
                if num_class > constants.CLASSIFICATION_NUM_CLASSES_2:
                    if use_gpu:
                        logger.info("Using GPU for rank multi encoding")
                        part_df, part_fmap = encode_rank_multi_cudf(partition_df, header, logger)  # noqa
                    else:
                        logger.info("Using CPU for rank multi encoding")
                        part_df, part_fmap = encode_rank_multi(partition_df, header, logger)  # noqa
                else:
                    if use_gpu:
                        logger.info("Using GPU for rank binary encoding")
                        part_df, part_fmap = encode_rank_binary_cudf(partition_df, header, logger)  # noqa
                    else:
                        logger.info("Using CPU for rank binary encoding")
                        part_df, part_fmap = encode_rank_binary(partition_df, header, logger)  # noqa
            else:  # regression
                if use_gpu:
                    logger.info("Using GPU for rank regression encoding")
                    part_df, part_fmap = encode_rank_regression_cudf(partition_df, header, logger)  # noqa
                else:
                    logger.info("Using CPU for rank regression encoding")
                    part_df, part_fmap = encode_rank_regression(partition_df, header, logger)  # noqa
        elif method == constants.CONDITIONAL_RANK_ENCODING:
            if model_type == constants.CLASSIFICATION:
                if num_class > constants.CLASSIFICATION_NUM_CLASSES_2:
                    if use_gpu:
                        logger.info("Using GPU for cond rank multi encoding")
                        part_df, part_fmap = encode_condrank_multi_cudf(
                            partition_df, header, logger, prior=prior, sample=sample
                        )
                    else:
                        logger.info("Using CPU for cond rank multi encoding")
                        part_df, part_fmap = encode_condrank_multi(
                            partition_df, header, logger, prior=prior, sample=sample
                        )
                else:
                    if use_gpu:
                        logger.info("Using GPU for cond rank binary encoding")
                        part_df, part_fmap = encode_condrank_binary_cudf(
                            partition_df, header, logger, prior=prior, sample=sample
                        )
                    else:
                        logger.info("Using CPU for cond rank binary encoding")
                        part_df, part_fmap = encode_condrank_binary(
                            partition_df, header, logger, prior=prior, sample=sample
                        )
            else:  # regression
                if use_gpu:
                    logger.info("Using GPU for cond rank regression encoding")
                    part_df, part_fmap = encode_condrank_regression_cudf(
                        partition_df, header, logger, prior=prior, sample=sample
                    )
                else:
                    logger.info("Using CPU for cond rank regression encoding")
                    part_df, part_fmap = encode_condrank_regression(
                        partition_df, header, logger, prior=prior, sample=sample
                    )
        return part_df

    # Apply categorical encoding based on distribution mode
    if is_distributed:
        # For distributed processing, apply encoding to each partition
        # Note: This approach doesn't properly combine feature maps across partitions
        # A more complex approach would be needed for a fully distributed solution
        meta_df = get_encoding_meta(df, dtype_dict if dtype_dict else {})

        df = df.map_partitions(
            lambda part: apply_categorical_encoding(
                part, header, categorical_encoding_method, model_type,
                num_class, use_gpu, logger,
                prior=prior_for_conditional_rank,
                sample=sample_size_for_conditional_rank
            ),
            meta=meta_df
        )

        # For simplicity in this implementation, we'll compute the fmap on a sample
        # In a production implementation, you'd want to aggregate fmaps across partitions
        try:
            # Use a fixed sample size instead of computing shape
            sample_size = constants.DEFAULT_SAMPLE_FIXED_SIZE_10000
            # Use compute_sample instead of head() to avoid tuple attribute error
            sample_df = df.compute().head(sample_size) if hasattr(
                df, 'compute') else df.head(sample_size)
        except Exception as e:
            logger.warning(f"Error getting sample: {str(e)}")
            # Create a small fixed sample
            sample_size = constants.DEFAULT_SAMPLE_FIXED_SIZE_100
            try:
                # Try alternative approach for getting sample
                sample_df = df.compute().head(sample_size) if hasattr(
                    df, 'compute') else df.head(sample_size)
            except Exception as e2:
                logger.warning(
                    f"Could not get sample, using default encoding: {str(e2)}")
                # Return with default encoding
                return df, [], header.copy()

        # Generate fmap from sample
        if categorical_encoding_method == constants.RANK_ENCODING:
            if model_type == constants.CLASSIFICATION:
                if num_class > constants.CLASSIFICATION_NUM_CLASSES_2:
                    if use_gpu:
                        _, fmap = encode_rank_multi_cudf(
                            sample_df, header, logger)
                    else:
                        _, fmap = encode_rank_multi(sample_df, header, logger)
                else:
                    if use_gpu:
                        _, fmap = encode_rank_binary_cudf(
                            sample_df, header, logger)
                    else:
                        _, fmap = encode_rank_binary(sample_df, header, logger)
            else:  # regression
                if use_gpu:
                    _, fmap = encode_rank_regression_cudf(
                        sample_df, header, logger)
                else:
                    _, fmap = encode_rank_regression(sample_df, header, logger)
        else:  # cond_rank
            if model_type == constants.CLASSIFICATION:
                if num_class > constants.CLASSIFICATION_NUM_CLASSES_2:
                    if use_gpu:
                        _, fmap = encode_condrank_multi_cudf(
                            sample_df, header, logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
                    else:
                        _, fmap = encode_condrank_multi(
                            sample_df, header, logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
                else:
                    if use_gpu:
                        _, fmap = encode_condrank_binary_cudf(
                            sample_df, header, logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
                    else:
                        _, fmap = encode_condrank_binary(
                            sample_df, header, logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
            else:  # regression
                if use_gpu:
                    _, fmap = encode_condrank_regression_cudf(
                        sample_df, header, logger,
                        prior=prior_for_conditional_rank,
                        sample=sample_size_for_conditional_rank
                    )
                else:
                    _, fmap = encode_condrank_regression(
                        sample_df, header, logger,
                        prior=prior_for_conditional_rank,
                        sample=sample_size_for_conditional_rank
                    )
    else:
        # Non-distributed encoding - use the existing code
        if categorical_encoding_method == constants.RANK_ENCODING:
            if model_type == constants.CLASSIFICATION:
                if num_class > constants.CLASSIFICATION_NUM_CLASSES_2:
                    if use_gpu:
                        df, fmap = encode_rank_multi_cudf(df, header, logger)
                    else:
                        df, fmap = encode_rank_multi(df, header, logger)
                else:
                    if use_gpu:
                        df, fmap = encode_rank_binary_cudf(df, header, logger)
                    else:
                        df, fmap = encode_rank_binary(df, header, logger)
            else:  # regression
                if use_gpu:
                    df, fmap = encode_rank_regression_cudf(df, header, logger)
                else:
                    df, fmap = encode_rank_regression(df, header, logger)

        elif categorical_encoding_method == constants.CONDITIONAL_RANK_ENCODING:
            if model_type == constants.CLASSIFICATION:
                if num_class > constants.CLASSIFICATION_NUM_CLASSES_2:
                    if use_gpu:
                        df, fmap = encode_condrank_multi_cudf(
                            df,
                            header,
                            logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
                    else:
                        df, fmap = encode_condrank_multi(
                            df,
                            header,
                            logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
                else:
                    if use_gpu:
                        df, fmap = encode_condrank_binary_cudf(
                            df,
                            header,
                            logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
                    else:
                        df, fmap = encode_condrank_binary(
                            df,
                            header,
                            logger,
                            prior=prior_for_conditional_rank,
                            sample=sample_size_for_conditional_rank
                        )
            else:  # regression
                if use_gpu:
                    df, fmap = encode_condrank_regression_cudf(
                        df,
                        header,
                        logger,
                        prior=prior_for_conditional_rank,
                        sample=sample_size_for_conditional_rank
                    )
                else:
                    df, fmap = encode_condrank_regression(
                        df,
                        header,
                        logger,
                        prior=prior_for_conditional_rank,
                        sample=sample_size_for_conditional_rank
                    )
        else:
            raise ValueError(
                f"Unsupported categorical encoding method: {categorical_encoding_method}")

    # Update header with feature map information
    processed_header = header.copy()
    processed_header[constants.FEATURE_MAP_KEY] = fmap

    # Save outputs if output_dir is provided
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Save processed data
    #     output_data_file = os.path.join(
    #         output_dir, f"{output_prefix}_data.{output_format}")
    #     logger.info(f"Saving processed data to {output_data_file}")

    #     if is_distributed:
    #         # For Dask DataFrames
    #         if output_format.lower() == constants.CSV_FORMAT:
    #             df.to_csv(output_data_file, single_file=True, index=False)
    #         elif output_format.lower() == constants.PARQUET_FORMAT:
    #             df.to_parquet(output_data_file, write_index=False)
    #     else:
    #         # Convert to pandas if using GPU
    #         if is_gpu_df:
    #             save_df = df.to_pandas()
    #         else:
    #             save_df = df

    #         save_processed_data(save_df, output_data_file,
    #                             data_format=output_format)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save processed data
        output_data_file = os.path.join(
            output_dir, f"{output_prefix}_data.{output_format}")
        logger.info(f"Saving processed data to {output_data_file}")

        if is_distributed:
            # For Dask DataFrames
            if output_format.lower() == constants.CSV_FORMAT:
                # Just use direct to_csv as it's less problematic
                print(df._meta)
                print(df.dtypes)
                df.to_csv(output_data_file, single_file=True, index=False)
            elif output_format.lower() == constants.PARQUET_FORMAT:
                # Match the approach used in DaskDataManager.save_transformed_data
                try:
                    print(df._meta)
                    print(df.dtypes)
                    import pandas as pd
                    # dd_copy = df.copy()
                    # dd_copy = df.mixed.apply(lambda elt: str(
                    #     int(elt)) if isinstance(elt, float) else str(elt))
                    # schema = get_problematic_columns(df)
                    # df = df.map_partitions(
                    #     lambda part: part.astype({col: constants.STRING_TYPE for col in part.select_dtypes(
                    #         constants.OBJECT_TYPE).columns})
                    # )

                    # import pyarrow as pa
                    # fields = []
                    # for col in df.columns:
                    #     if df[col].dtype == constants.OBJECT_TYPE:
                    #         fields.append((col, pa.string()))
                    #     else:
                    #         fields.append(
                    #             (col, pa.from_numpy_dtype(df[col].dtype)))

                    # schema = pa.schema(fields)

                    # print(df._meta)
                    # df.to_parquet(output_data_file, schema=schema)
                    save_dask_df_to_parquet(
                        df, output_data_file, header=header)
                    # save_df_to_parquet(df, output_data_file)

                except Exception as e:
                    # If parquet still fails, use csv as fallback
                    logger.warning(
                        f"Failed to save as parquet: {str(e)} {traceback.format_exc()}")
                    csv_path = os.path.join(
                        f"{output_dir}/{output_prefix}_", "data.csv")
                    logger.info(f"Falling back to CSV: {csv_path}")
                    df.to_csv(csv_path, single_file=True, index=False)

                    # Update the expected path in the output_data_file to match what we saved
                    output_data_file = csv_path
        else:
            # Convert to pandas if using GPU
            if is_gpu_df:
                save_df = df.to_pandas()
            else:
                save_df = df

            save_processed_data(save_df, output_data_file,
                                data_format=output_format)

        # Save processed header
        output_header_file = os.path.join(
            output_dir, f"{output_prefix}_header.json")
        logger.info(f"Saving processed header to {output_header_file}")
        save_header_file(processed_header, output_header_file)

    # Clean up Dask client if it was created
    if dask_client:
        try:
            dask_client.close()
        except Exception as e:
            logger.warning(f"Error closing Dask client: {str(e)}")

    logger.info("Data preprocessing completed successfully")
    return df, fmap, processed_header


# Function to determine if a column should be numeric
def should_be_numeric(col_name, df):
    # If column already has float64 dtype, it should be numeric
    if df[col_name].dtype == 'float64':
        return True

    # If column has object dtype but contains mostly numbers, it should be numeric
    if df[col_name].dtype == 'object':
        # Sample the column to check if values are numeric
        sample = df[col_name].dropna().sample(
            min(1000, len(df[col_name].dropna())))
        numeric_count = sum(1 for x in sample if isinstance(x, (int, float)) or
                            (isinstance(x, str) and x.replace('-', '').replace('.', '').isdigit()))
        if numeric_count > 0.8 * len(sample):  # 80% threshold
            return True

    return False


def save_dask_df_to_parquet(df, output_path, header=None):
    """Multi-fallback approach for saving Dask DataFrames to Parquet"""
    import os

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Attempt 1: Direct conversion with type forcing for all columns
    try:
        # Force all object columns to string and numeric columns to float64
        df2 = df.copy()
        for col in df2.columns:
            if df2[col].dtype == 'object':
                df2[col] = df2[col].astype(str)
            elif 'float' in str(df2[col].dtype) or 'int' in str(df2[col].dtype):
                df2[col] = df2[col].astype('float64')

        df2.to_parquet(output_path)
        return True
    except Exception as e:
        print(f"First parquet attempt failed: {str(e)}. Trying alternative...")

    # Attempt 2: Compute then save
    try:
        print("Computing full DataFrame...")
        pdf = df.compute()

        # Convert object columns to string
        for col in pdf.select_dtypes(include=['object']).columns:
            pdf[col] = pdf[col].astype(str)

        pdf.to_parquet(output_path)
        return True
    except Exception as e:
        print(
            f"Second parquet attempt failed: {str(e)}. Trying CSV fallback...")

    # Attempt 3: CSV fallback (almost always works)
    try:
        csv_path = output_path.replace('.parquet', '.csv')
        df.to_csv(csv_path, single_file=True, index=False)
        print(f"Saved as CSV instead at: {csv_path}")
        return False
    except Exception as e:
        print(f"All save attempts failed: {str(e)}")
        raise


# def save_dask_df_to_parquet(df, output_path, header=None):
#     """Save Dask DataFrame to parquet with type consistency enforcement"""
#     import pyarrow as pa

#     # Determine column types from sample data
#     # First, collect metadata about types across partitions
#     sample = df.head(100)  # Small sample to analyze

#     # Build type map based on header information and data sample
#     column_types = {}
#     for col in df.columns:
#         if header and "featuretypes" in header and "featurenames" in header:
#             # If we have header info, use it to guide type decisions
#             try:
#                 col_idx = header["featurenames"].index(col)
#                 col_type = header["featuretypes"][col_idx]

#                 # Use header type info to guide conversion
#                 if col_type in ["C", "K"]:  # Categorical or Key columns
#                     column_types[col] = "string"
#                 elif col_type in ["N", "L", "W"]:  # Numeric, Label, or Weight columns
#                     column_types[col] = "float64"
#                 else:
#                     # Default to inferred type
#                     column_types[col] = str(sample[col].dtype)
#             except (ValueError, IndexError):
#                 # Column not in header, use inferred type
#                 column_types[col] = str(sample[col].dtype)
#         else:
#             # No header, use inferred type from sample
#             column_types[col] = str(sample[col].dtype)

#     # Force consistent types across all partitions
#     typed_df = df.copy()
#     for col, dtype in column_types.items():
#         if dtype == "string" or "object" in dtype:
#             # Convert to string to ensure consistency
#             typed_df[col] = typed_df[col].astype(str)
#         elif "float" in dtype or "int" in dtype:
#             # Convert to float64 for numeric columns
#             typed_df[col] = typed_df[col].astype("float64")

#     # Create explicit schema for PyArrow
#     fields = []
#     for col in typed_df.columns:
#         if column_types[col] == "string" or "object" in column_types[col]:
#             fields.append((col, pa.string()))
#         else:
#             fields.append((col, pa.float64()))

#     schema = pa.schema(fields)

#     # Write with explicit schema
#     typed_df.to_parquet(output_path, schema=schema)


# def save_dask_df_to_parquet(df, output_path):
#     import pyarrow as pa
#     import pandas as pd

#     # Get metadata from all partitions
#     meta_dtypes = df._meta.dtypes

#     # Create schema dynamically
#     schema = {}
#     for col, dtype in meta_dtypes.items():
#         # If this is a string/object column in the metadata
#         if dtype == 'object' or str(dtype).startswith('string'):
#             schema[col] = pa.string()
#         else:
#             schema[col] = pa.from_numpy_dtype(dtype)

#     logger.info(f"Created schema for {len(schema)} columns")

#     # Force consistent dtypes across partitions
#     type_conversions = {}
#     for col in schema:
#         if schema[col] == pa.string():
#             type_conversions[col] = 'string'

#     # Apply type conversions to ensure consistency
#     if type_conversions:
#         logger.info(
#             f"Converting {len(type_conversions)} columns to string type")
#         df = df.astype(type_conversions)

#     # Save with schema
#     logger.info(f"Saving DataFrame to {output_path}")
#     df.to_parquet(output_path, engine='pyarrow', schema=schema)
#     return True


def get_problematic_columns(ddf):
    """For dask dataframe saving to parquet, convert mixed dtypes to same dtype per the header file"""
    logger.info("Dealing with problematic columns...")
    import pyarrow as pa

    sample = ddf.head(constants.DEFAULT_SAMPLE_FIXED_SIZE_1000)

    schema = {}
    for col in sample.columns:
        # Skip already numeric columns
        if sample[col].dtype in [constants.FLOAT64_TYPE, constants.FLOAT32_TYPE, constants.INT64_TYPE]:
            continue

        # Look for string values that would faile numeric conversion
        if sample[col].dtype == constants.OBJECT_TYPE:
            if sample[col].str.contains('-99999999').any():
                schema[col] = pa.string()
    logger.info("Schema after dealing with problematic columns %s", schema)
    return schema


def save_df_to_parquet(df, output_file):
    """
    Safely save a DataFrame (pandas or Dask) to Parquet format.
    Supports both local and GCS paths.

    Args:
        df: pandas DataFrame or Dask DataFrame to save
        output_file: Path to save the Parquet file (local or GCS path)
    """
    import pandas as pd
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Check if it's a GCS path
    is_gcs_path = output_file.startswith('gs://')

    # Check if it's a Dask DataFrame
    is_dask = hasattr(df, 'dask')

    # Create directory if it's a local path
    if not is_gcs_path:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if is_dask:
        try:
            # First attempt: Try direct write with PyArrow engine
            df.to_parquet(
                output_file,
                write_index=False,
                engine='pyarrow',
                compression='snappy'
            )
            logger.info(f"Successfully saved Dask DataFrame to {output_file}")
            return
        except Exception as e1:
            logger.warning(
                f"First Parquet attempt failed: {str(e1)}. Trying with schema conversion.")

            try:
                # Second attempt: Convert problematic columns to strings first
                # This is a common workaround for PyArrow type issues

                # Get metadata to identify object columns
                meta = df._meta
                object_columns = meta.select_dtypes(
                    include=['object']).columns.tolist()

                # Convert object columns to strings to avoid PyArrow conversion issues
                for col in object_columns:
                    df[col] = df[col].astype(str)

                # Try writing again
                df.to_parquet(
                    output_file,
                    write_index=False,
                    engine='pyarrow',
                    compression='snappy'
                )
                logger.info(
                    f"Successfully saved Dask DataFrame to {output_file} after schema conversion")
                return
            except Exception as e2:
                logger.warning(
                    f"Second Parquet attempt failed: {str(e2)}. Trying with compute.")

                try:
                    # Third attempt: Compute the DataFrame and then save
                    # This is more memory-intensive but can resolve many issues
                    logger.info(
                        "Computing Dask DataFrame before saving to Parquet...")

                    # Compute the DataFrame (convert to pandas)
                    pandas_df = df.compute()

                    # Prepare and save the pandas DataFrame
                    prepared_df = prepare_pandas_df_for_parquet(pandas_df)

                    if is_gcs_path:
                        try:
                            import gcsfs
                            fs = gcsfs.GCSFileSystem()
                            with fs.open(output_file, 'wb') as f:
                                prepared_df.to_parquet(f, index=False)
                        except ImportError:
                            prepared_df.to_parquet(output_file, index=False)
                    else:
                        prepared_df.to_parquet(output_file, index=False)

                    logger.info(
                        f"Successfully saved computed DataFrame to {output_file}")
                    return
                except Exception as e3:
                    logger.error(
                        f"All Parquet attempts failed: {str(e3)}. Unable to save as Parquet.")
                    raise
    else:
        # For pandas DataFrames, prepare and save
        prepared_df = prepare_pandas_df_for_parquet(df)

        if is_gcs_path:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                with fs.open(output_file, 'wb') as f:
                    prepared_df.to_parquet(f, index=False)
            except ImportError:
                prepared_df.to_parquet(output_file, index=False)
        else:
            prepared_df.to_parquet(output_file, index=False)

        logger.info(f"Successfully saved pandas DataFrame to {output_file}")


def prepare_pandas_df_for_parquet(df):
    """
    Prepare pandas DataFrame for Parquet by ensuring type compatibility.

    Args:
        df: pandas DataFrame to prepare

    Returns:
        pandas DataFrame with compatible types for Parquet
    """
    import pandas as pd
    import numpy as np

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Process each column
    for col in df_copy.columns:
        # Get the column data
        col_data = df_copy[col]
        col_dtype = col_data.dtype

        # Handle object (string) columns
        if col_dtype == 'object':
            try:
                # Try to convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(col_data, errors='coerce')

                # Check if conversion was mostly successful (less than 10% NaN)
                if numeric_series.isna().mean() < 0.1:
                    df_copy[col] = numeric_series
                else:
                    # Otherwise, ensure it's properly formatted as string
                    df_copy[col] = col_data.fillna('').astype(str)

                    # Find strings that look like numbers
                    mask = df_copy[col].str.match(r'^-?\d+(\.\d+)?$')
                    # Only process if any matches found
                    if mask.any():
                        # Add prefix to numeric-looking strings
                        df_copy.loc[mask, col] = 'str_' + \
                            df_copy.loc[mask, col]
            except Exception as e:
                # For any errors, ensure it's string
                df_copy[col] = col_data.fillna('').astype(str)

        # Handle numeric columns with potential NaN/None values
        elif np.issubdtype(col_dtype, np.number):
            # For numeric columns, just ensure NaN values are properly represented
            df_copy[col] = col_data.fillna(np.nan)

    return df_copy
