import numpy as np
import polars as pl
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, List, Set
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer


class WindowDataBuilder:
    """
    Builder class for creating time-windowed features for fraud detection.

    This class transforms raw transaction data into aggregated features
    organized by time windows, specifically designed for fraud detection models.
    """

    # Class constants for better maintainability
    CATEGORICAL_COLUMNS = ['city', 'country', 'description']
    DEFAULT_CORRELATION_THRESHOLD = 0.75
    DEFAULT_N_CLUSTERS = 3
    DEFAULT_MIN_BINS = 1
    DEFAULT_MAX_BINS = 5

    # Scale factors for feature normalization
    SCALE_FACTORS = {
        'amount': 1_000,
        'per_capita_income': 1_000,
        'total_debt': 10_000,
        'credit_score': 100,
        'credit_limit': 10_000,
    }

    # Numerical columns for discretization
    NUMERICAL_COLUMNS_FOR_DISCRETIZATION = [
        'amount', 'per_capita_income', 'total_debt', 'credit_score',
        'credit_limit', 'current_age', 'num_credit_cards'
    ]

    def __init__(self, df: pl.DataFrame, method_corr='pearson', save_features_corr=None,
                 discretize_numerical=True, discretization_strategy='uniform'):
        """
        Initialize the WindowDataBuilder.

        Args:
            df (pl.DataFrame): Input DataFrame containing transaction data
            method_corr (str): Correlation method for feature selection
            save_features_corr: Features to save from correlation removal
            discretize_numerical (bool): Whether to discretize numerical features
            discretization_strategy (str): Strategy for discretization ('uniform' or 'quantile')

        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        self._validate_input_data(df)
        self.df = df
        self._scaler = MinMaxScaler()
        self.method_corr = method_corr
        self.save_features_corr = save_features_corr
        self.discretize_numerical = discretize_numerical
        self.discretization_strategy = discretization_strategy
        # Dicionário para armazenar os intervalos das faixas de discretização
        self.discretization_intervals = {}

    def _validate_input_data(self, df: pl.DataFrame) -> None:
        """Validate that required columns exist in the input DataFrame."""
        required_columns = {'date', 'target', 'client_id', 'merchant_id', 'amount'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def create_cluster_target_fields(
        self,
        df: pl.DataFrame,
        group_column_name: str,
        target_column_name: str,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        drop_first: bool = False
    ) -> pl.DataFrame:
        """
        Create clustered categorical features based on target variable patterns.

        Args:
            df: Input DataFrame
            group_column_name: Column to group by
            target_column_name: Target variable column
            n_clusters: Number of clusters for KMeans
            drop_first: Whether to drop first dummy variable

        Returns:
            DataFrame with clustered dummy variables
        """
        try:
            # Define column names for better readability
            percent_target_col = f'percent_{target_column_name}'
            percent_no_target_col = f'percent_no_{target_column_name}'
            ab_col = 'ab'
            target_cluster_col = f'{target_column_name}_{group_column_name}'

            # Calculate target percentages by group
            df_cat_group = (
                df.group_by(group_column_name)
                .agg([
                    ((pl.col(target_column_name).sum() / pl.count()) * 100).alias(percent_target_col),
                    (((pl.count() - pl.col(target_column_name).sum()) / pl.count()) * 100).alias(percent_no_target_col)
                ])
                .with_columns([
                    # Calculate A/B ratio with proper handling of edge cases
                    pl.when(pl.col(percent_no_target_col) == 0)
                    .then(pl.lit(100.0))
                    .otherwise(pl.col(percent_target_col) / pl.col(percent_no_target_col))
                    .alias(ab_col)
                ])
                .with_columns([
                    # Cap the A/B ratio at 100 for stability
                    pl.when(pl.col(ab_col) > 100.0)
                    .then(pl.lit(100.0))
                    .otherwise(pl.col(ab_col))
                    .alias(ab_col)
                ])
            )

            # Apply clustering
            ab_values = df_cat_group.select(ab_col).to_pandas()
            scaled_values = self._scaler.fit_transform(ab_values)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_values)

            # Add cluster labels and create dummy variables
            df_cat_group = df_cat_group.with_columns(
                pl.lit(cluster_labels).alias(target_cluster_col)
            )

            return (
                df_cat_group
                .select([group_column_name, target_cluster_col])
                .to_dummies(columns=[target_cluster_col], drop_first=drop_first)
            )

        except Exception as e:
            raise RuntimeError(f"Error in create_cluster_target_fields: {str(e)}")

    def _discretize_numerical_features(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Discretize numerical features based on their amplitude.
        Applies scaling consistently with non-discretization approach.

        Args:
            df: Input DataFrame with numerical columns

        Returns:
            Dictionary with discretized features for each numerical column
        """
        discretized_features = {}

        try:
            for column in self.NUMERICAL_COLUMNS_FOR_DISCRETIZATION:
                if column not in df.columns:
                    continue

                # Convert to pandas for sklearn processing
                column_data = df.select([column, 'target']).to_pandas()

                # Skip if column has insufficient data
                if column_data[column].isna().all() or column_data[column].nunique() < 2:
                    continue

                # Apply scaling factor if defined (consistent with non-discretization approach)
                scale_factor = self.SCALE_FACTORS.get(column, 1.0)
                scaled_column_data = column_data[column] / scale_factor

                # Calculate amplitude and determine number of bins using scaled data
                min_val = scaled_column_data.min()
                max_val = scaled_column_data.max()
                amplitude = max_val - min_val

                # Determine number of bins based on data distribution (using scaled data)
                unique_vals = scaled_column_data.nunique()
                n_bins = min(
                    max(self.DEFAULT_MIN_BINS, min(unique_vals // 10, self.DEFAULT_MAX_BINS)),
                    unique_vals
                )

                if n_bins < 2:
                    continue

                # Apply discretization to scaled data
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode='ordinal',
                    strategy=self.discretization_strategy,
                    subsample=None
                )

                # Prepare scaled data for discretization (fill NaN with median of scaled data)
                clean_scaled_data = pd.DataFrame({column: scaled_column_data}).fillna(scaled_column_data.median())

                # Fit and transform the scaled data
                discretized_values = discretizer.fit_transform(clean_scaled_data).astype(int)

                # Capturar os intervalos das faixas geradas pelo discretizador (em escala aplicada)
                bin_edges = discretizer.bin_edges_[0]  # bin_edges_ é uma lista com arrays
                intervals = []

                for i in range(len(bin_edges) - 1):
                    interval = {
                        'bin_id': i,
                        'lower_bound': float(bin_edges[i]),
                        'upper_bound': float(bin_edges[i + 1]),
                        'is_left_inclusive': True,
                        'is_right_inclusive': i == len(bin_edges) - 2  # Última faixa inclui o limite superior
                    }
                    intervals.append(interval)

                # Armazenar os intervalos no dicionário da classe (com informações de escala)
                self.discretization_intervals[column] = {
                    'intervals': intervals,
                    'n_bins': n_bins,
                    'strategy': self.discretization_strategy,
                    'min_value': float(min_val),
                    'max_value': float(max_val),
                    'median_value': float(scaled_column_data.median()),
                    'scale_factor': scale_factor,  # Adicionando informação do fator de escala
                    'original_min_value': float(column_data[column].min()),  # Valores originais para referência
                    'original_max_value': float(column_data[column].max()),
                    'original_median_value': float(column_data[column].median())
                }

                # Create DataFrame with discretized values
                discretized_df = pl.DataFrame({
                    f'{column}_discretized': discretized_values.flatten(),
                    'target': column_data['target'].values
                })

                # Create dummy variables directly from discretized bins (no clustering needed)
                df_discretized_dummies = (
                    discretized_df
                    .select([f'{column}_discretized'])
                    .to_dummies(columns=[f'{column}_discretized'], drop_first=False)
                )

                if not df_discretized_dummies.is_empty():
                    discretized_features[column] = df_discretized_dummies

        except Exception as e:
            print(f"Warning: Error in discretization: {str(e)}")

        return discretized_features

    def _process_discretized_numerical_features(
        self,
        df: pl.DataFrame,
        discretized_features: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Process discretized numerical features similar to categorical features.

        Args:
            df: Original DataFrame with date_window
            discretized_features: Dictionary with discretized features

        Returns:
            DataFrame with discretized features aggregated by time window
        """
        if not discretized_features:
            return pl.DataFrame({'date_window': df.select('date_window').unique().to_pandas()['date_window']})

        result_features = []

        try:
            for column, df_discretized in discretized_features.items():
                # Add original data back for aggregation
                df_with_discretized = (
                    df.select(['date_window']).with_row_index()
                    .join(
                        df_discretized.with_row_index(),
                        on='index',
                        how='left'
                    )
                    .drop(['index'])
                )

                # Aggregate by time window
                df_agg = (
                    df_with_discretized.group_by('date_window')
                    .sum()
                )

                # Calculate proportions if there are discretized features
                if len(df_agg.columns) > 1:
                    df_agg = df_agg.with_columns(
                        pl.sum_horizontal([
                            pl.col(col) for col in df_agg.columns[1:]
                        ]).alias('total')
                    )

                    # Calculate proportions for each discretized bin
                    proportion_expressions = [
                        (pl.col(col) / pl.col('total')).alias(f'{column}_{col}')
                        for col in df_agg.columns[1:-1]  # Exclude date_window and total
                    ]

                    if proportion_expressions:
                        df_agg = (
                            df_agg.with_columns(proportion_expressions)
                            .select(['date_window'] + [f'{column}_{col}' for col in df_agg.columns[1:-1]])
                        )
                        result_features.append(df_agg)

        except Exception as e:
            print(f"Warning: Error processing discretized features: {str(e)}")

        # Combine all discretized features
        if result_features:
            combined_df = result_features[0]
            for df_feat in result_features[1:]:
                combined_df = combined_df.join(df_feat, on='date_window', how='outer', suffix='_dup')
                # Remove duplicate columns created by join
                duplicate_cols = [col for col in combined_df.columns if col.endswith('_dup')]
                if duplicate_cols:
                    combined_df = combined_df.drop(duplicate_cols)
            return combined_df
        else:
            return pl.DataFrame({'date_window': df.select('date_window').unique().to_pandas()['date_window']})

    def _generate_column_statistics(self, column_name: str, divide_by: float = 1.0) -> List[pl.Expr]:
        """
        Generate statistical aggregations for a column.

        Args:
            column_name: Name of the column to generate statistics for
            divide_by: Scaling factor for normalization

        Returns:
            List of Polars expressions for statistical aggregations
        """
        return [
            (pl.col(column_name).min() / divide_by).alias(f'min_{column_name}'),
            (pl.col(column_name).mean() / divide_by).alias(f'mean_{column_name}'),
            (pl.col(column_name).max() / divide_by).alias(f'max_{column_name}'),
            # (pl.col(column_name).std() / divide_by).alias(f'std_{column_name}'),
        ]

    def _create_base_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create base numerical features aggregated by time window.

        Args:
            df: Input DataFrame with date_window column

        Returns:
            DataFrame with aggregated numerical features
        """
        try:
            # Build aggregation expressions dynamically
            agg_expressions = [
                pl.col("target").sum().alias("frauds"),
                pl.col("target").count().alias("transactions"),
                pl.col("client_id").n_unique().alias("unique_clients"),
                pl.col("merchant_id").n_unique().alias("unique_merchants"),
                pl.col("amount").sum().alias("total_amount"),
            ]

            # Add statistical features for numerical columns only if discretization is disabled
            if not self.discretize_numerical:
                # Add statistical features for numerical columns
                for column, scale_factor in self.SCALE_FACTORS.items():
                    if column in df.columns:
                        agg_expressions.extend(
                            self._generate_column_statistics(column, scale_factor)
                        )

                # Add statistics for columns without scaling
                for column in ['current_age', 'num_credit_cards']:
                    if column in df.columns:
                        agg_expressions.extend(
                            self._generate_column_statistics(column, 1.0)
                        )

            # Perform aggregation
            df_features = (
                df.group_by("date_window")
                .agg(agg_expressions)
                .with_columns([
                    (pl.col("transactions") / pl.col("unique_clients")).alias("transaction_per_client"),
                    (pl.col("transactions") / pl.col("unique_merchants")).alias("transaction_per_merchant"),
                ])
            )

            # Remove columns based on discretization setting
            columns_to_drop = ["transactions", "total_amount", "unique_clients", "unique_merchants"]

            # Only drop min/max amount columns if discretization is disabled (they were created)
            if not self.discretize_numerical:
                columns_to_drop.extend([f"min_{col}" for col in ["amount"] if col in df.columns])
                columns_to_drop.extend([f"max_{col}" for col in ["amount"] if col in df.columns])

            df_features = df_features.drop([col for col in columns_to_drop if col in df_features.columns]).drop_nulls()

            return df_features

        except Exception as e:
            raise RuntimeError(f"Error creating base features: {str(e)}")

    def _process_categorical_features(self, df: pl.DataFrame, df_base: pl.DataFrame) -> pl.DataFrame:
        """
        Process categorical features and merge with base features.

        Args:
            df: Original DataFrame with categorical columns
            df_base: Base features DataFrame

        Returns:
            DataFrame with categorical and numerical features combined
        """
        result_df = df_base

        for column in self.CATEGORICAL_COLUMNS:
            if column not in df.columns:
                continue

            try:
                # Create cluster features for categorical column
                df_target_cat = self.create_cluster_target_fields(
                    df, column, 'target', drop_first=False
                )

                # Aggregate categorical features by time window
                df_with_cat = (
                    df.select(['date_window', column])
                    .join(df_target_cat, on=column, how='left')
                    .drop(column)
                    .group_by('date_window')
                    .sum()
                )

                # Calculate proportions if there are categorical features
                if len(df_with_cat.columns) > 1:
                    df_with_cat = df_with_cat.with_columns(
                        pl.sum_horizontal([
                            pl.col(col) for col in df_with_cat.columns[1:]
                        ]).alias('total')
                    )

                    # Calculate proportions for each categorical feature
                    proportion_expressions = [
                        (pl.col(col) / pl.col('total')).alias(col)
                        for col in df_with_cat.columns[1:-1]  # Exclude date_window and total
                    ]

                    if proportion_expressions:
                        df_with_cat = (
                            df_with_cat.with_columns(proportion_expressions)
                            .drop('total')
                        )

                    # Merge with result DataFrame
                    result_df = result_df.join(df_with_cat, on='date_window', how='left')

            except Exception as e:
                print(f"Warning: Error processing categorical column '{column}': {str(e)}")
                continue

        return result_df

    def _remove_highly_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = DEFAULT_CORRELATION_THRESHOLD
    ) -> List[str]:
        """
        Identify features with high correlation for removal.

        Args:
            df: Input DataFrame
            threshold: Correlation threshold for removal

        Returns:
            List of column names to remove
        """
        try:
            if df.empty or len(df.columns) <= 1:
                return []

            # Calculate correlation matrix more efficiently
            corr_matrix = df.corr(numeric_only=True, method=self.method_corr)

            # Create mask for upper triangle
            upper_triangle = np.triu(np.abs(corr_matrix), k=1)

            # Find highly correlated pairs
            high_corr_pairs = np.where(upper_triangle > threshold)
            columns_to_remove: Set[str] = set()

            # Process correlated pairs
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Skip if either column already marked for removal
                if col_i in columns_to_remove or col_j in columns_to_remove:
                    continue

                # Remove column with lower variance
                var_i = df[col_i].var()
                var_j = df[col_j].var()

                if pd.isna(var_i) or pd.isna(var_j):
                    continue

                if var_i < var_j:
                    columns_to_remove.add(col_i)
                else:
                    columns_to_remove.add(col_j)

            return list(columns_to_remove)

        except Exception as e:
            print(f"Warning: Error in correlation analysis: {str(e)}")
            return []

    def build(self, window: str, remove_high_correlation=True, remove_first_discretized=False) -> tuple[pd.DataFrame, dict]:
        """
        Build the complete feature dataset for the specified time window.

        Args:
            window: Time window specification (e.g., '1d', '1w', '1mo')
            remove_high_correlation: Whether to remove highly correlated features
            remove_first_discretized: Whether to remove the first discretized feature

        Returns:
            tuple: (pandas DataFrame with engineered features ready for modeling,
                   dict with discretization intervals for numerical features)

        Raises:
            ValueError: If window specification is invalid
            RuntimeError: If feature engineering fails
        """
        try:
            if not isinstance(window, str) or not window.strip():
                raise ValueError("Window must be a non-empty string")

            # Create time windows
            df_windowed = self.df.with_columns(
                pl.col('date').dt.truncate(window).alias('date_window')
            )

            # Create base numerical features
            df_base = self._create_base_features(df_windowed)

            # Process categorical features
            df_with_categories = self._process_categorical_features(df_windowed, df_base)

            # Process discretized numerical features if enabled
            result_df = df_with_categories
            if self.discretize_numerical:
                discretized_features = self._discretize_numerical_features(df_windowed)
                df_discretized = self._process_discretized_numerical_features(
                    df_windowed, discretized_features
                )

                # Merge discretized features if they exist
                if not df_discretized.is_empty() and len(df_discretized.columns) > 1:
                    result_df = result_df.join(df_discretized, on='date_window', how='left')

                    if remove_first_discretized:
                        first_discretized_columns = [column for column in result_df.columns if str(column).endswith('_0')]
                        result_df = result_df.drop(first_discretized_columns)

                    print(f"Added {len(df_discretized.columns) - 1} discretized numerical features")

            # Convert to pandas for correlation analysis
            df_pandas = result_df.to_pandas()

            # Remove highly correlated features
            removed_columns = ['date_window']
            if self.save_features_corr:
              if isinstance(self.save_features_corr, str):
                removed_columns.append(self.save_features_corr)
              elif isinstance(self.save_features_corr, list):
                removed_columns.extend(self.save_features_corr)
            columns_to_remove = self._remove_highly_correlated_features(
                df_pandas.drop(columns=removed_columns, errors='ignore')
            )

            if columns_to_remove and remove_high_correlation:
                df_pandas = df_pandas.drop(columns=columns_to_remove, errors='ignore')
                print(f"Removed {len(columns_to_remove)} highly correlated features")
                print(f'Columns removed: {", ".join(columns_to_remove)}')

            # Retornar tanto o DataFrame quanto os intervalos de discretização
            return df_pandas.sort_values('date_window'), self.__get_discretization_intervals()

        except Exception as e:
            raise RuntimeError(f"Error building features: {str(e)}")

    def get_feature_info(self) -> dict:
        """
        Get information about the feature engineering process.

        Returns:
            Dictionary with feature engineering configuration
        """
        return {
            'categorical_columns': self.CATEGORICAL_COLUMNS,
            'numerical_columns_for_discretization': self.NUMERICAL_COLUMNS_FOR_DISCRETIZATION,
            'correlation_threshold': self.DEFAULT_CORRELATION_THRESHOLD,
            'n_clusters': self.DEFAULT_N_CLUSTERS,
            'min_bins': self.DEFAULT_MIN_BINS,
            'max_bins': self.DEFAULT_MAX_BINS,
            'discretize_numerical': self.discretize_numerical,
            'discretization_strategy': self.discretization_strategy,
            'scale_factors': self.SCALE_FACTORS,
        }

    def __get_discretization_intervals(self) -> dict:
        """
        Retorna os intervalos de discretização para cada feature numérica.

        Returns:
            Dicionário com os intervalos de discretização, onde cada chave é o nome
            da feature e o valor contém informações sobre os intervalos (limites
            inferior e superior de cada faixa).

        Example:
            {
                'amount': {
                    'intervals': [
                        {'bin_id': 0, 'lower_bound': 0.0, 'upper_bound': 100.0,
                         'is_left_inclusive': True, 'is_right_inclusive': False},
                        {'bin_id': 1, 'lower_bound': 100.0, 'upper_bound': 500.0,
                         'is_left_inclusive': True, 'is_right_inclusive': True}
                    ],
                    'n_bins': 2,
                    'strategy': 'uniform',
                    'min_value': 0.0,
                    'max_value': 500.0,
                    'median_value': 250.0,
                    'scale_factor': 1000.0,
                    'original_min_value': 0.0,
                    'original_max_value': 500000.0,
                    'original_median_value': 250000.0
                }
            }
        """
        return self.discretization_intervals.copy()

    def get_discretization_intervals_original_scale(self) -> dict:
        """
        Retorna os intervalos de discretização convertidos para a escala original.
        Útil para interpretação e visualização dos intervalos.

        Returns:
            Dicionário com os intervalos em escala original
        """
        original_intervals = {}

        for column, info in self.discretization_intervals.items():
            scale_factor = info.get('scale_factor', 1.0)

            # Converter intervalos para escala original
            original_interval_list = []
            for interval in info['intervals']:
                original_interval = {
                    'bin_id': interval['bin_id'],
                    'lower_bound': float(interval['lower_bound'] * scale_factor),
                    'upper_bound': float(interval['upper_bound'] * scale_factor),
                    'is_left_inclusive': interval['is_left_inclusive'],
                    'is_right_inclusive': interval['is_right_inclusive']
                }
                original_interval_list.append(original_interval)

            original_intervals[column] = {
                'intervals': original_interval_list,
                'n_bins': info['n_bins'],
                'strategy': info['strategy'],
                'scale_factor': scale_factor,
                'min_value': info.get('original_min_value', info['min_value'] * scale_factor),
                'max_value': info.get('original_max_value', info['max_value'] * scale_factor),
                'median_value': info.get('original_median_value', info['median_value'] * scale_factor)
            }

        return original_intervals

    def get_bin_for_value(self, column: str, value: float) -> int:
        """
        Determina em qual faixa (bin) um valor específico se encaixa para uma feature.
        Aplica automaticamente a escala antes da classificação.

        Args:
            column: Nome da feature numérica
            value: Valor para classificar (em escala original)

        Returns:
            ID da faixa (bin) onde o valor se encaixa, ou -1 se não encontrado

        Raises:
            ValueError: Se a coluna não foi discretizada
        """
        if column not in self.discretization_intervals:
            raise ValueError(f"Column '{column}' was not discretized or does not exist")

        intervals_info = self.discretization_intervals[column]
        intervals = intervals_info['intervals']
        scale_factor = intervals_info.get('scale_factor', 1.0)

        # Aplicar escala ao valor (consistente com o processo de discretização)
        scaled_value = value / scale_factor if not pd.isna(value) else intervals_info['median_value']

        # Tratar valores NaN usando a mediana escalada
        if pd.isna(value):
            scaled_value = intervals_info['median_value']

        # Verificar em qual intervalo o valor escalado se encaixa
        for interval in intervals:
            lower_bound = interval['lower_bound']
            upper_bound = interval['upper_bound']
            is_left_inclusive = interval['is_left_inclusive']
            is_right_inclusive = interval['is_right_inclusive']

            # Verificar se o valor está dentro do intervalo
            left_condition = scaled_value >= lower_bound if is_left_inclusive else scaled_value > lower_bound
            right_condition = scaled_value <= upper_bound if is_right_inclusive else scaled_value < upper_bound

            if left_condition and right_condition:
                return interval['bin_id']

        # Se não encontrou nenhuma faixa, retornar a última faixa (casos extremos)
        return len(intervals) - 1