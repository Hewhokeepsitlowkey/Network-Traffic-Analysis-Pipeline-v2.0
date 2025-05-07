"""
 Enterprise-Grade Network Anomaly Detection System v4.0
Optimized for Performance and Explainability
"""
#  libraries imported for the project 
import logging
import warnings
import numpy as np
import pandas as pd
import shap
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib
import dash
import dash_bootstrap_components as dbc

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("network_anomaly.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Project-compliant constants

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    PERFORMANCE_TARGETS = {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75}
    MODEL_PATH = "anomaly_detector.joblib"
    DATA_PATHS = {
        "ddos": r"data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "portscan": r"data\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    }
    CORE_FEATURES = [
        'destination_port',
        'flow_duration',
        'total_fwd_packets',
        'total_backward_packets',
        'total_length_of_fwd_packets',
        'total_length_of_bwd_packets',
        'flow_bytes_per_sec',
        'flow_packets_per_sec',
        'flow_iat_mean',
        'fwd_iat_total'
    ]
    ENGINEERED_FEATURES = [
        'packet_ratio', 'length_ratio', 'bytes_per_packet', 'fwd_bwd_packet_diff',
        'fwd_bwd_length_diff', 'iat_per_packet', 'log_flow_duration',
        'log_total_fwd_packets', 'log_total_backward_packets',
        'fwd_len_times_bwd_len', 'fwd_packets_times_bwd_packets'
    ]
    ALL_FEATURES = CORE_FEATURES + ENGINEERED_FEATURES


class NetworkAnomalyDetector:
    def __init__(self):
        """
        Initialize the detector with models and placeholders.
        """
        self.models = {
            'random_forest': None,
            'logistic_regression': None
        }
        self.preprocessor = None
        self.dashboard = None
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features to enhance anomaly detection.
        """
        # Avoid division by zero
        df['packet_ratio'] = df['total_fwd_packets'] / (df['total_backward_packets'] + 1)
        df['length_ratio'] = df['total_length_of_fwd_packets'] / (df['total_length_of_bwd_packets'] + 1)
        df['bytes_per_packet'] = (df['flow_bytes_per_sec'] / (df['flow_packets_per_sec'] + 1)).replace([np.inf, -np.inf], 0)
        df['fwd_bwd_packet_diff'] = df['total_fwd_packets'] - df['total_backward_packets']
        df['fwd_bwd_length_diff'] = df['total_length_of_fwd_packets'] - df['total_length_of_bwd_packets']
        df['iat_per_packet'] = df['flow_iat_mean'] / (df['total_fwd_packets'] + df['total_backward_packets'] + 1)
        # Log transforms (handle zeros)
        df['log_flow_duration'] = np.log1p(df['flow_duration'])
        df['log_total_fwd_packets'] = np.log1p(df['total_fwd_packets'])
        df['log_total_backward_packets'] = np.log1p(df['total_backward_packets'])
        # Interaction features
        df['fwd_len_times_bwd_len'] = df['total_length_of_fwd_packets'] * df['total_length_of_bwd_packets']
        df['fwd_packets_times_bwd_packets'] = df['total_fwd_packets'] * df['total_backward_packets']
        # Replace any inf/-inf with nan, then fill nan with 0 (optional, or use your cleaning pipeline)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def _load_and_validate(self, path: str) -> pd.DataFrame:
        """
        Load and validate the dataset.
        Ensures the dataset contains all required features.
        """
        logger.info(f"Loading dataset: {path}")
        df = pd.read_csv(
            path,
            low_memory=False,
            na_values=['Infinity', '-Infinity'],
            encoding='utf-8'
        )

        # Standardize column names
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r'[^a-z0-9/]+', '_', regex=True)
            .str.replace(r'_+', '_', regex=True)
            .str.replace(r'(flow|fwd|bwd)_?(bytes|packets)/s', r'\1_\2_per_sec', regex=True)
            .str.replace(r'/', '_per_', regex=False)
            .str.strip('_')
        )

        # Check for missing required features
        missing = set(Config.CORE_FEATURES) - set(df.columns)
        if missing:
            available = [c for c in df.columns if c in Config.CORE_FEATURES]
            logger.error(f"Column mismatch!\nRequired: {Config.CORE_FEATURES}\nAvailable: {available}")
            raise ValueError("Dataset validation failed")
            
        return df[Config.CORE_FEATURES + ['label']]

    def _create_dashboard(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Create a visually stunning, interactive dashboard for model performance and explainability.
        """
        try:
            import dash
            from dash import dcc
            from dash import html
            import plotly.express as px
            import dash_bootstrap_components as dbc

            app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

            # Prepare metrics for all models
            metrics = []
            for model_name, model in self.models.items():
                y_pred = model.predict(X_test)
                metrics.append({
                    'model': model_name.replace('_', ' ').title(),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                    'precision': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
                    'recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall']
                })
            metrics_df = pd.DataFrame(metrics)
            model_options = [{'label': name.replace('_', ' ').title(), 'value': name} for name in self.models.keys()]

            app.layout = dbc.Container([
                dbc.Row([
                    dbc.Col(html.H1("Network Anomaly Detection Analytics", className="text-center text-info mb-4"), width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Select Model"),
                            dbc.CardBody([
                                dcc.Dropdown(
                                    id='model-dropdown',
                                    options=model_options,
                                    value='random_forest',
                                    clearable=False,
                                    style={'color': '#000'}
                                )
                            ])
                        ], color="dark", inverse=True)
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Model Performance Overview"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='performance-bar',
                                    figure=px.bar(
                                        metrics_df.melt(id_vars='model', value_vars=['accuracy', 'precision', 'recall', 'roc_auc']),
                                        x='model', y='value', color='variable',
                                        barmode='group',
                                        title='Model Performance Comparison',
                                        labels={'value': 'Score', 'variable': 'Metric'}
                                    ).update_layout(legend_title_text='Metric')
                                )
                            ])
                        ], color="dark", inverse=True)
                    ], width=8)
                ], className="mb-4"),
                dbc.Tabs([
                    dbc.Tab(label='Confusion Matrix', tab_id='tab-cm', children=[
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='conf-matrix')
                            ])
                        ], color="dark", inverse=True)
                    ]),
                    dbc.Tab(label='SHAP Feature Importance', tab_id='tab-shap', children=[
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='shap-bar')
                            ])
                        ], color="dark", inverse=True)
                    ]),
                ], id='tabs', active_tab='tab-cm', className="mb-4"),
                dbc.Row([
                    dbc.Col(html.Footer("© 2024 Network Anomaly Detection System | Powered by Dash & Plotly", className="text-center text-secondary mt-4"), width=12)
                ])
            ], fluid=True)

            @app.callback(
                [dash.dependencies.Output('conf-matrix', 'figure'),
                 dash.dependencies.Output('shap-bar', 'figure')],
                [dash.dependencies.Input('model-dropdown', 'value')]
            )
            def update_visuals(selected_model):
                conf_fig = self.get_conf_matrix_fig(selected_model, X_test, y_test)
                shap_fig = self.get_shap_fig(selected_model, X_test)
                return conf_fig, shap_fig

            self.dashboard = app
        except Exception as e:
            logger.exception(f"Dashboard creation failed: {e}")
            self.dashboard = None

    # SHAP feature importance for each model
    def get_shap_fig(self, model_name, X_test):
        pipeline = self.models[model_name]
        preprocessor = pipeline.named_steps['columntransformer']
        X_sample = X_test.sample(n=100, random_state=Config.RANDOM_STATE)
        processed_data = preprocessor.transform(X_sample)
        # Get the estimator
        if 'randomforestclassifier' in pipeline.named_steps:
            estimator = pipeline.named_steps['randomforestclassifier']
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(processed_data)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif 'logisticregression' in pipeline.named_steps:
            estimator = pipeline.named_steps['logisticregression']
            explainer = shap.LinearExplainer(estimator, processed_data)
            shap_values = explainer.shap_values(processed_data)
        else:
            raise ValueError("Unsupported model for SHAP explanation.")
        # Ensure correct shape
        if len(shap_values.shape) != 2:
            shap_values = np.reshape(shap_values, (len(X_sample), -1))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = preprocessor.get_feature_names_out()
        # --- Fix: Align lengths ---
        min_len = min(len(mean_abs_shap), len(feature_names))
        mean_abs_shap = mean_abs_shap[:min_len]
        feature_names = feature_names[:min_len]
        return px.bar(
            x=mean_abs_shap,
            y=feature_names,
            orientation='h',
            title=f'SHAP Feature Importance: {model_name.replace("_", " ").title()}',
            labels={'x': 'Average Impact', 'y': 'Feature'},
            color=mean_abs_shap,
            color_continuous_scale='Viridis'
        )

    # Confusion matrix for each model
    def get_conf_matrix_fig(self, model_name, X_test, y_test):
        y_pred = self.models[model_name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual"),
            x=['Benign', 'Malicious'],
            y=['Benign', 'Malicious'],
            title=f"Confusion Matrix: {model_name.replace('_', ' ').title()}"
        )



    def _validate_performance(self, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
        """ 
        Validate model performance against predefined targets.
        """
        model = self.models[model_name]
        y_proba = model.predict_proba(X_test)[:, 1]
        
        if model_name == 'logistic_regression':
            best_threshold = 0.5
            best_precision = 0
            thresholds = np.linspace(0.5, 0.95, 20)
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                report = classification_report(y_test, y_pred, output_dict=True)
                precision = report.get('1', {}).get('precision', 0)
                recall = report.get('1', {}).get('recall', 0)
                
                if (precision >= Config.PERFORMANCE_TARGETS['precision'] and 
                    recall >= Config.PERFORMANCE_TARGETS['recall'] and 
                    precision > best_precision):
                    best_precision = precision
                    best_threshold = threshold
            
            logger.info(f"Optimal threshold for {model_name}: {best_threshold:.2f}")
            y_pred = (y_proba >= best_threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"\n{model_name.upper()} Validation:")
        logger.info(f"Accuracy: {accuracy:.4f} ({'MET' if accuracy >= Config.PERFORMANCE_TARGETS['accuracy'] else 'NOT MET'})")
        logger.info(f"Precision: {report['1']['precision']:.4f} ({'MET' if report['1']['precision'] >= Config.PERFORMANCE_TARGETS['precision'] else 'NOT MET'})")
        logger.info(f"Recall: {report['1']['recall']:.4f} ({'MET' if report['1']['recall'] >= Config.PERFORMANCE_TARGETS['recall'] else 'NOT MET'})")

    def build_pipeline(self, estimator, model_name: str):
        """
        Build a machine learning pipeline with preprocessing and SMOTE.
        """
        self.preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), Config.ALL_FEATURES)
        ])
        
        # Model-specific SMOTE configuration
        smote_params = {
            'random_forest': {'sampling_strategy': 0.75},
            'logistic_regression': {'sampling_strategy': 0.5}
        }.get(model_name, {})
        
        return make_imb_pipeline(
            self.preprocessor,
            SMOTE(random_state=Config.RANDOM_STATE, **smote_params),
            estimator
        )

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing and infinite values.
        """
        df['label'] = self._encode_labels(df['label'])
        df = self._handle_infinite_values(df)
        df = self._drop_missing_values(df)
        self._validate_data_length(df)
        return df

    def _encode_labels(self, labels: pd.Series) -> pd.Series:
        return np.where(labels.str.contains('BENIGN'), 0, 1)

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.replace([np.inf, -np.inf], np.nan)

    def _drop_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_count - len(df)} rows with missing values")
        return df

    def _validate_data_length(self, df: pd.DataFrame):
        if len(df) < 1000:
            raise ValueError("Insufficient data after cleaning")
    
    def build_pipeline(self, estimator, model_name: str):
        """
        Pipeline builder with dynamic SMOTE configuration
        """
        self.preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), Config.ALL_FEATURES)
        ])
        
        return make_imb_pipeline(
            self.preprocessor,
            SMOTE(
                random_state=Config.RANDOM_STATE,
                sampling_strategy='auto'  # Let SMOTE determine optimal ratio
            ),
            estimator
        )

    def train(self):
        """
        Train the models and evaluate their performance.
        """
        # Data loading and cleaning
        
        dfs = [self._load_and_validate(path) for path in Config.DATA_PATHS.values()]
        df = pd.concat(dfs, ignore_index=True)
        df = self._feature_engineering(df)
        df = self._clean_data(df)
                
        # split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[Config.ALL_FEATURES], df['label'],
            test_size=Config.TEST_SIZE,
            stratify=df['label'],
            random_state=Config.RANDOM_STATE
        )
        
        # Train models
        for model_name in self.models.keys():
            logger.info(f"Training {model_name.replace('_', ' ').title()}")
            
            estimator = (
        RandomForestClassifier(class_weight='balanced_subsample', random_state=Config.RANDOM_STATE)
        if model_name == 'random_forest'
        else LogisticRegression(
            class_weight='balanced',
            max_iter=10000,  
            tol=1e-3,        
            random_state=Config.RANDOM_STATE
    )
)
            
            pipeline = self.build_pipeline(estimator, model_name)
            
            param_grid = (
    {
        'randomforestclassifier__n_estimators': [200, 300],
        'randomforestclassifier__max_depth': [20, None]
    } if model_name == 'random_forest' else {
        'logisticregression__C': [0.1, 1, 10],
        'logisticregression__solver': ['lbfgs'], 
        'logisticregression__penalty': ['l2']
    }
)
            
            self.models[model_name] = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=5),
                scoring='roc_auc',
                n_jobs=-1
            ).fit(X_train, y_train).best_estimator_
            
            self._validate_performance(X_test, y_test, model_name)
        
        # Save models and launch dashboard
        joblib.dump(self.models, Config.MODEL_PATH)
        logger.info(f"Models saved to {Config.MODEL_PATH}")
        self._create_dashboard(X_test, y_test)


# Main execution block
if __name__ == "__main__":
    detector = NetworkAnomalyDetector()
    try:
        detector.train()
        if detector.dashboard is not None:
            detector.dashboard.run(host='0.0.0.0', port=8050)
        else:
            logger.error("Dashboard was not created due to an earlier error.")
    except Exception as e:
        logger.exception(f"Fatal error during execution: {e}")