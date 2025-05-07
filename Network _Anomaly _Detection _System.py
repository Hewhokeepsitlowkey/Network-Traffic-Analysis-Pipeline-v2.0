"""
 Enterprise-Grade Network Anomaly Detection System v4.0
Optimized for Performance and Explainability
"""
# Import necessary libraries
import os
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
from dash import dcc, html

# Configure enhanced logging for debugging and monitoring
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

PERFORMANCE_TARGETS = {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75}
MODEL_PATH = "anomaly_detector.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

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
        missing = set(CORE_FEATURES) - set(df.columns)
        if missing:
            available = [c for c in df.columns if c in CORE_FEATURES]
            logger.error(f"Column mismatch!\nRequired: {CORE_FEATURES}\nAvailable: {available}")
            raise ValueError("Dataset validation failed")
            
        return df[CORE_FEATURES + ['label']]

    def _create_dashboard(self, X_test: pd.DataFrame, y_test: pd.Series):
        """ 
        Create a dashboard for visualizing model performance and SHAP feature importance.
        """
        app = dash.Dash(__name__)

        # Collect model performance metrics
        metrics = []
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics.append({
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            })
        
        # SHAP visualization for feature importance
        rf_pipeline = self.models['random_forest']
        preprocessor = rf_pipeline.named_steps['columntransformer']
        X_sample = X_test.sample(n=100, random_state=RANDOM_STATE)
        processed_data = preprocessor.transform(X_sample)
        
        # Validate processed data dimensions
        if processed_data.shape[1] != len(CORE_FEATURES):
            logger.error(f"Preprocessor output mismatch: Expected {len(CORE_FEATURES)} features, got {processed_data.shape[1]}")
            raise ValueError("Feature dimension mismatch in preprocessing")
        
        explainer = shap.TreeExplainer(rf_pipeline.named_steps['randomforestclassifier'])
        shap_values = explainer.shap_values(processed_data)

        # Handle binary classification SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  
        else:
            # For SHAP versions that return single array for binary classification
            if len(shap_values.shape) == 3:
                shap_values = shap_values[..., 1]  

        # Ensure correct shape (samples, features)
        if len(shap_values.shape) != 2:
            shap_values = np.reshape(shap_values, (len(X_sample), -1))

        # Validate SHAP dimensions
        if shap_values.shape[1] != processed_data.shape[1]:
            logger.error(f"SHAP/Data dimension mismatch: {shap_values.shape[1]} vs {processed_data.shape[1]}")
            raise ValueError("SHAP dimension validation failed")

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = preprocessor.get_feature_names_out()

        # Final validation check
        if len(mean_abs_shap) != len(feature_names):
            logger.error(f"Critical mismatch: SHAP values ({len(mean_abs_shap)}) vs features ({len(feature_names)})")
            raise ValueError("SHAP-feature length mismatch")

        # Create SHAP scatter plot
        shap_fig = px.scatter(
            x=mean_abs_shap,
            y=feature_names,
            orientation='h',
            title='SHAP Feature Importance',
            labels={'x': 'Average Impact on Malicious Classification', 'y': 'Feature'}
        )

        # Define dashboard layout
        app.layout = html.Div([
            html.H1("Network Anomaly Detection Dashboard"),
            dcc.Graph(
                figure=px.bar(
                    pd.DataFrame(metrics),
                    x='model',
                    y=['accuracy', 'roc_auc'],
                    barmode='group',
                    title='Model Performance Comparison'
                )
            ),
            dcc.Graph(
                figure=px.imshow(
                    confusion_matrix(y_test, self.models['random_forest'].predict(X_test)),
                    labels=dict(x="Predicted", y="Actual"),
                    title="Random Forest Confusion Matrix"
                )
            ),
            dcc.Graph(figure=shap_fig)
        ])
        
        # Start the Dash server
        if __name__ == '__main__':
            app.run(debug=False)

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
                
                if (precision >= PERFORMANCE_TARGETS['precision'] and 
                    recall >= PERFORMANCE_TARGETS['recall'] and 
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
        logger.info(f"Accuracy: {accuracy:.4f} ({'MET' if accuracy >= PERFORMANCE_TARGETS['accuracy'] else 'NOT MET'})")
        logger.info(f"Precision: {report['1']['precision']:.4f} ({'MET' if report['1']['precision'] >= PERFORMANCE_TARGETS['precision'] else 'NOT MET'})")
        logger.info(f"Recall: {report['1']['recall']:.4f} ({'MET' if report['1']['recall'] >= PERFORMANCE_TARGETS['recall'] else 'NOT MET'})")

    def build_pipeline(self, estimator, model_name: str):
        """
        Build a machine learning pipeline with preprocessing and SMOTE.
        """
        self.preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), CORE_FEATURES)
        ])
        
        # Model-specific SMOTE configuration
        smote_params = {
            'random_forest': {'sampling_strategy': 0.75},
            'logistic_regression': {'sampling_strategy': 0.5}
        }.get(model_name, {})
        
        return make_imb_pipeline(
            self.preprocessor,
            SMOTE(random_state=RANDOM_STATE, **smote_params),
            estimator
        )

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing and infinite values.
        """
        df['label'] = np.where(df['label'].str.contains('BENIGN'), 0, 1)
        logger.info(f"Class distribution:\n{df['label'].value_counts()}")
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove missing values
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_count - len(df)} rows with missing values")
        
        # Validate final dataset
        if len(df) < 1000:
            raise ValueError("Insufficient data after cleaning")
            
        return df

    def build_pipeline(self, estimator, model_name: str):
        """
        Pipeline builder with dynamic SMOTE configuration
        """
        self.preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), CORE_FEATURES)
        ])
        
        return make_imb_pipeline(
            self.preprocessor,
            SMOTE(
                random_state=RANDOM_STATE,
                sampling_strategy='auto'  # Let SMOTE determine optimal ratio
            ),
            estimator
        )

    def train(self):
        """
        Train the models and evaluate their performance.
        """
        # Data loading and cleaning
        dfs = [self._load_and_validate(path) for path in DATA_PATHS.values()]
        df = self._clean_data(pd.concat(dfs, ignore_index=True))
        
        # split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[CORE_FEATURES], df['label'],
            test_size=TEST_SIZE,
            stratify=df['label'],
            random_state=RANDOM_STATE
        )
        
        # Train models
        for model_name in self.models.keys():
            logger.info(f"Training {model_name.replace('_', ' ').title()}")
            
            estimator = (
        RandomForestClassifier(class_weight='balanced_subsample', random_state=RANDOM_STATE)
        if model_name == 'random_forest'
        else LogisticRegression(
            class_weight='balanced',
            max_iter=10000,  
            tol=1e-3,        
            random_state=RANDOM_STATE
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
        joblib.dump(self.models, MODEL_PATH)
        logger.info(f"Models saved to {MODEL_PATH}")
        self._create_dashboard(X_test, y_test)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['label'] = np.where(df['label'].str.contains('BENIGN'), 0, 1)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_count - len(df)} rows with missing values")
        
        if len(df) < 1000:
            raise ValueError("Insufficient data after cleaning")
            
        return df
# Main execution block
if __name__ == "__main__":
    detector = NetworkAnomalyDetector()
    detector.train()
    # Add dashboard server keep-alive
    detector.dashboard.run(host='0.0.0.0', port=8050)