import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from sklearn.model_selection import learning_curve
from wordcloud import WordCloud, STOPWORDS



def plot_classification_metrics(model, X_train, y_train, X_test, y_test, class_names=None):
    '''
    Plots various classification metrics including Confusion Matrix, ROC Curve,
    Precision-Recall Curve, Learning Curve, and Class Distribution.
    Args:
        model: Trained classification model with predict and predict_proba methods.
        X_train (pd.DataFrame or np.ndarray): Training feature set.
        y_train (pd.Series or np.ndarray): Training target labels.
        X_test (pd.DataFrame or np.ndarray): Testing feature set.
        y_test (pd.Series or np.ndarray): Testing target labels.
        class_names (List[str], optional): List of class names for labeling plots. Defaults to None.
    Returns:
        List[Tuple[str, go.Figure]]: List of plot tuples for dashboard
    '''
    plots=[]
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    unique_classes = model.classes_
        
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_percentage,
        x=unique_classes,
        y=unique_classes,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Percentage: %{z}<extra></extra>',
        showscale=True
    ))
    
    # Add annotations (text in each cell)
    for i in range(len(unique_classes)):
        for j in range(len(unique_classes)):
            percentage = cm_percentage[i, j]
            count = cm[i, j]
            text = f"{percentage:.1f}%\n({count})"
            fig_cm.add_annotation(
                x=unique_classes[j],
                y=unique_classes[i],
                text=text,
                showarrow=False,
                font=dict(
                color='white' if percentage > 50 else 'black',
                size=12
            )
    )

    fig_cm.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        template='plotly_white',
        width=600,
        height=600
    )
    
    plots.append(("Confusion Matrix", fig_cm))

    
    # 2. ROC Curve (for binary classification)
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        
        # ROC Curve
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template='plotly_white',
            width=600,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        plots.append(("ROC Curve", fig_roc))
    
    # 3. Precision-Recall Curve (for binary classification)
    if y_proba is not None and len(unique_classes) == 2:
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        fig_pr = go.Figure()
        
        fig_pr.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'Precision-Recall Curve (AUC = {pr_auc:.3f})',
            line=dict(color='green', width=3),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
        
        # Add no-skill line (precision = proportion of positive class)
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        fig_pr.add_trace(go.Scatter(
            x=[0, 1],
            y=[no_skill, no_skill],
            mode='lines',
            name='No Skill',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
        
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            template='plotly_white',
            width=600,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.02)
        )
        
        plots.append(("Precision-Recall Curve", fig_pr))
    
    # 4. Learning Curve
    cv = min(5, np.min(np.bincount(y_train)))
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='accuracy'
    )
    
    fig_learning = go.Figure()
    
    # Training scores with confidence interval
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    fig_learning.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='Training Size: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
    ))
    
    # Add training confidence interval
    fig_learning.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Training ±1 STD',
        showlegend=True
    ))
    
    # Validation scores with confidence interval
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig_learning.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='red', width=3),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='Training Size: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
    ))
    
    # Add validation confidence interval
    fig_learning.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Validation ±1 STD',
        showlegend=True
    ))
    
    fig_learning.update_layout(
        xaxis_title="Training Set Size",
        yaxis_title="Accuracy",
        hovermode='x unified',
        template='plotly_white',
        width=800,
        height=500,
        legend=dict(x=0.02, y=0.02)
    )
 
    plots.append(("Learning Curve", fig_learning))

    # 5. Class Distribution
    fig_distribution = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Class Distribution", "Testing Class Distribution")
    )
    
    # Training distribution
    train_counts = pd.Series(y_train).value_counts().sort_index()
    fig_distribution.add_trace(go.Bar(
        x=[str(cls) for cls in train_counts.index],
        y=train_counts.values,
        name='Training',
        marker_color='lightblue',
        hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)
    
    # Testing distribution
    test_counts = pd.Series(y_test).value_counts().sort_index()
    fig_distribution.add_trace(go.Bar(
        x=[str(cls) for cls in test_counts.index],
        y=test_counts.values,
        name='Testing',
        marker_color='lightcoral',
        hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>'
    ), row=1, col=2)
    
    fig_distribution.update_layout(
        template='plotly_white',
        width=800,
        height=400,
        showlegend=False
    )
    
    fig_distribution.update_xaxes(title_text="Class", row=1, col=1)
    fig_distribution.update_xaxes(title_text="Class", row=1, col=2)
    fig_distribution.update_yaxes(title_text="Count", row=1, col=1)
    fig_distribution.update_yaxes(title_text="Count", row=1, col=2)
   
    plots.append(("Class Distribution", fig_distribution))
    # 6. Additional: Classification Report Heatmap
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :]  # Remove averages
    
    fig_report = go.Figure(data=go.Heatmap(
        z=report_df.values,
        x=report_df.columns,
        y=report_df.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Metric: %{x}<br>Class: %{y}<br>Value: %{z:.3f}<extra></extra>',
        showscale=True
    ))
    
    # Add annotations
    for i in range(len(report_df.index)):
        for j in range(len(report_df.columns)):
            fig_report.add_annotation(
                x=report_df.columns[j],
                y=report_df.index[i],
                text=f'{report_df.iloc[i, j]:.3f}',
                showarrow=False,
                font=dict(color='white' if report_df.iloc[i, j] > 0.5 else 'black')
            )
    
    fig_report.update_layout(
        template='plotly_white',
        width=600,
        height=400
    )
   
    plots.append(("Classification Report", fig_report))
    return plots


def plot_regression_metrics(model, X_train, y_train, X_test, y_test, feature_names):
    '''
    Plots various regression metrics including Actual vs Predicted, Residual Plot,
    Distribution of Residuals, and Learning Curve.
    Args:
        model: Trained regression model with predict method.
        X_train (pd.DataFrame or np.ndarray): Training feature set.
        y_train (pd.Series or np.ndarray): Training target values.
        X_test (pd.DataFrame or np.ndarray): Testing feature set.
        y_test (pd.Series or np.ndarray): Testing target values.
        feature_names (List[str]): List of feature names for labeling plots.
    Returns:
        List[Tuple[str, go.Figure]]: List of plot tuples for dashboard
    '''
    plots=[]
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
        
    # 1. Actual vs Predicted Plot
    fig_actual_vs_predicted = go.Figure()
    
    # Add scatter points
    fig_actual_vs_predicted.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(
            color='blue',
            opacity=0.7,
            size=8,
            line=dict(width=1, color='darkblue')
        ),
        name='Predictions',
        hovertemplate='<b>Actual</b>: %{x:.2f}<br><b>Predicted</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_actual_vs_predicted.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=3),
        name='Perfect Prediction'
    ))
    
    fig_actual_vs_predicted.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        showlegend=True,
        template='plotly_white',
        width=800,
        height=600
    )
   
    plots.append(("Actual vs Predicted", fig_actual_vs_predicted))
    # 2. Residual Plot
    fig_residual = go.Figure()
    
    fig_residual.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            color='green',
            opacity=0.7,
            size=8,
            line=dict(width=1, color='darkgreen')
        ),
        name='Residuals',
        hovertemplate='<b>Predicted</b>: %{x:.2f}<br><b>Residual</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig_residual.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text="Zero Residual Line",
        annotation_position="bottom right"
    )
    
    fig_residual.update_layout(
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        showlegend=True,
        template='plotly_white',
        width=800,
        height=600
    )
   
    plots.append(("Residual Plot", fig_residual))
    
    # 3. Distribution of Residuals
    fig_residual_dist = go.Figure()
    
    # Histogram
    fig_residual_dist.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Residuals',
        opacity=0.7,
        marker_color='lightblue',
        hovertemplate='<b>Residual Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
    ))
    
    # Add KDE line
    hist_data = np.histogram(residuals, bins=30, density=True)
    kde_x = np.linspace(residuals.min(), residuals.max(), 100)
    kde_y = np.exp(-0.5 * ((kde_x - residuals.mean()) / residuals.std()) ** 2) / (residuals.std() * np.sqrt(2 * np.pi))
    
    fig_residual_dist.add_trace(go.Scatter(
        x=kde_x,
        y=kde_y * len(residuals) * (hist_data[1][1] - hist_data[1][0]),  # Scale to match histogram
        mode='lines',
        name='Density',
        line=dict(color='red', width=3),
        yaxis='y1'
    ))
    
    fig_residual_dist.update_layout(
        xaxis_title="Residual Value",
        yaxis_title="Frequency",
        showlegend=True,
        template='plotly_white',
        width=800,
        height=600,
        bargap=0.1
    )
   
    plots.append(("Residual Distribution", fig_residual_dist))
    # 4. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='r2'
    )
    
    fig_learning_curve = go.Figure()
    
    # Training scores with confidence interval
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    fig_learning_curve.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Training Size</b>: %{x}<br><b>R² Score</b>: %{y:.3f}<extra></extra>'
    ))
    
    # Add training confidence interval
    fig_learning_curve.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Training ±1 STD',
        showlegend=True
    ))
    
    # Validation scores with confidence interval
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig_learning_curve.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red', width=3),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>Training Size</b>: %{x}<br><b>R² Score</b>: %{y:.3f}<extra></extra>'
    ))
    
    # Add validation confidence interval
    fig_learning_curve.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Validation ±1 STD',
        showlegend=True
    ))
    
    fig_learning_curve.update_layout(
        xaxis_title="Training Set Size",
        yaxis_title="R² Score",
        legend=dict(
            x=0.02,
            y=0.02,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='x unified',
        template='plotly_white',
        width=800,
        height=600
    )
   
    plots.append(("Learning Curve", fig_learning_curve))

    # 5. Additional: Prediction Error Distribution
    fig_error_dist = go.Figure()
    
    absolute_errors = np.abs(residuals)
    
    fig_error_dist.add_trace(go.Box(
        y=absolute_errors,
        name='Absolute Errors',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color='lightcoral',
        line_color='darkred'
    ))
    
    fig_error_dist.update_layout(
        yaxis_title="Absolute Error |Actual - Predicted|",
        template='plotly_white',
        width=800,
        height=600
    )
   
    plots.append(("Error Distribution", fig_error_dist))
    return plots
    

def feature_importance(model, feature_names):
    """
    Plot feature importances using Plotly
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
    Returns:
        List[Tuple[str, go.Figure]]: List of plot tuples for dashboard
    """
    plots=[]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Clean feature names (same logic as original)
        feature_clean_name = []
        for i in feature_names:
            if i.split("__")[0] == "StandardScaler" or i.split("__")[0] == "OrdinalEncoder":
                feature_clean_name.append(i.split("__")[1])
            elif i.split("__")[0] == "OneHotEncoder":
                category, value = i.split("__")[1].rsplit("_", 1)
                feature_clean_name.append(f"{category} : {value}")
            else:
                feature_clean_name.append(i)

        # Convert to percentages and sort
        importances_percent = 100 * (importances / importances.sum())
        indices = np.argsort(importances_percent)[::-1]  # Sort in descending order
        
        # Get sorted features and importances
        sorted_features = [feature_clean_name[i] for i in indices]
        sorted_importances = importances_percent[indices]
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add horizontal bars
        fig.add_trace(go.Bar(
            y=sorted_features,
            x=sorted_importances,
            orientation='h',
            marker=dict(
                color=sorted_importances,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(
                    title="Importance %",
                    title_side="right"
                )
            ),
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Importance: %{x:.2f}%<br>" +
                "<extra></extra>"
            ),
            text=[f'{imp:.1f}%' for imp in sorted_importances],
            textposition='auto',
        ))
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                title="Global Importance (%)",
                range=[0, max(sorted_importances) * 1.1],  # Leave room for labels
                gridcolor='lightgray',
                zerolinecolor='lightgray'
            ),
            yaxis=dict(
                title="Features",
                tickfont=dict(size=10),
                autorange="reversed"  # Highest importance at top
            ),
            template='plotly_white',
            height=max(400, len(sorted_features) * 30),  # Dynamic height based on number of features
            width=900,
            margin=dict(l=10, r=10, t=80, b=20),
            showlegend=False
        )
        
        plots.append(("Feature Importances", fig))
    return plots


def create_cloud(df):
    """
    Generate and plot a word cloud from text data in a DataFrame using Plotly.
    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column with text data.
    Returns:
        List[Tuple[str, go.Figure]]: List of plot tuples for dashboard
    """
    plots=[]

    # Prepare text data
    text_data = " ".join(df["text"].astype(str).tolist())
    stopwords = set(STOPWORDS)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,  # Higher resolution for better quality
        height=600,
        background_color="black",
        stopwords=stopwords,
        colormap="viridis",
        max_words=200,
        relative_scaling=0.5,
        random_state=42
    ).generate(text_data)
    
    # Convert word cloud to image array
    wordcloud_image = wordcloud.to_array()
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add the word cloud as an image
    fig.add_trace(go.Image(
        z=wordcloud_image,
        hoverinfo='skip'  # No hover info for image
    ))
    
    # Update layout for better appearance
    fig.update_layout(
        xaxis=dict(
            visible=False,
            range=[0, wordcloud_image.shape[1]]
        ),
        yaxis=dict(
            visible=False,
            range=[0, wordcloud_image.shape[0]],
            autorange='reversed'  # Match image coordinate system
        ),
        template='plotly_white',
        width=1000,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='black',
        paper_bgcolor='white'
    )
    
    # Remove axes completely
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    plots.append(("Word Cloud", fig))
    return plots

def plot_data_analysis(df, target_column=None):
    """
    Generate comprehensive data analysis plots to understand the dataset.
    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of target column (optional)
        numerical_features (List[str]): List of numerical feature names
        categorical_features (List[str]): List of categorical feature names
    Returns:
        List[Tuple[str, go.Figure]]: List of plot tuples for dashboard
    """
    plots = []
    numerical_features=[i for i in df.columns if df[i].dtype != 'object' and i!=target_column]
    categorical_features=[i for i in df.columns if df[i].dtype == 'object' and i!=target_column]
    # 1. Dataset Overview (Summary Statistics)
    fig_overview = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#0066ff',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                ['Dataset Shape', 'Missing Values %', 'Duplicate Rows', 'Memory Usage (MB)', 'Features', 'Rows'],
                [
                    f'{df.shape[0]} rows × {df.shape[1]} cols',
                    f'{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%',
                    f'{df.duplicated().sum()}',
                    f'{df.memory_usage(deep=True).sum() / 1024**2:.2f}',
                    f'{df.shape[1]}',
                    f'{df.shape[0]}'
                ]
            ],
            fill_color='#f9fafb',
            align='center',
            font=dict(size=11)
        )
    )])
    
    fig_overview.update_layout(
        height=300,
        width=800,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    plots.append(("Dataset Overview", fig_overview))
    
    # 2. Missing Values Heatmap
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig_missing = go.Figure(data=go.Bar(
            x=missing_data[missing_data > 0].index,
            y=missing_data[missing_data > 0].values,
            marker=dict(
                color=missing_data[missing_data > 0].values,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            hovertemplate='<b>%{x}</b><br>Missing: %{y}<extra></extra>',
            text=missing_data[missing_data > 0].values,
            textposition='auto'
        ))
        
        fig_missing.update_layout(
            xaxis_title="Features",
            yaxis_title="Count",
            template='plotly_white',
            height=400,
            width=900
        )
        plots.append(("Missing values per feature", fig_missing))
    
    # 3. Numerical Features Distribution
    if numerical_features:
        fig_numerical = make_subplots(
            rows=(len(numerical_features) + 1) // 2,
            cols=2,
            subplot_titles=numerical_features,
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}]] * ((len(numerical_features) + 1) // 2)
        )
        
        for idx, feature in enumerate(numerical_features[:10]):  # Limit to first 10
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig_numerical.add_trace(
                go.Histogram(
                    x=df[feature].dropna(),
                    nbinsx=30,
                    name=feature,
                    marker_color='lightblue',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_numerical.update_layout(
            template='plotly_white',
            height=300 * ((len(numerical_features) + 1) // 2),
            width=1000,
            showlegend=False
        )
        plots.append(("Numerical Features Distribution", fig_numerical))
    
    # 4. Categorical Features Distribution
    if categorical_features:
        fig_categorical = make_subplots(
            rows=(len(categorical_features) + 1) // 2,
            cols=2,
            subplot_titles=categorical_features,
            specs=[[{'type': 'bar'}, {'type': 'bar'}]] * ((len(categorical_features) + 1) // 2)
        )
        
        for idx, feature in enumerate(categorical_features[:10]):  # Limit to first 10
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            value_counts = df[feature].value_counts().head(10)
            fig_categorical.add_trace(
                go.Bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    name=feature,
                    marker_color='lightcoral',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_categorical.update_layout(
            template='plotly_white',
            height=300 * ((len(categorical_features) + 1) // 2),
            width=1000,
            showlegend=False
        )
        plots.append(("Categorical Features Distribution (Top 10 Values)", fig_categorical))
             
    # 5. Target Variable Analysis (if provided)
    if target_column and target_column in df.columns:
        if df[target_column].dtype in ['object', 'category']:
            # Categorical target
            target_counts = df[target_column].value_counts()
            fig_target = go.Figure(data=[
                go.Bar(
                    x=target_counts.index.astype(str),
                    y=target_counts.values,
                    marker=dict(
                        color=target_counts.values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=target_counts.values,
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )
            ])
            fig_target.update_layout(
                xaxis_title="Class",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                width=800
            )
        else:
            # Numerical target
            fig_target = go.Figure(data=[
                go.Histogram(
                    x=df[target_column].dropna(),
                    nbinsx=50,
                    marker_color='lightgreen',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )
            ])
            fig_target.update_layout(
                xaxis_title="Value",
                yaxis_title="Frequency",
                template='plotly_white',
                height=500,
                width=800
            )
        
        plots.append((f"Target Variable Distribution: {target_column}", fig_target))
    
    # 6. Data Type Summary
    dtype_counts = df.dtypes.value_counts()
    fig_dtypes = go.Figure(data=[
        go.Pie(
            labels=['Numerical', 'Categorical', 'Other'],
            values=[
                len(df.select_dtypes(include=['number']).columns),
                len(df.select_dtypes(include=['object', 'category']).columns),
                len(df.select_dtypes(exclude=['number', 'object', 'category']).columns)
            ],
            marker=dict(colors=['#3385ff', '#f59e0b', '#ef4444']),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        )
    ])
    fig_dtypes.update_layout(
        template='plotly_white',
        height=500,
        width=600
    )
    plots.append(("Data Type Distribution", fig_dtypes))
    
    # 7. Skewness and Kurtosis for Numerical Features
    if numerical_features:
        from scipy.stats import skew, kurtosis
        
        skewness_data = []
        kurtosis_data = []
        features_list = []
        
        for feature in numerical_features[:15]:  # Limit to first 15
            skewness_data.append(skew(df[feature].dropna()))
            kurtosis_data.append(kurtosis(df[feature].dropna()))
            features_list.append(feature)
        
        fig_skew = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Skewness", "Kurtosis")
        )
        
        fig_skew.add_trace(
            go.Bar(
                x=features_list,
                y=skewness_data,
                marker_color='lightblue',
                name='Skewness',
                hovertemplate='<b>%{x}</b><br>Skewness: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig_skew.add_trace(
            go.Bar(
                x=features_list,
                y=kurtosis_data,
                marker_color='lightcoral',
                name='Kurtosis',
                hovertemplate='<b>%{x}</b><br>Kurtosis: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig_skew.update_layout(
            template='plotly_white',
            height=400,
            width=1000,
            showlegend=False
        )
        
        plots.append(("Skewness & Kurtosis Analysis", fig_skew))
    
    return plots