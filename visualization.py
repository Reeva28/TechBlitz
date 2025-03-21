import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap.
    
    Args:
        df: pandas.DataFrame containing the data
        
    Returns:
        matplotlib.figure.Figure: Figure containing the heatmap
    """
    # Calculate correlation
    correlation = df.corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        correlation, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm',
        square=True,
        linewidths=0.5
    )
    
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_distribution(df, feature, target):
    """
    Plot feature distribution colored by target.
    
    Args:
        df: pandas.DataFrame containing the data
        feature: Feature to plot
        target: Target variable
        
    Returns:
        plotly.graph_objects.Figure: Interactive histogram
    """
    # Create histogram
    fig = px.histogram(
        df, 
        x=feature,
        color=target,
        marginal="box",
        opacity=0.7,
        color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
        title=f"Distribution of {feature} by {target}"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        barmode='overlay'
    )
    
    return fig

def plot_feature_importance(importance_df):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        
    Returns:
        matplotlib.figure.Figure: Bar plot of feature importance
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bar chart
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df,
        palette='viridis'
    )
    
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_scatter_matrix(df, features, color_by):
    """
    Plot scatter matrix.
    
    Args:
        df: pandas.DataFrame containing the data
        features: List of features to include
        color_by: Column to color points by
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter matrix
    """
    # Create scatter matrix
    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color=color_by,
        color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        title=f"Scatter Matrix colored by {color_by}",
        height=600
    )
    
    return fig

def plot_boxplots_by_target(df, features, target):
    """
    Plot box plots for features grouped by target.
    
    Args:
        df: pandas.DataFrame containing the data
        features: List of features to plot
        target: Target variable
        
    Returns:
        matplotlib.figure.Figure: Figure with box plots
    """
    # Calculate number of rows needed
    n_rows = (len(features) + 2) // 3
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    
    # Flatten axes if needed
    if n_rows > 1:
        axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(features):
        if n_rows == 1:
            ax = axes[i]
        else:
            ax = axes[i]
            
        sns.boxplot(
            x=target, 
            y=feature, 
            data=df,
            palette={True: '#FF6B6B', False: '#4ECDC4'},
            ax=ax
        )
        
        ax.set_title(f"{feature} by {target}")
        ax.set_xlabel(target)
        ax.set_ylabel(feature)
    
    # Hide empty subplots
    if len(features) < len(axes):
        for i in range(len(features), len(axes)):
            if n_rows == 1:
                axes[i].set_visible(False)
            else:
                axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def plot_actual_vs_predicted(results_df, features):
    """
    Plot actual vs predicted values with additional features.
    
    Args:
        results_df: DataFrame with actual and predicted values
        features: Features to use for additional dimensions
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot
    """
    # Create figure
    if len(features) >= 1:
        # Use first feature for size
        size_feature = features[0]
        
        # Use second feature for color if available, otherwise use actual values
        color_feature = features[1] if len(features) >= 2 else "Actual"
        
        fig = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            size=size_feature,
            color=color_feature,
            hover_data=["Actual", "Predicted"] + list(features),
            title="Actual vs Predicted Values"
        )
    else:
        # Simple scatter plot if no features provided
        fig = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            hover_data=["Actual", "Predicted"],
            title="Actual vs Predicted Values"
        )
    
    # Add diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction"
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        xaxis=dict(range=[-0.1, 1.1]),
        yaxis=dict(range=[-0.1, 1.1])
    )
    
    return fig
