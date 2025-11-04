import plotly.io as pio
import math
import numpy as np
import pandas as pd
def generate_dashboard(plots, dashboard_path="Dashboard.html", title="mlforgex Dashboard", metrics=None, arguments=None,model_comparison_df=None):
    '''
    Generates a comprehensive HTML dashboard for visualizing model performance and metrics.
    Args:
        plots (list): List of Plotly figure objects to include in the dashboard.
        dashboard_path (str): Path to save the generated HTML dashboard.
        title (str): Title of the dashboard.
        metrics (dict): Dictionary of key metrics to display.
        arguments (dict): Dictionary of model arguments and hyperparameters.
    
    Returns:
        None
    '''
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    :root {
        /* Premium Color Palette - Inspired by enterprise SaaS */
        --primary: #0066ff;
        --primary-dark: #0052cc;
        --primary-light: #3385ff;
        --secondary: #6b7280;
        --accent: #00c896;
        --warning: #f59e0b;
        --danger: #ef4444;
        --success: #10b981;
        --info: #06b6d4;
        
        /* Sophisticated Neutrals */
        --dark: #111827;
        --darker: #0a0f1c;
        --light: #ffffff;
        --lighter: #f9fafb;
        --border: #e5e7eb;
        --gray: #6b7280;
        --gray-light: #9ca3af;
        
        /* Premium Gradients */
        --gradient-primary: linear-gradient(135deg, var(--primary) 0%, #7c3aed 100%);
        --gradient-success: linear-gradient(135deg, var(--success) 0%, var(--accent) 100%);
        --gradient-header: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
        --gradient-card: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        
        /* Premium Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: var(--dark);
        line-height: 1.6;
        font-weight: 400;
        min-height: 100vh;
        background-attachment: fixed;
    }
    
    .dashboard-wrapper {
        min-height: 100vh;
        position: relative;
    }
    
    /* Premium Header */
    .premium-header {
        background: var(--gradient-header);
        color: white;
        padding: 60px 0 40px;
        margin-bottom: 50px;
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.03' fill-rule='evenodd'/%3E%3C/svg%3E");
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
        text-align: center;
        position: relative;
        z-index: 2;
    }
    
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .header-badge i {
        color: var(--accent);
    }
    
    .header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #ffffff 0%, #e0f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        max-width: 700px;
        margin: 0 auto 30px;
        line-height: 1.6;
    }
    
    .header-stats {
        display: flex;
        justify-content: center;
        gap: 40px;
        flex-wrap: wrap;
    }
    
    .stat-pill {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
    }
    
    /* Main Container */
    .dashboard-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Premium Grid Layout */
    .main-content {
        display: grid;
        grid-template-columns: 1fr 380px;
        gap: 40px;
        align-items: start;
    }
    
    /* Premium Cards */
    .premium-card {
        background: var(--gradient-card);
        border-radius: 16px;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border);
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .premium-card:hover {
        box-shadow: var(--shadow-2xl);
        transform: translateY(-5px);
    }
    
    .card-header {
        padding: 24px 28px;
        border-bottom: 1px solid var(--border);
        background: var(--lighter);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--dark);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .card-title i {
        color: var(--primary);
        font-size: 1.1em;
        width: 24px;
        text-align: center;
    }
    
    .card-content {
    padding:15px;
        display: flex;
    align-items: center;
    justify-content: center;
    }
    
    /* Enhanced Plots Grid */
    .plots-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
        gap: 30px;
    }
    
    .plot-card {
        min-height: 500px;
        display: flex;
        flex-direction: column;
        overflow:scroll;
    }
    
    .plot-container {
        flex: 1;
        padding: 0 20px 20px;
        min-height: 420px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .plot-container .js-plotly-plot {
        width: 100% !important;
        height: 100% !important;
        border-radius: 8px;
    }
    
    /* Premium Sidebar */
    .sidebar {
        flex-direction: column;
    display: flex;
    position: sticky;
    top: 30px;
    row-gap: 20px;
    }
    
    /* Enhanced Metrics Display */
    .metrics-grid {
        display: grid;
        gap: 16px;
        width: 90%;
    }
    .metric-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border-left: 4px solid var(--primary);
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
    border: 1px solid var(--border);
    min-height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    }

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
    gap: 12px;
}

.metric-info {
    flex: 1;
    min-width: 0; /* Important for text truncation */
}

.metric-name {
    font-size: 0.82rem;
    color: var(--gray);
    font-weight: 500;
    margin-bottom: 6px;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.metric-value {
    font-size: 1rem;
    font-weight: 800;
    color: var(--dark);
    line-height: 1.1;
    overflow-wrap:anywhere;
}

.metric-badge {
    padding: 4px 10px;
    border-radius: 16px;
    font-size: 0.7rem;
    font-weight: 700;
    background: var(--primary-light);
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-left: 8px;
}

.metric-badge.success { background: var(--success); }
.metric-badge.warning { background: var(--warning); }
.metric-badge.danger { background: var(--danger); }
.metric-badge.accent { background: var(--accent); }

.metric-trend {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--gray);
    margin-top: 6px;
    white-space: nowrap;
}

.trend-up { color: var(--success); }
.trend-down { color: var(--danger); }

/* Ensure the metrics grid has proper spacing */
.metrics-grid {
    display: grid;
    gap: 14px;
    grid-template-columns: 1fr;
}

/* Responsive adjustments */
@media (max-width: 1400px) {
    .metric-value {
        font-size: 1.3rem;
    }
    
    .metric-name {
        font-size: 0.8rem;
    }
}

@media (max-width: 768px) {
    .metric-header {
        flex-direction: column;
        gap: 8px;
    }
    
    .metric-badge {
        align-self: flex-start;
        margin-left: 0;
    }
    
    .metric-value {
        font-size: 1.4rem;
    }
}
    
    /* Enhanced Arguments Table */
    .args-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .args-table tr {
        border-bottom: 1px solid var(--border);
        transition: background-color 0.2s ease;
    }
    
    .args-table tr:hover {
        background-color: var(--lighter);
    }
    
    .args-table tr:last-child {
        border-bottom: none;
    }
    
    .args-table td {
        padding: 14px 12px;
        vertical-align: top;
    }
    
    .args-table td:first-child {
        font-weight: 600;
        color: var(--dark);
        width: 45%;
        font-size: 0.9rem;
    }
    
    .args-table td:last-child {
        color: var(--gray);
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Quality Indicators */
    .quality-score {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 20px;
        background: linear-gradient(135deg, var(--lighter) 0%, white 100%);
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    .score-circle {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: conic-gradient(var(--success) 0% 85%, var(--border) 85% 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
   
    .score-inner {
        width: 60px;
        height: 60px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.2rem;
        color: var(--dark);
    }
    
    .score-info {
        flex: 1;
    }
    
    .score-label {
        font-size: 0.9rem;
        color: var(--gray);
        margin-bottom: 4px;
    }
    
    .score-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--dark);
        margin-bottom: 4px;
    }
    
    .score-description {
        font-size: 0.8rem;
        color: var(--gray-light);
    }
    
    /* Premium Footer */
    .premium-footer {
        background: var(--dark);
        color: white;
        padding: 50px 0 30px;
        margin-top: 80px;
    }
    
    .footer-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
        display: grid;
        grid-template-columns: 2fr 1fr 1fr;
        gap: 60px;
    }
    
    .footer-brand {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    
    .footer-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
    }
    
    .logo-icon {
        width: 40px;
        height: 40px;
        background: var(--gradient-primary);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1rem;
        font-weight: bold;
    }
    
    .footer-description {
        color: var(--gray-light);
        line-height: 1.6;
        max-width: 400px;
    }
    
    .footer-links h4 {
        color: white;
        margin-bottom: 16px;
        font-size: 1.1rem;
        font-weight: 600;
        
    }
    
    .footer-links ul {
        list-style: none;
    }
    
    .footer-links li {
        margin-bottom: 8px;
    }
    
    .footer-links a {
        color: var(--gray-light);
        text-decoration: none;
        transition: color 0.2s ease;
    }
    
    .footer-links a:hover {
        color: white;
    }
    
    .footer-bottom {
        max-width: 1200px;
        margin: 0 auto;
        padding: 30px 40px 0;
        border-top: 1px solid #374151;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--gray-light);
        font-size: 0.9rem;
    }
    
    .footer-logo-img{
        height: 7vw;
    }
    /* Responsive Design */
    @media (max-width: 1200px) {
        .plots-grid {
            grid-template-columns: 1fr;
        }
        
        .main-content {
            grid-template-columns: 1fr;
            gap: 30px;
        }
        
        .sidebar {
            position: static;
        }
    }
    
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2.2rem;
        }
        
        .header-content {
            padding: 0 20px;
        }
        
        .footer-content {
            grid-template-columns: 1fr;
            gap: 40px;
        }
        
        .header-stats {
            gap: 20px;
        }
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--lighter);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    .model-table-container {
    overflow-x: auto;
}

.model-table {
    width: 100%;
    border-collapse: collapse;
}

.model-table th {
    background: #0066ff;
    color: white;
    padding: 12px;
    text-align: center;
    font-weight: 600;
}

.model-table td {
    padding: 10px;
    text-align: center;
    border-bottom: 1px solid #e5e7eb;
}

.model-table tr:hover {
    background-color: #f9fafb;
}
.premium-dashboard {
    overflow: scroll;
    scrollbar-width: none;
}

    
    </style>
    """

    # Calculate quality score based on metrics
    # ...existing code...

    
# ...existing code...
    dashboard_content = f"""
    <!DOCTYPE html>
    <html lang="en" class="premium-dashboard">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} | mlforgex Analytics</title>
        <link rel="icon" type="image/png" href="https://res.cloudinary.com/datrajdgv/image/upload/v1759823370/logo_vmruwp.png"/>
        {css}
    </head>
    <body>
        <div class="dashboard-wrapper">
            <header class="premium-header">
                <div class="header-content">
                    <div class="header-badge">
                        <i class="fas fa-shield-check"></i>
                        Enterprise-Grade Analysis
                    </div>
                    <h1>{title}</h1>
                    <div class="header-subtitle">
                        Comprehensive machine learning model performance analysis with enterprise-level insights and actionable intelligence.
                    </div>
                    <div class="header-stats">
                        <div class="stat-pill">
                            <div class="stat-value">{len(plots)}</div>
                            <div class="stat-label">Visualizations</div>
                        </div>
                        <div class="stat-pill">
                            <div class="stat-value">{len(metrics) if metrics else 0}</div>
                            <div class="stat-label">Metrics Tracked</div>
                        </div>
                        <div class="stat-pill">
                            <div class="stat-value">{len(arguments) if arguments else 0}</div>
                            <div class="stat-label">Parameters</div>
                        </div>
                    </div>
                </div>
            </header>

            <div class="dashboard-container">
                <div class="main-content">
                    <div class="plots-section">
                        <div class="plots-grid">
    """

    # Generate premium plot cards
    for plot_title, fig in plots:
        plot_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, config={
            'displayModeBar': True,
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': plot_title.lower().replace(' ', '_'),
                'height': 500,
                'width': 700,
                'scale': 1
            }
        })
        
        dashboard_content += f"""
                            <div class="premium-card plot-card">
                                <div class="card-header">
                                    <div class="card-title">
                                        <i class="fas fa-chart-line"></i>
                                        {plot_title}
                                    </div>
                                </div>
                                <div class="plot-container">
                                    {plot_html}
                                </div>
                            </div>
        """

    dashboard_content += """
                        </div>
                    </div>

                    <div class="sidebar">
    """
    if metrics:
        dashboard_content += """
                        <div class="premium-card">
                            <div class="card-header">
                                <div class="card-title">
                                    <i class="fas fa-tachometer-alt"></i>
                                    Performance Metrics
                                </div>
                            </div>
                            <div class="card-content">
                                <div class="metrics-grid">
        """
        
        for key, value in metrics.items():
            # Determine metric styling and trend
            status_class = ""
            badge_text = ""
            trend_html = ""
            
            if isinstance(value, (int, float, np.integer, np.floating)):
                if key in ['Test accuracy', 'Test F1', 'Test precision', 'Test recall', 'Test rocauc','Train accuracy', 'Train F1', 'Train precision', 'Train recall', 'Train rocauc',"Train R2","Test R2"]:
                    if value > 0.9:
                        status_class = "success"
                        badge_text = "EXCELLENT"
                        trend_html = '<div class="metric-trend trend-up"><i class="fas fa-arrow-up"></i> Optimal</div>'
                    elif value > 0.8:
                        status_class = "accent"
                        badge_text = "GOOD"
                        trend_html = '<div class="metric-trend trend-up"><i class="fas fa-arrow-up"></i> Strong</div>'
                    elif value > 0.7:
                        status_class = "warning"
                        badge_text = "FAIR"
                        trend_html = '<div class="metric-trend"><i class="fas fa-minus"></i> Adequate</div>'
                    else:
                        status_class = "danger"
                        badge_text = "REVIEW"
                        trend_html = '<div class="metric-trend trend-down"><i class="fas fa-arrow-down"></i> Needs Work</div>'
                    
                    value_display = f"{value:.3f}"
                else:
                    value_display = str(value)
                    status_class = "accent"
                    badge_text = "INFO"
            else:
                value_display = str(value)
                status_class = "accent"
                badge_text = "INFO"
            
            # Truncate long metric names and values
            truncated_key = (key[:43] + '...') if len(key) > 43 else key
            truncated_value = (value_display[:20] + '...') if len(value_display) > 20 else value_display
            dashboard_content += f"""
                                    <div class="metric-card {status_class}">
                                        <div class="metric-header">
                                            <div class="metric-info">
                                                <div class="metric-name" title="{key}">{truncated_key}</div>
                                                <div class="metric-value" title="{value_display}">{truncated_value}</div>
                                            </div>
                                            <div class="metric-badge {status_class}">{badge_text}</div>
                                        </div>
                                        {trend_html}
                                    </div>
            """
        
        dashboard_content += """
                                </div>
                            </div>
                        </div>
        """

    # Enhanced Configuration Section
    if arguments:
        dashboard_content += """
                        <div class="premium-card">
                            <div class="card-header">
                                <div class="card-title">
                                    <i class="fas fa-cogs"></i>
                                    Model Configuration
                                </div>
                            </div>
                            <div class="card-content">
                                <table class="args-table">
        """
        
        for key, value in arguments.items():
            dashboard_content += f"""
                                    <tr>
                                        <td>{key}</td>
                                        <td>{value}</td>
                                    </tr>
            """
        
        dashboard_content += """
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
        """
    if "total_accuracy" in model_comparison_df.columns:
        model_comparison_df.drop(columns=['total_accuracy', 'total_f1', 'Norm F1', 'Norm Accuracy', 'Combined Score'], inplace=True, errors='ignore')
    else:
        model_comparison_df.drop(columns=['total_r2', 'total_rmse', 'Norm R2', 'Norm RMSE', 'Combined Score'], inplace=True, errors='ignore')
    columns = model_comparison_df.columns.tolist()

# Start building HTML table
    html_content = """
    <div class="model-comparison-section" style="margin-top: 20px;">
        <div class="premium-card" style="background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid #e5e7eb;">
            <div class="card-header" style="padding: 20px 24px; border-bottom: 1px solid #e5e7eb; background: linear-gradient(135deg, #1e293b 0%, #334155 100%);">
                <div class="card-title" style="font-size: 1.25rem; font-weight: 600; color: white; display: flex; align-items: center; gap: 12px;">
                    <i class="fas fa-chart-line" style="color: #00c896;"></i>
                    Model Performance Comparison
                </div>
            </div>
            <div class="card-content" style="padding: 0;">
                <div class="model-table-container" style="overflow-x: auto;width: 100%;">
                    <table class="model-table" style="width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #0066ff 0%, #0052cc 100%);">
    """
        
    for col in columns:
        html_content += f'''
            <th style="padding: 16px 12px; color: white; font-weight: 600; font-size: 0.85rem; text-align: center; border-right: 1px solid rgba(255,255,255,0.1); text-transform: uppercase; letter-spacing: 0.5px;">
                {col.replace('_', ' ').title()}
            </th>'''

    html_content += """
                            </tr>
                        </thead>
                        <tbody>
    """

    # Add table rows with alternating colors
    for idx, (_, row) in enumerate(model_comparison_df.iterrows()):
        row_bg = "background: #f8fafc;" if idx % 2 == 0 else "background: white;"
        html_content += f'<tr style="{row_bg} transition: all 0.2s ease;">'
        
        for col in columns:
            value = row[col]
            
            # Handle different value types with styling
            if col.lower() == 'tuned':
                badge_color = "#10b981" if value==1 else "#6b7280"
                badge_text = "True" if value==1 else "False"
                html_content += f'''
                <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid #e5e7eb;">
                    <span style="background: {badge_color}; color: white; padding: 4px 8px; border-radius: 8px; font-size: 0.75rem; font-weight: 600;">
                        {badge_text}
                    </span>
                </td>'''
            elif pd.isna(value):
                html_content += '<td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #9ca3af; font-weight: 500;">-</td>'
            elif isinstance(value, (int, float)):
                # Color code based on value ranges for metrics
                value_style = "color: #111827; font-weight: 600;"
                if 'accuracy' in col.lower() or 'f1' in col.lower() or 'precision' in col.lower() or 'recall' in col.lower() or "rocauc" in col.lower() or 'r2' in col.lower():
                    if value > 0.8:
                        value_style = "color: #10b981; font-weight: 700;"
                    elif value > 0.6:
                        value_style = "color: #f59e0b; font-weight: 600;"
                    else:
                        value_style = "color: #ef4444; font-weight: 600;"
                else :
                  value_style = "color: #111827; font-weight: 500;"          
                html_content += f'<td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid #e5e7eb; {value_style}">{value:.4f}</td>'
            else:
                html_content += f'<td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid #e5e7eb; color: #111827; font-weight: 500;">{value}</td>'
        
        html_content += '</tr>'

    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    """.format(
        model_count=len(model_comparison_df),
        metric_count=len(columns)
    )
    dashboard_content += html_content
    dashboard_content += """
            </div>

            <footer class="premium-footer">
                <div class="footer-content">
                    <div class="footer-brand">
                        <div class="footer-logo">
                            <img class="footer-logo-img" src="https://res.cloudinary.com/datrajdgv/image/upload/v1759823370/logo_vmruwp.png" alt="Logo">
                            mlforgex Analytics
                        </div>
                        <p class="footer-description">
                            Trusted by data science teams worldwide for reliable, scalable, and professional machine learning model analysis and monitoring.
                        </p>
                    </div>
                    <div class="footer-links" style="justify-content: center;display: flex;align-items: center;">
                        <ul>
                            <li><i class="fa-solid fa-book" style="color: #ffffff;"></i><a target="_blank" href="https://dhgefergfefruiwefhjhcduc.github.io/mlforgex_documentation/"> Documentation</a></li>
                            <li><i class="fa-solid fa-arrow-up-right-from-square" style="color: #ffffff;"></i><a target="_blank" href="https://pypi.org/project/mlforgex/"> pypi package</a></li>

                        </ul>
                    </div>
                    <div class="footer-links" style="justify-content: flex-end;display: flex;align-items: center;">
                        <ul>
                            <li><i class="fa-brands fa-github" style="color: #ffffff;"></i><a target="_blank" href="https://github.com/dhgefergfefruiwefhjhcduc"> Github</a></li>
                            <li><i class="fa-solid fa-envelope" style="color: #ffffff;"></i><a target="_blank" href="mailto:mathurpriyansh2006@gmail.com"> Support</a></li>
                        </ul>
                    </div>
                </div>
                <div class="footer-bottom">
                    <div>© 2025 mlforgex. All rights reserved.</div>
                    <div>Generated on {date}</div>
                </div>
            </footer>
        </div>
    </body>
    </html>
    """.format(date=__import__('datetime').datetime.now().strftime('%B %d, %Y at %H:%M'))

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_content)
    print(f"Dashboard saved to {dashboard_path}")

