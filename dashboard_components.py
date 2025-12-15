"""
Dashboard Components for GSFL Visualization
Reusable Plotly chart components for the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_accuracy_loss_chart(metrics, title="Training Progress"):
    """Create dual-axis accuracy and loss chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    rounds = list(range(1, len(metrics.get("accuracy", [])) + 1))
    
    # Accuracy line
    fig.add_trace(
        go.Scatter(
            x=rounds, 
            y=[a * 100 for a in metrics.get("accuracy", [])],
            name="Accuracy",
            line=dict(color="#00d4aa", width=3),
            mode="lines+markers",
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    # Loss line
    fig.add_trace(
        go.Scatter(
            x=rounds, 
            y=metrics.get("loss", []),
            name="Loss",
            line=dict(color="#ff6b6b", width=3),
            mode="lines+markers",
            marker=dict(size=8)
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="white")),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    fig.update_xaxes(title_text="Round", gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Accuracy (%)", secondary_y=False, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Loss", secondary_y=True, gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_comparison_chart(sl_metrics, gsfl_metrics, metric_key, title, y_label):
    """Create comparison chart between SL and GSFL."""
    fig = go.Figure()
    
    sl_rounds = list(range(1, len(sl_metrics.get(metric_key, [])) + 1))
    gsfl_rounds = list(range(1, len(gsfl_metrics.get(metric_key, [])) + 1))
    
    # Multiply accuracy by 100 for percentage
    sl_values = sl_metrics.get(metric_key, [])
    gsfl_values = gsfl_metrics.get(metric_key, [])
    
    if metric_key == "accuracy":
        sl_values = [v * 100 for v in sl_values]
        gsfl_values = [v * 100 for v in gsfl_values]
    
    fig.add_trace(go.Scatter(
        x=sl_rounds, y=sl_values,
        name="Split Learning (SL)",
        line=dict(color="#ffa726", width=3),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle")
    ))
    
    fig.add_trace(go.Scatter(
        x=gsfl_rounds, y=gsfl_values,
        name="GSFL",
        line=dict(color="#42a5f5", width=3),
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond")
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        xaxis_title="Round",
        yaxis_title=y_label,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=350
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_latency_breakdown(metrics, title="Latency Breakdown"):
    """Create stacked bar chart for latency components."""
    rounds = list(range(1, len(metrics.get("uplink", [])) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Uplink",
        x=rounds,
        y=metrics.get("uplink", []),
        marker_color="#4ecdc4"
    ))
    
    fig.add_trace(go.Bar(
        name="Downlink", 
        x=rounds,
        y=metrics.get("downlink", []),
        marker_color="#ff6b6b"
    ))
    
    fig.add_trace(go.Bar(
        name="Compute",
        x=rounds,
        y=metrics.get("compute", []),
        marker_color="#ffe66d"
    ))
    
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(size=18, color="white")),
        xaxis_title="Round",
        yaxis_title="Time (seconds)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=350
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_latency_pie(metrics, title="Latency Distribution"):
    """Create pie chart for total latency distribution."""
    total_uplink = sum(metrics.get("uplink", [0]))
    total_downlink = sum(metrics.get("downlink", [0]))
    total_compute = sum(metrics.get("compute", [0]))
    
    fig = go.Figure(data=[go.Pie(
        labels=["Uplink", "Downlink", "Compute"],
        values=[total_uplink, total_downlink, total_compute],
        hole=0.4,
        marker_colors=["#4ecdc4", "#ff6b6b", "#ffe66d"],
        textinfo="label+percent",
        textfont=dict(size=14, color="white")
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=300,
        showlegend=False
    )
    
    return fig


def create_tradeoff_plot(sl_metrics, gsfl_metrics):
    """Create communication vs accuracy trade-off scatter plot."""
    fig = go.Figure()
    
    # Calculate cumulative communication for each round
    sl_acc = [a * 100 for a in sl_metrics.get("accuracy", [])]
    sl_comm = []
    cum_comm = 0
    for up, down in zip(sl_metrics.get("uplink", []), sl_metrics.get("downlink", [])):
        cum_comm += up + down
        sl_comm.append(cum_comm)
    
    gsfl_acc = [a * 100 for a in gsfl_metrics.get("accuracy", [])]
    gsfl_comm = []
    cum_comm = 0
    for up, down in zip(gsfl_metrics.get("uplink", []), gsfl_metrics.get("downlink", [])):
        cum_comm += up + down
        gsfl_comm.append(cum_comm)
    
    fig.add_trace(go.Scatter(
        x=sl_comm, y=sl_acc,
        name="Split Learning",
        mode="lines+markers",
        line=dict(color="#ffa726", width=3),
        marker=dict(size=10, symbol="circle")
    ))
    
    fig.add_trace(go.Scatter(
        x=gsfl_comm, y=gsfl_acc,
        name="GSFL",
        mode="lines+markers",
        line=dict(color="#42a5f5", width=3),
        marker=dict(size=10, symbol="diamond")
    ))
    
    fig.update_layout(
        title=dict(text="Communication vs. Accuracy Trade-off", font=dict(size=18, color="white")),
        xaxis_title="Cumulative Communication Time (s)",
        yaxis_title="Accuracy (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_network_topology(num_clients=16, num_groups=4):
    """Create network topology visualization."""
    fig = go.Figure()
    
    # Access Point / Edge Server at center top
    fig.add_trace(go.Scatter(
        x=[0], y=[2],
        mode="markers+text",
        marker=dict(size=50, color="#ff6b6b", symbol="square"),
        text=["Access Point<br>(Edge Server)"],
        textposition="top center",
        textfont=dict(size=12, color="white"),
        name="Access Point"
    ))
    
    # Calculate client positions in groups
    clients_per_group = num_clients // num_groups
    group_colors = ["#4ecdc4", "#ffe66d", "#42a5f5", "#ab47bc", "#26a69a", "#ec407a"]
    
    client_x = []
    client_y = []
    client_colors = []
    client_text = []
    
    for g in range(num_groups):
        group_center_x = (g - (num_groups - 1) / 2) * 2.5
        
        for c in range(clients_per_group):
            angle = (c / clients_per_group) * np.pi + np.pi / 4
            x = group_center_x + 0.8 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            
            client_x.append(x)
            client_y.append(y)
            client_colors.append(group_colors[g % len(group_colors)])
            client_text.append(f"C{g * clients_per_group + c}")
    
    # Add clients
    fig.add_trace(go.Scatter(
        x=client_x, y=client_y,
        mode="markers+text",
        marker=dict(size=25, color=client_colors),
        text=client_text,
        textposition="bottom center",
        textfont=dict(size=10, color="white"),
        name="Clients"
    ))
    
    # Add connections (lines from AP to clients)
    for x, y in zip(client_x, client_y):
        fig.add_trace(go.Scatter(
            x=[0, x], y=[2, y],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # Add group labels
    for g in range(num_groups):
        group_center_x = (g - (num_groups - 1) / 2) * 2.5
        fig.add_annotation(
            x=group_center_x,
            y=-0.2,
            text=f"Group {g + 1}",
            showarrow=False,
            font=dict(size=12, color=group_colors[g % len(group_colors)])
        )
    
    fig.update_layout(
        title=dict(text="Network Topology", font=dict(size=18, color="white")),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-6, 6]),
        yaxis=dict(visible=False, range=[-0.5, 3]),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_group_progress_chart(group_metrics):
    """Create per-group training progress chart."""
    fig = go.Figure()
    
    colors = ["#4ecdc4", "#ffe66d", "#42a5f5", "#ab47bc", "#26a69a", "#ec407a"]
    
    for i, (group_name, metrics) in enumerate(group_metrics.items()):
        rounds = list(range(1, len(metrics.get("loss", [])) + 1))
        fig.add_trace(go.Scatter(
            x=rounds,
            y=metrics.get("loss", []),
            name=group_name,
            line=dict(color=colors[i % len(colors)], width=2),
            mode="lines+markers",
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=dict(text="Per-Group Training Progress", font=dict(size=18, color="white")),
        xaxis_title="Round",
        yaxis_title="Loss",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=350
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_confusion_matrix(y_true, y_pred, class_names=None):
    """Create confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        texttemplate="%{z}",
        textfont=dict(size=10),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=18, color="white")),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400
    )
    
    return fig


def create_metric_cards(metrics):
    """Generate summary metrics for display."""
    if not metrics.get("accuracy"):
        return {}
    
    final_accuracy = metrics["accuracy"][-1] * 100
    final_loss = metrics["loss"][-1]
    total_uplink = sum(metrics.get("uplink", [0]))
    total_downlink = sum(metrics.get("downlink", [0]))
    total_compute = sum(metrics.get("compute", [0]))
    total_time = total_uplink + total_downlink + total_compute
    
    return {
        "Final Accuracy": f"{final_accuracy:.2f}%",
        "Final Loss": f"{final_loss:.4f}",
        "Total Training Time": f"{total_time:.2f}s",
        "Communication Time": f"{total_uplink + total_downlink:.2f}s",
        "Compute Time": f"{total_compute:.2f}s"
    }


# ============== LIMITATIONS VISUALIZATIONS ==============

def create_cut_layer_analysis():
    """Create visualization showing cut layer trade-off analysis."""
    # Simulated data for different cut layer positions
    cut_layers = ["Layer 1\n(Early)", "Layer 2", "Layer 3\n(Current)", "Layer 4", "Layer 5\n(Late)"]
    
    # Smashed data size (decreases with later cut)
    smashed_data_size = [784, 512, 3136, 1024, 128]  # Example sizes in bytes
    
    # Client compute load (increases with later cut)
    client_compute = [10, 25, 40, 65, 85]  # Percentage
    
    # Server compute load
    server_compute = [90, 75, 60, 35, 15]  # Percentage
    
    # Communication overhead
    comm_overhead = [100, 65, 40, 25, 15]  # Relative scale
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Compute Distribution", "Communication vs. Smashed Data Size"),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Left plot: Stacked bar for compute distribution
    fig.add_trace(
        go.Bar(name="Client Compute", x=cut_layers, y=client_compute, marker_color="#4ecdc4"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="Server Compute", x=cut_layers, y=server_compute, marker_color="#ff6b6b"),
        row=1, col=1
    )
    
    # Right plot: Communication overhead
    fig.add_trace(
        go.Scatter(
            x=cut_layers, y=comm_overhead,
            mode="lines+markers",
            name="Communication Overhead",
            line=dict(color="#ffe66d", width=3),
            marker=dict(size=12)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=cut_layers, y=[s/30 for s in smashed_data_size],  # Normalized
            mode="lines+markers",
            name="Smashed Data Size",
            line=dict(color="#42a5f5", width=3),
            marker=dict(size=12)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        barmode="stack",
        title=dict(text="Cut Layer Position Trade-off Analysis", font=dict(size=18, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        showlegend=True
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_data_distribution_chart():
    """Create visualization comparing IID vs Non-IID data distribution."""
    classes = [f"Class {i}" for i in range(10)]
    
    # IID distribution (uniform)
    iid_client1 = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    iid_client2 = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    iid_client3 = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    # Non-IID distribution (skewed)
    noniid_client1 = [300, 250, 50, 20, 10, 5, 5, 5, 5, 5]
    noniid_client2 = [10, 20, 300, 280, 200, 50, 10, 5, 5, 5]
    noniid_client3 = [5, 5, 10, 20, 50, 200, 280, 300, 250, 180]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("IID Distribution (Current)", "Non-IID Distribution (Real-World)"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # IID
    fig.add_trace(go.Bar(name="Client 1", x=classes, y=iid_client1, marker_color="#4ecdc4"), row=1, col=1)
    fig.add_trace(go.Bar(name="Client 2", x=classes, y=iid_client2, marker_color="#ffe66d"), row=1, col=1)
    fig.add_trace(go.Bar(name="Client 3", x=classes, y=iid_client3, marker_color="#ff6b6b"), row=1, col=1)
    
    # Non-IID
    fig.add_trace(go.Bar(name="Client 1", x=classes, y=noniid_client1, marker_color="#4ecdc4", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="Client 2", x=classes, y=noniid_client2, marker_color="#ffe66d", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="Client 3", x=classes, y=noniid_client3, marker_color="#ff6b6b", showlegend=False), row=1, col=2)
    
    fig.update_layout(
        barmode="group",
        title=dict(text="Data Distribution: IID vs Non-IID", font=dict(size=18, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Samples", gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_privacy_risk_chart():
    """Create visualization showing privacy risk of smashed data."""
    fig = go.Figure()
    
    # Data flow stages
    stages = ["Raw Data", "Client Model", "Smashed Data", "Server Model", "Predictions"]
    
    # Privacy risk levels (1-10)
    privacy_risk = [10, 7, 5, 3, 1]  # Decreasing as we go deeper
    
    # Information leakage potential
    info_leakage = [100, 75, 45, 20, 5]
    
    fig.add_trace(go.Bar(
        x=stages,
        y=privacy_risk,
        name="Privacy Risk",
        marker_color=["#ff6b6b", "#ffa726", "#ffe66d", "#4ecdc4", "#00d4aa"],
        text=["HIGH", "MEDIUM-HIGH", "MEDIUM", "LOW", "MINIMAL"],
        textposition="outside"
    ))
    
    fig.add_trace(go.Scatter(
        x=stages,
        y=[r * 10 for r in privacy_risk],
        mode="lines+markers",
        name="Information Exposure",
        line=dict(color="white", width=2, dash="dash"),
        marker=dict(size=10),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title=dict(text="Privacy Risk at Different Stages", font=dict(size=18, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        yaxis=dict(title="Risk Level (1-10)", gridcolor="rgba(255,255,255,0.1)"),
        yaxis2=dict(title="Information Exposure %", overlaying="y", side="right", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        bargap=0.3
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_grouping_strategy_chart():
    """Create visualization comparing different grouping strategies."""
    strategies = ["Fixed Equal\n(Current)", "Bandwidth-Based", "Distance-Based", "Adaptive Dynamic"]
    
    # Performance metrics (simulated)
    training_time = [100, 75, 80, 60]  # Relative to baseline
    convergence_rounds = [10, 8, 9, 6]
    resource_efficiency = [60, 80, 75, 95]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Relative Training Time",
        x=strategies,
        y=training_time,
        marker_color="#ff6b6b"
    ))
    
    fig.add_trace(go.Bar(
        name="Convergence Rounds",
        x=strategies,
        y=[r * 10 for r in convergence_rounds],
        marker_color="#ffe66d"
    ))
    
    fig.add_trace(go.Bar(
        name="Resource Efficiency",
        x=strategies,
        y=resource_efficiency,
        marker_color="#4ecdc4"
    ))
    
    fig.update_layout(
        barmode="group",
        title=dict(text="Grouping Strategy Comparison", font=dict(size=18, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        yaxis=dict(title="Performance Score", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_wireless_model_chart():
    """Create visualization showing simplified vs realistic wireless model."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Current: Static Channel", "Real-World: Dynamic Channel"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    time_steps = list(range(1, 51))
    
    # Static channel (current implementation)
    static_bandwidth = [50] * 50  # Constant
    
    # Dynamic channel (more realistic)
    np.random.seed(42)
    dynamic_bandwidth = [50 + 30 * np.sin(t/10) + np.random.normal(0, 10) for t in time_steps]
    dynamic_bandwidth = [max(5, b) for b in dynamic_bandwidth]  # Floor at 5
    
    # Static
    fig.add_trace(
        go.Scatter(x=time_steps, y=static_bandwidth, mode="lines", 
                   line=dict(color="#4ecdc4", width=3), name="Static BW"),
        row=1, col=1
    )
    
    # Dynamic with fading
    fig.add_trace(
        go.Scatter(x=time_steps, y=dynamic_bandwidth, mode="lines",
                   line=dict(color="#ff6b6b", width=2), name="Dynamic BW"),
        row=1, col=2
    )
    
    # Add interference events
    interference_times = [15, 30, 42]
    interference_values = [10, 15, 8]
    fig.add_trace(
        go.Scatter(x=interference_times, y=interference_values, mode="markers",
                   marker=dict(size=15, color="yellow", symbol="x"),
                   name="Interference"),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text="Wireless Channel Model Comparison", font=dict(size=18, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(title_text="Time", gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Bandwidth (Mbps)", gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_limitations_radar():
    """Create radar chart showing severity of different limitations."""
    categories = [
        "Fixed Cut Layer",
        "Static Grouping", 
        "Simplified Wireless",
        "IID Data Only",
        "Single Dataset",
        "No Privacy Analysis"
    ]
    
    # Severity scores (0-10, higher = more severe)
    severity = [7, 8, 6, 9, 5, 8]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=severity + [severity[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255, 107, 107, 0.3)",
        line=dict(color="#ff6b6b", width=2),
        name="Severity"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor="rgba(255,255,255,0.2)",
                color="white"
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.2)",
                color="white"
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=dict(text="Research Limitations Severity", font=dict(size=18, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        showlegend=False
    )
    
    return fig

