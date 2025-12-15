"""
GSFL Visualization Dashboard
A comprehensive Streamlit dashboard for Group-based Split Federated Learning.
"""

import streamlit as st
import json
import os
import sys
import time
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from dashboard_components import (
    create_accuracy_loss_chart,
    create_comparison_chart,
    create_latency_breakdown,
    create_latency_pie,
    create_tradeoff_plot,
    create_network_topology,
    create_group_progress_chart,
    create_metric_cards,
    # Limitations visualizations
    create_cut_layer_analysis,
    create_data_distribution_chart,
    create_privacy_risk_chart,
    create_grouping_strategy_chart,
    create_wireless_model_chart,
    create_limitations_radar,
)

# Page configuration
st.set_page_config(
    page_title="GSFL Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    /* Dark theme colors */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 5px 0 0 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2d2d44 0%, #1e1e2f 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4aa, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(30, 30, 47, 0.8);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-ready {
        background: linear-gradient(90deg, #00d4aa, #00a896);
        color: white;
    }
    
    .status-training {
        background: linear-gradient(90deg, #ffa726, #fb8c00);
        color: white;
    }
    
    .status-missing {
        background: linear-gradient(90deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(45, 45, 68, 0.8);
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4aa, #4ecdc4);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = {"accuracy": [], "loss": [], "uplink": [], "downlink": [], "compute": []}
if "training_log" not in st.session_state:
    st.session_state.training_log = []


def load_results(path):
    """Load training results from JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def check_results_exist():
    """Check if training results exist."""
    sl_exists = os.path.exists(PROJECT_ROOT / "results" / "sl_results.json")
    gsfl_exists = os.path.exists(PROJECT_ROOT / "results" / "gsfl_results.json")
    return sl_exists, gsfl_exists


def run_training(mode, rounds, num_clients, num_groups, batch_size, lr_client, lr_server, progress_placeholder, metrics_placeholder):
    """Run training with real-time updates."""
    from gsfl.data import get_client_datasets, get_test_loader
    from gsfl.core import GSFLTrainer, SplitLearningTrainer
    import gsfl.config as config
    
    # Update config
    config.NUM_CLIENTS = num_clients
    config.NUM_GROUPS = num_groups
    config.BATCH_SIZE = batch_size
    config.LR_CLIENT = lr_client
    config.LR_SERVER = lr_server
    
    # Get data
    client_datasets = get_client_datasets()
    test_loader = get_test_loader()
    
    # Initialize trainer
    if mode == "GSFL":
        trainer = GSFLTrainer(client_datasets, test_loader)
    else:
        trainer = SplitLearningTrainer(client_datasets, test_loader)
    
    metrics = {"accuracy": [], "loss": [], "uplink": [], "downlink": [], "compute": []}
    
    for r in range(1, rounds + 1):
        loss, up, down, comp = trainer.train_round()
        acc = trainer.evaluate()
        
        metrics["accuracy"].append(acc)
        metrics["loss"].append(loss)
        metrics["uplink"].append(up)
        metrics["downlink"].append(down)
        metrics["compute"].append(comp)
        
        # Update session state
        st.session_state.current_metrics = metrics.copy()
        st.session_state.training_log.append(
            f"Round {r}: Acc={acc:.4f}, Loss={loss:.4f}, Up={up:.4f}s, Down={down:.4f}s"
        )
        
        # Update progress
        progress_placeholder.progress(r / rounds, text=f"Training Round {r}/{rounds}")
        
        # Update live chart
        with metrics_placeholder.container():
            st.plotly_chart(create_accuracy_loss_chart(metrics, f"{mode} Training Progress"), use_container_width=True)
    
    # Save results
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)
    result_file = f"{'gsfl' if mode == 'GSFL' else 'sl'}_results.json"
    with open(PROJECT_ROOT / "results" / result_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save models
    if mode == "GSFL":
        torch.save(trainer.client.state_dict(), PROJECT_ROOT / "results" / "gsfl_client.pt")
        torch.save(trainer.servers[0].state_dict(), PROJECT_ROOT / "results" / "gsfl_server.pt")
    else:
        torch.save(trainer.client.state_dict(), PROJECT_ROOT / "results" / "sl_client.pt")
        torch.save(trainer.server.state_dict(), PROJECT_ROOT / "results" / "sl_server.pt")
    
    return metrics


# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 GSFL Dashboard</h1>
    <p>Group-based Split Federated Learning Visualization & Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Check results status
    sl_exists, gsfl_exists = check_results_exist()
    
    st.markdown("### 📊 Results Status")
    col1, col2 = st.columns(2)
    with col1:
        if sl_exists:
            st.markdown('<span class="status-badge status-ready">SL ✓</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-missing">SL ✗</span>', unsafe_allow_html=True)
    with col2:
        if gsfl_exists:
            st.markdown('<span class="status-badge status-ready">GSFL ✓</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-missing">GSFL ✗</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Training mode selector
    st.markdown("### 🎯 Training Mode")
    training_mode = st.selectbox(
        "Select Algorithm",
        ["GSFL", "Split Learning (SL)"],
        help="Choose the training algorithm to run"
    )
    
    st.markdown("---")
    
    # Hyperparameters
    st.markdown("### 🔧 Hyperparameters")
    
    num_rounds = st.slider("Training Rounds", 1, 50, 10)
    num_clients = st.slider("Number of Clients", 4, 32, 16)
    num_groups = st.slider("Number of Groups", 2, 8, 4)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with st.expander("Advanced Settings"):
        lr_client = st.number_input("Client Learning Rate", 0.001, 0.5, 0.05, 0.01)
        lr_server = st.number_input("Server Learning Rate", 0.001, 0.5, 0.05, 0.01)
    
    st.markdown("---")
    
    # Training buttons
    st.markdown("### 🚀 Actions")
    
    mode_key = "GSFL" if "GSFL" in training_mode else "SL"
    
    if st.button(f"▶️ Run {mode_key} Training", use_container_width=True, disabled=st.session_state.training_in_progress):
        st.session_state.training_in_progress = True
        st.session_state.training_log = []
        st.rerun()
    
    if st.button("🔄 Refresh Results", use_container_width=True):
        st.rerun()


# Main content area
if st.session_state.training_in_progress:
    # Training view
    st.markdown("## 🏃 Training in Progress")
    
    mode_key = "GSFL" if "GSFL" in training_mode else "SL"
    
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Run training
    try:
        metrics = run_training(
            mode_key, num_rounds, num_clients, num_groups, batch_size, lr_client, lr_server,
            progress_placeholder, metrics_placeholder
        )
        
        st.session_state.training_in_progress = False
        st.success(f"✅ {mode_key} Training completed successfully!")
        st.balloons()
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.session_state.training_in_progress = False
        st.error(f"❌ Training failed: {str(e)}")
        st.exception(e)

else:
    # Results view
    tabs = st.tabs(["📈 Overview", "🔄 Comparison", "📊 Latency Analysis", "🌐 Network", "📋 Training Log", "⚠️ Limitations"])
    
    # Load results
    sl_metrics = load_results(PROJECT_ROOT / "results" / "sl_results.json")
    gsfl_metrics = load_results(PROJECT_ROOT / "results" / "gsfl_results.json")
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown("## 📈 Training Overview")
        
        if not sl_metrics and not gsfl_metrics:
            st.warning("⚠️ No training results found. Use the sidebar to run training first!")
            
            # Show what would be displayed
            st.markdown("### 🎯 Available Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                st.info("📊 **Accuracy & Loss Charts** - Real-time training progress")
                st.info("📈 **Per-Round Metrics** - Detailed performance tracking")
            with col2:
                st.info("⏱️ **Latency Breakdown** - Communication vs Compute time")
                st.info("🔄 **Trade-off Analysis** - Communication vs Accuracy")
        else:
            # Metrics for selected mode
            selected_metrics = gsfl_metrics if gsfl_metrics else sl_metrics
            mode_name = "GSFL" if gsfl_metrics else "SL"
            
            # Summary cards
            if selected_metrics:
                cards = create_metric_cards(selected_metrics)
                cols = st.columns(5)
                
                for i, (label, value) in enumerate(cards.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{value}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if gsfl_metrics:
                    st.plotly_chart(create_accuracy_loss_chart(gsfl_metrics, "GSFL Training Progress"), use_container_width=True)
                else:
                    st.info("Run GSFL training to see results")
            
            with col2:
                if sl_metrics:
                    st.plotly_chart(create_accuracy_loss_chart(sl_metrics, "Split Learning Training Progress"), use_container_width=True)
                else:
                    st.info("Run SL training to see results")
    
    # Tab 2: Comparison
    with tabs[1]:
        st.markdown("## 🔄 SL vs GSFL Comparison")
        
        if sl_metrics and gsfl_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_comparison_chart(sl_metrics, gsfl_metrics, "accuracy", "Accuracy Comparison", "Accuracy (%)"),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_comparison_chart(sl_metrics, gsfl_metrics, "loss", "Loss Comparison", "Loss"),
                    use_container_width=True
                )
            
            # Trade-off plot
            st.plotly_chart(create_tradeoff_plot(sl_metrics, gsfl_metrics), use_container_width=True)
            
            # Comparison table
            st.markdown("### 📊 Performance Summary")
            
            sl_cards = create_metric_cards(sl_metrics)
            gsfl_cards = create_metric_cards(gsfl_metrics)
            
            comparison_data = {
                "Metric": list(sl_cards.keys()),
                "Split Learning": list(sl_cards.values()),
                "GSFL": list(gsfl_cards.values())
            }
            
            st.dataframe(comparison_data, use_container_width=True, hide_index=True)
            
        else:
            st.warning("⚠️ Both SL and GSFL results are required for comparison. Train both models first!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Split Learning")
                if sl_metrics:
                    st.success("✅ Results available")
                else:
                    st.error("❌ No results - Run training")
            
            with col2:
                st.markdown("### GSFL")
                if gsfl_metrics:
                    st.success("✅ Results available")
                else:
                    st.error("❌ No results - Run training")
    
    # Tab 3: Latency Analysis
    with tabs[2]:
        st.markdown("## 📊 Latency Analysis")
        
        if sl_metrics or gsfl_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                if gsfl_metrics:
                    st.plotly_chart(create_latency_breakdown(gsfl_metrics, "GSFL Latency Breakdown"), use_container_width=True)
                    st.plotly_chart(create_latency_pie(gsfl_metrics, "GSFL Latency Distribution"), use_container_width=True)
            
            with col2:
                if sl_metrics:
                    st.plotly_chart(create_latency_breakdown(sl_metrics, "SL Latency Breakdown"), use_container_width=True)
                    st.plotly_chart(create_latency_pie(sl_metrics, "SL Latency Distribution"), use_container_width=True)
            
            # Latency comparison
            if sl_metrics and gsfl_metrics:
                st.markdown("### ⏱️ Total Latency Comparison")
                
                sl_total = sum(sl_metrics["uplink"]) + sum(sl_metrics["downlink"]) + sum(sl_metrics["compute"])
                gsfl_total = sum(gsfl_metrics["uplink"]) + sum(gsfl_metrics["downlink"]) + sum(gsfl_metrics["compute"])
                
                reduction = ((sl_total - gsfl_total) / sl_total) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SL Total Time", f"{sl_total:.2f}s")
                with col2:
                    st.metric("GSFL Total Time", f"{gsfl_total:.2f}s")
                with col3:
                    st.metric("Time Reduction", f"{reduction:.1f}%", delta=f"-{sl_total - gsfl_total:.2f}s")
        else:
            st.warning("⚠️ No training results available. Run training first!")
    
    # Tab 4: Network
    with tabs[3]:
        st.markdown("## 🌐 Network Topology")
        
        st.plotly_chart(create_network_topology(num_clients, num_groups), use_container_width=True)
        
        st.markdown("### 📡 Network Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Clients**: {num_clients}  
            **Groups**: {num_groups}  
            **Clients per Group**: {num_clients // num_groups}
            """)
        
        with col2:
            st.markdown(f"""
            **Uplink Bandwidth**: 5 MB/s  
            **Downlink Bandwidth**: 5 MB/s  
            **Carrier Frequency**: 3.5 GHz
            """)
        
        # Per-group info
        if gsfl_metrics and "group_metrics" in gsfl_metrics:
            st.markdown("### 📊 Per-Group Training Progress")
            st.plotly_chart(create_group_progress_chart(gsfl_metrics["group_metrics"]), use_container_width=True)
    
    # Tab 5: Training Log
    with tabs[4]:
        st.markdown("## 📋 Training Log")
        
        if st.session_state.training_log:
            for log in st.session_state.training_log:
                st.text(log)
        else:
            st.info("Training logs will appear here after running training.")
        
        # Show existing result files
        st.markdown("### 📁 Saved Results")
        
        results_dir = PROJECT_ROOT / "results"
        if results_dir.exists():
            files = list(results_dir.glob("*"))
            for f in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f.name)
                with col2:
                    st.text(f"{f.stat().st_size / 1024:.1f} KB")
        else:
            st.info("No results saved yet.")
    
    # Tab 6: Limitations
    with tabs[5]:
        st.markdown("## ⚠️ Research Paper Limitations & Solutions")
        
        st.markdown("""
        This section analyzes the key limitations of the GSFL research paper 
        and proposes potential solutions for each limitation.
        """)
        
        # Limitations Radar Overview
        st.plotly_chart(create_limitations_radar(), use_container_width=True)
        
        st.markdown("---")
        
        # Limitation 1: Fixed Cut Layer
        st.markdown("### 🔪 1. Fixed Cut Layer Position")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_cut_layer_analysis(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #ff6b6b;">🔴 Problem</h4>
                <p>The paper uses a single, fixed cut layer position without analyzing the trade-off between:</p>
                <ul>
                    <li>Smashed data size</li>
                    <li>Client vs Server compute load</li>
                    <li>Communication overhead</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #00d4aa;">✅ Solution</h4>
                <ul>
                    <li><b>Dynamic cut layer selection</b> based on network conditions</li>
                    <li><b>Adaptive algorithms</b> that adjust cut layer per client</li>
                    <li><b>Multi-objective optimization</b> considering latency, accuracy, and energy</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Limitation 2: Static Client Grouping
        st.markdown("### 👥 2. Static Client Grouping")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_grouping_strategy_chart(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #ff6b6b;">🔴 Problem</h4>
                <p>Clients are divided into fixed, equal-sized groups without considering:</p>
                <ul>
                    <li>Heterogeneous network conditions</li>
                    <li>Varying computational capabilities</li>
                    <li>Dynamic client availability</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #00d4aa;">✅ Solution</h4>
                <ul>
                    <li><b>Bandwidth-based grouping</b>: Group clients with similar speeds</li>
                    <li><b>Distance-aware grouping</b>: Minimize communication latency</li>
                    <li><b>Adaptive dynamic grouping</b>: Re-group based on real-time conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Limitation 3: IID Data Only
        st.markdown("### 📊 3. IID Data Distribution Only")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_data_distribution_chart(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #ff6b6b;">🔴 Problem</h4>
                <p>Experiments assume IID (Independent and Identically Distributed) data:</p>
                <ul>
                    <li>Real-world data is often highly skewed</li>
                    <li>Clients may have different label distributions</li>
                    <li>Model convergence is harder with Non-IID data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #00d4aa;">✅ Solution</h4>
                <ul>
                    <li><b>Dirichlet distribution</b> for simulating Non-IID scenarios</li>
                    <li><b>FedProx algorithm</b> for better Non-IID handling</li>
                    <li><b>Data augmentation</b> and knowledge distillation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Limitation 4: Simplified Wireless Model
        st.markdown("### 📡 4. Simplified Wireless Channel Model")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_wireless_model_chart(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #ff6b6b;">🔴 Problem</h4>
                <p>The wireless simulation assumes:</p>
                <ul>
                    <li>Static client positions (no mobility)</li>
                    <li>Uniform resource allocation</li>
                    <li>No interference or congestion</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #00d4aa;">✅ Solution</h4>
                <ul>
                    <li><b>Time-varying channels</b> with Rayleigh/Rician fading</li>
                    <li><b>Mobility models</b> for moving clients</li>
                    <li><b>Interference modeling</b> and NOMA techniques</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Limitation 5: Privacy Concerns
        st.markdown("### 🔐 5. No Privacy Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_privacy_risk_chart(), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #ff6b6b;">🔴 Problem</h4>
                <p>Smashed data transmitted to server may leak information:</p>
                <ul>
                    <li>No differential privacy mechanisms</li>
                    <li>No secure aggregation</li>
                    <li>Potential for gradient inversion attacks</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #00d4aa;">✅ Solution</h4>
                <ul>
                    <li><b>Differential Privacy</b>: Add calibrated noise to smashed data</li>
                    <li><b>Secure Multi-Party Computation</b></li>
                    <li><b>Homomorphic Encryption</b> for server-side computations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summary Table
        st.markdown("### 📋 Limitations Summary")
        
        limitations_data = {
            "Limitation": [
                "Fixed Cut Layer",
                "Static Grouping",
                "IID Data Only",
                "Simplified Wireless",
                "Single Dataset",
                "No Privacy Analysis"
            ],
            "Severity": ["⚠️ High", "⚠️ High", "🔴 Critical", "⚠️ Medium", "⚠️ Medium", "⚠️ High"],
            "Paper Mentions": ["✅ Yes", "✅ Yes", "❌ No", "✅ Yes", "❌ No", "❌ No"],
            "Proposed Solution": [
                "Dynamic cut layer selection",
                "Adaptive bandwidth-based grouping",
                "Non-IID support with FedProx",
                "Time-varying channel models",
                "Multi-dataset evaluation",
                "Differential privacy"
            ]
        }
        
        st.dataframe(limitations_data, use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem;">
    GSFL Dashboard | PDC Project | Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

