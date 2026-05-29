import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="SASRec Dashboard", layout="wide")

st.title("SASRec System Dashboard")
st.markdown("Technical metrics and model performance analysis")

# Load results
@st.cache_data
def load_results():
    results = {}
    for file in ['training_log.json', 'ablation_results.json', 'segment_analysis.json', 'benchmark_results.json']:
        path = Path(f"results/{file}")
        if path.exists():
            with open(path) as f:
                results[file] = json.load(f)
    return results

results = load_results()

# ========================================================================
# METRICS
# ========================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Hit@10", "0.7849", "vs 0.8245")
with col2:
    st.metric("NDCG@10", "0.5811", "vs 0.5905")
with col3:
    st.metric("Accuracy", "98.4%")
with col4:
    st.metric("Parameters", "211,950")

st.divider()

# ========================================================================
# TABS
# ========================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Training", "Ablation", "Segments", "Speed"])

with tab1:
    st.markdown("### Training Progress (500 epochs)")
    if 'training_log.json' in results:
        train = results['training_log.json']
        final = train['final_results']
        
        epochs = []
        losses = []
        hits = []
        ndcgs = []
        
        for log in train['training_log'][::5]:
            epochs.append(log['epoch'])
            losses.append(log['train_loss'])
            hits.append(log['val_hit_at_10'])
            ndcgs.append(log['val_ndcg_at_10'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart({"Loss": losses, "Hit Rate": hits}, width="stretch")
        with col2:
            st.line_chart({"NDCG Score": ndcgs}, width="stretch")

with tab2:
    st.markdown("### Model Architecture Comparison")
    if 'ablation_results.json' in results:
        ablation = results['ablation_results.json']
        
        df_data = {
            "Model": [],
            "Hit@10": [],
            "NDCG@10": [],
            "Description": []
        }
        
        for model_name, res in ablation.items():
            df_data["Model"].append(model_name)
            df_data["Hit@10"].append(f"{res['hit_at_10']:.4f}")
            df_data["NDCG@10"].append(f"{res['ndcg_at_10']:.4f}")
            df_data["Description"].append(res['description'])
        
        st.dataframe(pd.DataFrame(df_data), width="stretch")
        
        # Per-segment analysis
        st.markdown("#### Positional Embedding Impact Per Segment")
        if 'no_pe' in ablation and 'per_segment' in ablation['no_pe']:
            seg_data = []
            for seg_name in ['short', 'medium', 'long']:
                with_pe = ablation['default']['per_segment'][seg_name]['ndcg_at_10']
                without_pe = ablation['no_pe']['per_segment'][seg_name]['ndcg_at_10']
                impact = ablation['no_pe']['per_segment'][seg_name].get('pe_impact_ndcg_pct', 0)
                
                seg_data.append({
                    "Segment": seg_name.upper(),
                    "With PE": f"{with_pe:.4f}",
                    "Without PE": f"{without_pe:.4f}",
                    "PE Impact": f"{impact:+.1f}%"
                })
            
            st.dataframe(pd.DataFrame(seg_data), width="stretch")

with tab3:
    st.markdown("### User Segment Performance")
    if 'segment_analysis.json' in results:
        segments = results['segment_analysis.json']
        
        df_data = {
            "Segment": ["SHORT (<20)", "MEDIUM (20-100)", "LONG (>100)"],
            "Users": [
                segments['short']['num_users'],
                segments['medium']['num_users'],
                segments['long']['num_users']
            ],
            "Hit@10": [
                f"{segments['short']['hit_at_10']:.4f}",
                f"{segments['medium']['hit_at_10']:.4f}",
                f"{segments['long']['hit_at_10']:.4f}"
            ],
            "NDCG@10": [
                f"{segments['short']['ndcg_at_10']:.4f}",
                f"{segments['medium']['ndcg_at_10']:.4f}",
                f"{segments['long']['ndcg_at_10']:.4f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(df_data), width="stretch")
        
        st.warning("Model performs worse on long sequences. Attention struggles with >100 items.")

with tab4:
    st.markdown("### Inference Latency Benchmarks")
    if 'benchmark_results.json' in results:
        bench = results['benchmark_results.json']
        
        latencies = []
        for batch_size in ['1', '4', '8', '16', '32', '64', '128']:
            if batch_size in bench:
                latencies.append(bench[batch_size]['mean_latency_ms'])
        
        st.line_chart({"Latency (ms)": latencies}, width="stretch")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch 16", f"{bench['16']['mean_latency_ms']:.2f}ms", "Real-time")
        with col2:
            st.metric("Batch 128", f"{bench['128']['mean_latency_ms']:.2f}ms", "Batch")
        with col3:
            st.metric("Throughput", "8,366/sec", "Max")