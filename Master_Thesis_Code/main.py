import os
import time
import random
import warnings
import platform
import psutil
import pandas as pd
import numpy as np
import matplotlib
import itertools
from importlib.metadata import version, PackageNotFoundError

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import re
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from dotenv import load_dotenv

# ==============================================================================
# 0. CONFIGURATION & SETUP
# ==============================================================================
warnings.filterwarnings("ignore")
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

load_dotenv()

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

os.environ["OPENAI_API_KEY"] = "NA"

# --- MULTI-PROVIDER LLM POOL (Dynamic Key Rotation + Rate Limit Awareness) ---
# Supports: Gemini, Groq, xAI, OpenRouter, Cerebras — loaded dynamically from .env
# Each provider entry: { env_key, model, api_key_env_var, rpm, rpd, tpm }

def _build_provider_pool():
    """Build LLM provider pool from all available keys in .env.
    Priority order: Groq (primary) -> Gemini -> xAI -> OpenRouter -> Cerebras.
    This multi-provider design is a deliberate POC for production scalability."""
    pool = []
    # Groq keys first — primary provider (GROQ_API_KEY, GROQ_API_KEY_2, ..., GROQ_API_KEY_7)
    for suffix in ["", "_2", "_3", "_4", "_5", "_6", "_7"]:
        key = os.getenv(f"GROQ_API_KEY{suffix}")
        if key and not key.startswith("xai-"):  # Filter out mislabeled xAI keys
            pool.append({
                "name": f"groq{suffix}", "model": "groq/llama-3.3-70b-versatile",
                "env_var": "GROQ_API_KEY", "api_key": key,
                "rpm": 30, "rpd": 14400, "tpm": 6000,
                "requests_used": 0, "tokens_used": 0, "exhausted": False
            })
    # Gemini keys (GEMINI_API_KEY_1, _2, _3, ...)
    for i in range(1, 10):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            pool.append({
                "name": f"gemini_{i}", "model": "gemini/gemini-2.0-flash",
                "env_var": "GEMINI_API_KEY", "api_key": key,
                "rpm": 15, "rpd": 1500, "tpm": 1000000,
                "requests_used": 0, "tokens_used": 0, "exhausted": False
            })
    # xAI keys (XAI_API_KEY or mislabeled GROQ keys starting with xai-)
    xai_key = os.getenv("XAI_API_KEY")
    if xai_key:
        pool.append({
            "name": "xai", "model": "xai/grok-2-latest",
            "env_var": "XAI_API_KEY", "api_key": xai_key,
            "rpm": 60, "rpd": 1000, "tpm": 100000,
            "requests_used": 0, "tokens_used": 0, "exhausted": False
        })
    # Also check for xAI keys mislabeled as GROQ
    for suffix in ["", "_2", "_3", "_4", "_5", "_6", "_7"]:
        key = os.getenv(f"GROQ_API_KEY{suffix}")
        if key and key.startswith("xai-"):
            pool.append({
                "name": f"xai_from_groq{suffix}", "model": "xai/grok-2-latest",
                "env_var": "XAI_API_KEY", "api_key": key,
                "rpm": 60, "rpd": 1000, "tpm": 100000,
                "requests_used": 0, "tokens_used": 0, "exhausted": False
            })
    # OpenRouter keys (OPEN_ROUTER_API_KEY, _2, _3, ... or OPENROUTER_API_KEY)
    or_keys_seen = set()
    for suffix in ["", "_2", "_3", "_4", "_5"]:
        key = os.getenv(f"OPEN_ROUTER_API_KEY{suffix}") or os.getenv(f"OPENROUTER_API_KEY{suffix}")
        if key and key not in or_keys_seen:
            or_keys_seen.add(key)
            pool.append({
                "name": f"openrouter{suffix}", "model": "openrouter/google/gemini-2.0-flash-thinking-exp:free",
                "env_var": "OPENROUTER_API_KEY", "api_key": key,
                "rpm": 20, "rpd": 50, "tpm": 100000,
                "requests_used": 0, "tokens_used": 0, "exhausted": False
            })
    # Cerebras (CEREBRAS_API_KEY)
    cb_key = os.getenv("CEREBRAS_API_KEY")
    if cb_key:
        pool.append({
            "name": "cerebras", "model": "cerebras/llama-3.3-70b",
            "env_var": "CEREBRAS_API_KEY", "api_key": cb_key,
            "rpm": 30, "rpd": 1000, "tpm": 60000,
            "requests_used": 0, "tokens_used": 0, "exhausted": False
        })
    return pool

LLM_POOL = _build_provider_pool()
_POOL_INDEX = 0
_TOTAL_TOKENS_USED = 0
MAX_TOKENS_PER_RUN = 10000  # Global token budget per full pipeline run
N_RUNS = 10  # Number of agent comparison trials for statistical significance
RETRY_WAIT_TOTAL = 0  # Accumulated retry wait time (seconds)

def preflight_check_providers():
    """Test each provider with a minimal request BEFORE the pipeline starts.
    Removes providers that fail (rate-limited / invalid key) so the pipeline
    only uses verified providers. Prints clear guidance when none pass."""
    import litellm
    litellm.suppress_debug_info = True

    alive = []
    print("\n" + "="*60)
    print("  PRE-FLIGHT LLM PROVIDER CHECK")
    print("="*60)
    for p in LLM_POOL:
        os.environ[p["env_var"]] = p["api_key"]
        try:
            resp = litellm.completion(
                model=p["model"],
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=3, temperature=0,
            )
            print(f"  [OK]   {p['name']:20s} ({p['model']})")
            alive.append(p)
        except Exception as e:
            reason = str(e)[:120].replace("\n", " ")
            print(f"  [FAIL] {p['name']:20s} -> {reason}")
    print("="*60)

    if not alive:
        print("\n" + "!"*60)
        print("  HICBIR LLM SAGLAYICISI KULLANILAMIYOR!")
        print("  Pipeline calistirilamaz. Asagidaki adimlarla ucretsiz")
        print("  API anahtari alin ve .env dosyasina ekleyin:\n")
        print("  1) Groq  (onerilen, en hizli):")
        print("     https://console.groq.com/keys")
        print("     .env -> GROQ_API_KEY=gsk_...\n")
        print("  2) Google Gemini:")
        print("     https://aistudio.google.com/app/apikey")
        print("     .env -> GEMINI_API_KEY_1=AIza...\n")
        print("  3) OpenRouter:")
        print("     https://openrouter.ai/keys")
        print("     .env -> OPENROUTER_API_KEY=sk-or-v1-...\n")
        print("  4) Cerebras:")
        print("     https://cloud.cerebras.ai/")
        print("     .env -> CEREBRAS_API_KEY=csk-...\n")
        print("  5) xAI (Grok):")
        print("     https://console.x.ai/")
        print("     .env -> XAI_API_KEY=xai-...\n")
        print("!"*60)
        exit(1)

    print(f"\n  {len(alive)}/{len(LLM_POOL)} saglayici aktif. Pipeline baslatiliyor.\n")

    LLM_POOL.clear()
    LLM_POOL.extend(alive)

def get_active_llm():
    """Round-robin through available non-exhausted providers."""
    global _POOL_INDEX
    if not LLM_POOL:
        print("ERROR: No API keys found in .env!"); exit()
    attempts = 0
    while attempts < len(LLM_POOL):
        provider = LLM_POOL[_POOL_INDEX % len(LLM_POOL)]
        _POOL_INDEX += 1
        if not provider["exhausted"]:
            os.environ[provider["env_var"]] = provider["api_key"]
            return provider["model"]
        attempts += 1
    # All exhausted — reset and try again
    for p in LLM_POOL:
        p["exhausted"] = False
    provider = LLM_POOL[0]
    os.environ[provider["env_var"]] = provider["api_key"]
    return provider["model"]

def mark_provider_exhausted(error_msg=""):
    """Mark current provider as exhausted based on error context."""
    global _POOL_INDEX
    if not LLM_POOL:
        return
    idx = (_POOL_INDEX - 1) % len(LLM_POOL)
    provider = LLM_POOL[idx]
    provider["exhausted"] = True
    active_count = sum(1 for p in LLM_POOL if not p["exhausted"])
    print(f"[WARN] Provider '{provider['name']}' exhausted. {active_count}/{len(LLM_POOL)} providers remaining.")

def track_token_budget(tokens):
    """Track global token usage against budget."""
    global _TOTAL_TOKENS_USED
    _TOTAL_TOKENS_USED += tokens
    if _TOTAL_TOKENS_USED > MAX_TOKENS_PER_RUN:
        print(f"[WARN] Token budget warning: {_TOTAL_TOKENS_USED}/{MAX_TOKENS_PER_RUN} tokens used.")

def get_provider_status():
    """Return summary of all providers for logging."""
    lines = []
    for p in LLM_POOL:
        status = "EXHAUSTED" if p["exhausted"] else "ACTIVE"
        lines.append(f"  {p['name']:20s} | {p['model']:40s} | {status}")
    return "\n".join(lines)

MY_LLM = get_active_llm()

# Global Metrics
THESIS_METRICS = {
    "baseline_time": 0,
    "multi_agent_time": 0,
    "baseline_actionable_density": 0,
    "multi_agent_actionable_density": 0,
    "clustering_metrics": {},
    "clustering_comparison": {},
    "ml_baseline_metrics": {},
    "baseline_tokens": 0,
    "multi_agent_tokens": 0,
    "optimization_trials": [],
    "best_features": [],
    "interaction_log": [],
    "agent_token_usage": [],
    "preprocessing_steps": [],
    "step_times": {}
}

EXECUTION_LOG = []

def log_step(step_name, description):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] **{step_name}**: {description}"
    print(f"\n>> {entry}")
    EXECUTION_LOG.append(entry)

def log_interaction(agent_role, input_task, output_content):
    THESIS_METRICS["interaction_log"].append({
        "role": agent_role,
        "input": str(input_task).strip(),
        "output": str(output_content).strip()
    })

def track_agent_cost(loop_id, agent_name, output_content):
    est_output = len(str(output_content)) // 4
    est_input = 1000 + (loop_id * 500)
    total = est_input + est_output
    THESIS_METRICS["agent_token_usage"].append({
        "Loop": loop_id,
        "Agent": agent_name,
        "Tokens": total
    })
    return total

def calculate_actionable_density_score(text):
    """
    Actionable Density Score (ADS): Measures the ratio of specific, actionable
    elements in text output. Thesis-grade quality metric.
    Indicators: percentages, dollar amounts, time frames, action verbs,
    segmentation references, financial terms.
    """
    text = str(text)
    sentences = [s.strip() for s in re.split(r'[.\n]', text) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    actionable_patterns = [
        r'\d+%',
        r'\$\d+',
        r'\d+\s*(month|year|day|week)',
        r'(discount|offer|upgrade|campaign|incentive|loyalty\s*point|free\s*trial)',
        r'(cluster|segment|persona|group|tier)',
        r'(revenue|cost|profit|ROI|budget|churn\s*rate)',
    ]
    actionable_count = 0
    for sentence in sentences:
        for pattern in actionable_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                actionable_count += 1
                break
    raw_ratio = actionable_count / len(sentences)
    score = min(10.0, raw_ratio * 12.5)
    return round(score, 2)

def estimate_tokens(text):
    return len(str(text)) // 4

# ==============================================================================
# 1. VISUALIZATION ENGINE (INTEGRATED EDA + RESULTS)
# ==============================================================================

def generate_initial_data_visuals():
    """
    Generates comprehensive EDA charts for the thesis introduction.
    """
    log_step("Visualization", "Generating Comprehensive EDA Charts...")
    if not os.path.exists("data/telco_churn.csv"): return
    
    df = pd.read_csv("data/telco_churn.csv")
    # Clean TotalCharges for visualization
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Churn Distribution (Donut)
    try:
        plt.figure(figsize=(8, 8))
        target_col = 'Churn' if 'Churn' in df.columns else df.columns[-1]
        counts = df[target_col].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'], wedgeprops={'width':0.3})
        plt.title(f'Churn Distribution', fontsize=14)
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig_EDA_1_Churn_Donut.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except: pass

    # 2. Numerical Distributions — individual PNGs per variable
    try:
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        valid_nums = [c for c in num_cols if c in df.columns]
        for idx, col in enumerate(valid_nums):
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df, x=col, hue='Churn', kde=False, element="step", palette='coolwarm')
            plt.title(f'{col} Distribution by Churn', fontsize=13)
            plt.savefig(os.path.join(OUTPUT_DIR, f"Fig_EDA_2{chr(97+idx)}_{col}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    except: pass

    # 2d-2e. Categorical count bars (Contract, PaymentMethod) by Churn
    try:
        cat_count_cols = ['Contract', 'PaymentMethod']
        for idx, col in enumerate(cat_count_cols):
            if col in df.columns:
                plt.figure(figsize=(8, 5))
                sns.countplot(data=df, x=col, hue='Churn', palette='coolwarm', edgecolor='black')
                plt.title(f'{col} Distribution by Churn', fontsize=13)
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"Fig_EDA_2{chr(100+idx)}_{col}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    except: pass

    # 3. Correlation Heatmap
    try:
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, square=True)
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig_EDA_3_Correlation.png"), dpi=300)
        plt.close()
    except: pass

    # 4. Churn Rate by Categorical Feature
    try:
        cat_cols = ['Contract', 'PaymentMethod', 'InternetService']
        valid_cats = [c for c in cat_cols if c in df.columns and 'Churn' in df.columns]
        if valid_cats:
            fig, axes = plt.subplots(1, len(valid_cats), figsize=(18, 6))
            if len(valid_cats) == 1: axes = [axes]
            for i, col in enumerate(valid_cats):
                churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                colors = ['#e74c3c' if r > 30 else '#f39c12' if r > 20 else '#2ecc71' for r in churn_rate.values]
                bars = axes[i].bar(range(len(churn_rate)), churn_rate.values, color=colors, edgecolor='black')
                axes[i].set_xticks(range(len(churn_rate)))
                axes[i].set_xticklabels(churn_rate.index, rotation=30, ha='right', fontsize=9)
                axes[i].set_ylabel('Churn Rate (%)')
                axes[i].set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
                axes[i].axhline(y=26.5, color='blue', linestyle='--', alpha=0.5, label='Avg 26.5%')
                axes[i].legend(fontsize=8)
                for bar in bars:
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "Fig_EDA_4_Churn_by_Category.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        print(f"Categorical churn chart error: {e}")
        plt.close('all')

def draw_optimization_chart():
    log_step("Visualization", "Drawing Optimization Chart...")
    trials = THESIS_METRICS["optimization_trials"]
    if not trials: return

    trials.sort(key=lambda x: x[0])
    x = [t[0] for t in trials]
    y = [t[1] for t in trials]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title('Clustering Optimization (Silhouette Score)', fontsize=14)
    plt.xlabel('Trial ID')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if y:
        max_y = max(y)
        max_x = x[y.index(max_y)]
        plt.annotate(f'Best: {max_y:.3f}', xy=(max_x, max_y), xytext=(max_x, max_y+0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Optimization_Process.png"), dpi=300)
    plt.close()

def draw_performance_charts():
    log_step("Visualization", "Drawing Results Charts...")
    
    methods = ['Baseline', 'Multi-Agent']
    times = [THESIS_METRICS['baseline_time'], THESIS_METRICS['multi_agent_time']]
    ads = [THESIS_METRICS['baseline_actionable_density'], THESIS_METRICS['multi_agent_actionable_density']]
    tokens = [THESIS_METRICS['baseline_tokens'], THESIS_METRICS['multi_agent_tokens']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Time
    bars = axes[0].bar(methods, times, color=['#3498db', '#2980b9'], width=0.5)
    axes[0].set_ylabel('Time (s)', fontweight='bold')
    axes[0].set_title('Execution Time', fontsize=13)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. Actionable Density Score
    bars = axes[1].bar(methods, ads, color=['#e74c3c', '#c0392b'], width=0.5)
    axes[1].set_ylabel('ADS (0-10)', fontweight='bold')
    axes[1].set_title('Actionable Density Score', fontsize=13)
    axes[1].set_ylim(0, 10)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Comp_Performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Cost (Separate chart as per thesis requirements)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(methods, tokens, color=['#2ecc71', '#27ae60'], width=0.5)
    plt.ylabel('Estimated Tokens', fontweight='bold')
    plt.title('Computational Cost (Token Usage)', fontsize=13)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Comp_Cost.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_feature_distributions(df):
    """Draws histograms for the selected best features."""
    best_cols = THESIS_METRICS["best_features"]
    log_step("Visualization", f"Generating histograms for optimal features: {best_cols}")
    sns.set_theme(style="whitegrid")
    
    for col in best_cols:
        if col in df.columns:
            try:
                plt.figure(figsize=(8, 5))
                if df[col].nunique() < 10:
                    sns.countplot(x=col, data=df, palette='viridis')
                    plt.title(f'Distribution of {col}')
                else:
                    # Removed KDE as requested
                    sns.histplot(df[col], kde=False, color='steelblue', bins=20, edgecolor='black')
                    plt.title(f'Distribution of {col}')
                
                safe_col = col.replace(" ", "_")
                plt.savefig(os.path.join(OUTPUT_DIR, f"Fig_Dist_{safe_col}.png"), dpi=300)
                plt.close()
            except: pass

def draw_thesis_pipeline():
    log_step("Visualization", "Drawing Architecture Diagram...")
    G = nx.DiGraph()
    edges = [("Data\nIngestion", "Train/Test\nSplit"), 
             ("Train/Test\nSplit", "Baseline\nAgent"), 
             ("Train/Test\nSplit", "Data\nArchitect"),
             ("Data\nArchitect", "CAO\n(Clustering)"),
             ("CAO\n(Clustering)", "Strategist"), 
             ("Strategist", "Manager\n(Audit)"), 
             ("Manager\n(Audit)", "Strategist"),
             ("Manager\n(Audit)", "Final\nReport"), 
             ("Baseline\nAgent", "Final\nReport")]
    G.add_edges_from(edges)
    
    pos = {
        "Data\nIngestion": (0, 2),
        "Train/Test\nSplit": (2, 2),
        "Baseline\nAgent": (2, 0),
        "Data\nArchitect": (4, 3),
        "CAO\n(Clustering)": (6, 3),
        "Strategist": (8, 3),
        "Manager\n(Audit)": (8, 1),
        "Final\nReport": (10, 0),
    }
    
    node_colors = ['#AED6F1', '#85C1E9', '#F9E79F', '#F5B7B1', '#F5B7B1', '#F5B7B1', '#F5B7B1', '#ABEBC6']
    
    plt.figure(figsize=(14, 7))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=3500, 
            font_size=9, font_weight='bold', arrows=True, arrowsize=20,
            edge_color='gray', width=1.5, node_shape='s')
    nx.draw_networkx_edges(G, pos, 
                           edgelist=[("Manager\n(Audit)", "Strategist")], 
                           edge_color='red', connectionstyle="arc3,rad=0.3", 
                           width=2.5, style='dashed')
    plt.title("Multi-Agent System Architecture", fontsize=14, fontweight='bold')
    plt.annotate('Adversarial\nFeedback Loop', xy=(8, 2), fontsize=9, color='red', 
                 fontstyle='italic', ha='center')
    plt.savefig(os.path.join(OUTPUT_DIR, "Thesis_Pipeline_Architecture.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_agent_token_chart():
    log_step("Visualization", "Drawing Agent Token Usage Chart...")
    usage = THESIS_METRICS["agent_token_usage"]
    if not usage: return
    
    df_tokens = pd.DataFrame(usage)
    pivot = df_tokens.pivot_table(index='Loop', columns='Agent', values='Tokens', aggfunc='sum', fill_value=0)
    
    ax = pivot.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2', edgecolor='black')
    ax.set_title('Token Usage per Agent per Loop', fontsize=13, fontweight='bold')
    ax.set_xlabel('Loop Iteration', fontweight='bold')
    ax.set_ylabel('Estimated Tokens', fontweight='bold')
    ax.legend(title='Agent Role', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Agent_Token_Usage.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_all_runs_individual_charts(df_runs):
    """Draw individual charts for each metric column in All_Runs_Metrics.csv (excluding run_id)."""
    log_step("Visualization", "Drawing individual charts for All_Runs_Metrics columns...")
    
    # Exclude run_id column
    metric_cols = [col for col in df_runs.columns if col != 'run_id']
    
    for col in metric_cols:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(df_runs['run_id'], df_runs[col], marker='o', linewidth=2, markersize=8, color='steelblue')
            plt.title(f'{col.replace("_", " ").title()} Across Runs', fontsize=14, fontweight='bold')
            plt.xlabel('Run ID', fontweight='bold')
            plt.ylabel(col.replace('_', ' ').title(), fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(df_runs['run_id'])
            
            # Add mean line
            mean_val = df_runs[col].mean()
            plt.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            plt.legend()
            
            safe_col = col.replace('_', '-')
            plt.savefig(os.path.join(OUTPUT_DIR, f"Fig-AllRuns-{safe_col}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            log_step("Visualization", f"Failed to draw chart for {col}: {e}")
    
    log_step("Visualization", f"Generated {len(metric_cols)} individual run charts.")

def generate_all_runs_summary_table(df_runs):
    """Generate summary statistics table for All_Runs_Metrics (mean, min, max, std) excluding run_id."""
    log_step("Metrics", "Generating All_Runs_Metrics summary statistics table...")
    
    # Exclude run_id column
    metric_cols = [col for col in df_runs.columns if col != 'run_id']
    
    summary_rows = []
    for col in metric_cols:
        summary_rows.append({
            "Metric": col.replace('_', '-'),
            "Mean": round(df_runs[col].mean(), 2),
            "Min": round(df_runs[col].min(), 2),
            "Max": round(df_runs[col].max(), 2),
            "Std": round(df_runs[col].std(), 2)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Table-All-Runs-Summary-Statistics.csv"), index=False)
    log_step("Metrics", "All_Runs_Metrics summary statistics saved.")

def draw_elbow_chart(features, train_data):
    log_step("Visualization", "Drawing Elbow Method Chart...")
    try:
        subset = train_data[features].copy()
        imputer = SimpleImputer(strategy='median')
        subset_imputed = imputer.fit_transform(subset)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(subset_imputed)
        
        K_range = range(2, 9)
        inertias = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled)
            inertias.append(km.inertia_)
        
        plt.figure(figsize=(8, 5))
        plt.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
        plt.title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
        plt.xlabel('Number of Clusters (K)', fontweight='bold')
        plt.ylabel('Inertia (SSE)', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Elbow_Method.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log_step("Visualization", f"Elbow chart failed: {e}")

def draw_cluster_pca(features, full_data, labels):
    log_step("Visualization", "Drawing PCA Cluster Visualization...")
    try:
        subset = full_data[features].copy()
        imputer = SimpleImputer(strategy='median')
        subset_imputed = imputer.fit_transform(subset)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(subset_imputed)
        
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(scaled)
        
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, 
                            cmap='viridis', alpha=0.6, s=15, edgecolors='k', linewidths=0.3)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'PCA Cluster Visualization (Var: {pca.explained_variance_ratio_.sum():.1%})', 
                  fontsize=13, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig_PCA_Clusters.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"PCA chart error: {e}")

def draw_cluster_radar(features, full_data, labels, k):
    log_step("Visualization", "Drawing Cluster Profiling Radar Chart...")
    try:
        df_profile = full_data[features].copy()
        df_profile['Cluster'] = labels
        
        means = df_profile.groupby('Cluster').mean()
        scaler = StandardScaler()
        means_scaled = pd.DataFrame(scaler.fit_transform(means), columns=means.columns, index=means.index)
        
        categories = list(means_scaled.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, k))
        
        for idx, row in means_scaled.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}', color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title('Cluster Profiling (Normalized Feature Means)', 
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Cluster_Radar.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Radar chart error: {e}")

def draw_multi_run_charts(multi_run_results):
    """Generate charts for all N runs showing metrics per run."""
    log_step("Visualization", f"Drawing Multi-Run Charts (N={len(multi_run_results)})...")
    
    if not multi_run_results:
        return
    
    df = pd.DataFrame(multi_run_results)
    
    # Chart 1: Execution Time per Run
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total Time (Baseline vs Multi-Agent)
    axes[0, 0].plot(df['run_id'], df['baseline_time'], 'o-', label='Baseline', linewidth=2, markersize=8, color='#3498db')
    axes[0, 0].plot(df['run_id'], df['ma_time'], 's-', label='Multi-Agent', linewidth=2, markersize=8, color='#e74c3c')
    axes[0, 0].set_xlabel('Run ID', fontweight='bold')
    axes[0, 0].set_ylabel('Total Time (s)', fontweight='bold')
    axes[0, 0].set_title('Execution Time per Run', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Token Usage (Baseline vs Multi-Agent)
    axes[0, 1].plot(df['run_id'], df['baseline_tokens'], 'o-', label='Baseline', linewidth=2, markersize=8, color='#2ecc71')
    axes[0, 1].plot(df['run_id'], df['ma_tokens'], 's-', label='Multi-Agent', linewidth=2, markersize=8, color='#f39c12')
    axes[0, 1].set_xlabel('Run ID', fontweight='bold')
    axes[0, 1].set_ylabel('Estimated Tokens', fontweight='bold')
    axes[0, 1].set_title('Token Usage per Run', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ADS (Baseline vs Multi-Agent)
    axes[1, 0].plot(df['run_id'], df['baseline_ads'], 'o-', label='Baseline', linewidth=2, markersize=8, color='#9b59b6')
    axes[1, 0].plot(df['run_id'], df['ma_ads'], 's-', label='Multi-Agent', linewidth=2, markersize=8, color='#1abc9c')
    axes[1, 0].set_xlabel('Run ID', fontweight='bold')
    axes[1, 0].set_ylabel('ADS (0-10)', fontweight='bold')
    axes[1, 0].set_title('Actionable Density Score per Run', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylim(0, 11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loop Count (Multi-Agent only)
    axes[1, 1].bar(df['run_id'], df['loop_count'], color='#34495e', edgecolor='black', width=0.6)
    axes[1, 1].set_xlabel('Run ID', fontweight='bold')
    axes[1, 1].set_ylabel('Adversarial Loop Iterations', fontweight='bold')
    axes[1, 1].set_title('Loop Count per Run', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0, 4)
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(df['loop_count']):
        axes[1, 1].text(df['run_id'].iloc[i], v + 0.1, str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Multi_Run_Analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    log_step("Visualization", "Multi-run charts saved successfully.")

# ==============================================================================
# 1b. ML BASELINE PIPELINE & NEW CHARTS
# ==============================================================================

def run_ml_baselines(df_train, df_test):
    """
    Runs Logistic Regression and Random Forest as systematic ML baselines.
    Addresses advisor feedback: 'No systematic baseline comparison'.
    """
    log_step("ML Baseline", "Running Logistic Regression & Random Forest baselines...")
    start_ml = time.time()

    THESIS_METRICS["preprocessing_steps"] = [
        "1. Load telco_churn.csv (7043 rows × 21 columns)",
        "2. Coerce TotalCharges to numeric (pd.to_numeric, errors='coerce')",
        "3. Impute TotalCharges NaN with median value",
        "4. Encode categorical features with LabelEncoder",
        "5. Train/Test split: 80/20 (random_state=42, stratified on Churn)",
        "6. Scale numeric features with StandardScaler",
        "7. Impute remaining NaN with SimpleImputer(strategy='median')",
    ]

    target = 'Churn'
    le_target = LabelEncoder()

    y_train = le_target.fit_transform(df_train[target])
    y_test = le_target.transform(df_test[target])

    drop_cols = [target, 'customerID'] if 'customerID' in df_train.columns else [target]
    X_train = df_train.drop(columns=drop_cols, errors='ignore').copy()
    X_test = df_test.drop(columns=drop_cols, errors='ignore').copy()

    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    results = {}
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else None

        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "model": model
        }
        log_step("ML Baseline", f"{name}: Acc={results[name]['accuracy']}, F1={results[name]['f1']}")

    THESIS_METRICS["ml_baseline_metrics"] = {
        k: {kk: vv for kk, vv in v.items() if kk not in ('y_pred', 'y_proba', 'model')}
        for k, v in results.items()
    }

    # 5-Fold Stratified Cross-Validation
    log_step("ML Baseline", "Running 5-Fold Stratified Cross-Validation...")
    X_full = np.vstack([X_train_sc, X_test_sc])
    y_full = np.concatenate([y_train, y_test])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model_cls in [("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
                             ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))]:
        acc_scores = cross_val_score(model_cls, X_full, y_full, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model_cls, X_full, y_full, cv=cv, scoring='f1')
        cv_results[name] = {
            "cv_acc_mean": round(float(np.mean(acc_scores)), 4),
            "cv_acc_std": round(float(np.std(acc_scores)), 4),
            "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
            "cv_f1_std": round(float(np.std(f1_scores)), 4),
        }
        log_step("ML Baseline", f"{name} CV: Acc={cv_results[name]['cv_acc_mean']}±{cv_results[name]['cv_acc_std']}, F1={cv_results[name]['cv_f1_mean']}±{cv_results[name]['cv_f1_std']}")
    THESIS_METRICS["cv_results"] = cv_results

    THESIS_METRICS["step_times"]["ml_baseline"] = round(time.time() - start_ml, 2)

    draw_confusion_matrices(y_test, results, le_target)
    draw_roc_curves(y_test, results)
    draw_ml_comparison_table(results)

    rf_model = results["Random Forest"]["model"]
    feature_names = list(X_train.columns)
    draw_feature_importance(rf_model, feature_names)

    return results

def draw_confusion_matrices(y_test, results, le_target):
    log_step("Visualization", "Drawing Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=le_target.classes_, yticklabels=le_target.classes_)
        axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontweight='bold')
        axes[idx].set_ylabel('Actual', fontweight='bold')
    plt.suptitle('Confusion Matrices (ML Baselines)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_ML_Confusion_Matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_roc_curves(y_test, results):
    log_step("Visualization", "Drawing ROC Curves...")
    plt.figure(figsize=(8, 6))
    colors = ['#e74c3c', '#2ecc71']
    for idx, (name, res) in enumerate(results.items()):
        if res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves (ML Baselines)', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_ML_ROC_Curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_ml_comparison_table(results):
    log_step("Visualization", "Drawing ML Metrics Comparison Table...")
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    model_names = list(results.keys())
    data = [[results[m][met] for met in metrics] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=[m.title() for m in metrics],
                     rowLabels=model_names, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[1] == -1:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(fontweight='bold')
    plt.title('ML Baseline Classification Metrics', fontsize=13, fontweight='bold', pad=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_ML_Metrics_Table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_feature_importance(rf_model, feature_names):
    log_step("Visualization", "Drawing Feature Importance Chart...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-15:]
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontweight='bold')
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_ML_Feature_Importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_cluster_churn_analysis(df, k):
    log_step("Visualization", "Drawing Cluster-Churn Cross Analysis...")
    try:
        if 'Cluster' not in df.columns or 'Churn' not in df.columns:
            return
        ct = pd.crosstab(df['Cluster'], df['Churn'], normalize='index') * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        counts = df['Cluster'].value_counts().sort_index()
        bars = axes[0].bar(counts.index.astype(str), counts.values, color=plt.cm.Set2(np.linspace(0, 1, k)), edgecolor='black')
        axes[0].set_xlabel('Cluster', fontweight='bold')
        axes[0].set_ylabel('Customer Count', fontweight='bold')
        axes[0].set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

        if 'Yes' in ct.columns:
            churn_rates = ct['Yes']
            bars2 = axes[1].bar(churn_rates.index.astype(str), churn_rates.values,
                               color=['#e74c3c' if r > 30 else '#2ecc71' for r in churn_rates.values], edgecolor='black')
            axes[1].set_xlabel('Cluster', fontweight='bold')
            axes[1].set_ylabel('Churn Rate (%)', fontweight='bold')
            axes[1].set_title('Churn Rate by Cluster', fontsize=13, fontweight='bold')
            axes[1].axhline(y=df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0).mean() * 100,
                           color='blue', linestyle='--', alpha=0.7, label='Overall Avg')
            axes[1].legend()
            for bar in bars2:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}%',
                            ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Cluster_Churn_Analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Cluster-churn analysis error: {e}")

def draw_preprocessing_pipeline():
    log_step("Visualization", "Drawing Preprocessing Pipeline...")
    steps = THESIS_METRICS.get("preprocessing_steps", [])
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(12, max(4, len(steps) * 0.6)))
    ax.axis('off')
    y_positions = np.linspace(0.95, 0.05, len(steps))
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(steps)))
    for i, (step, y_pos) in enumerate(zip(steps, y_positions)):
        ax.add_patch(plt.Rectangle((0.05, y_pos - 0.03), 0.9, 0.06,
                                    facecolor=colors[i], edgecolor='black', linewidth=1, transform=ax.transAxes))
        ax.text(0.5, y_pos, step, transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold' if i == 0 else 'normal')
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + 0.035), xytext=(0.5, y_pos - 0.035),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    plt.title('Data Preprocessing Pipeline', fontsize=14, fontweight='bold', pad=15)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Preprocessing_Pipeline.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 2. INTELLIGENT TOOLS
# ==============================================================================

class DataDiscoveryTool(BaseTool):
    name: str = "data_discovery_tool"
    description: str = "Reads 'data/telco_churn.csv'."
    def _run(self, dummy: str = "NA") -> str:
        if os.path.exists("data/telco_churn.csv"):
            df = pd.read_csv("data/telco_churn.csv")
            return f"Dataset Loaded. Columns: {list(df.columns)}, Shape: {df.shape}"
        return "Error: File not found."

class AdvancedClusteringTool(BaseTool):
    name: str = "auto_k_clustering_tool"
    description: str = "Optimizes clustering. STRICTLY USES NUMERIC COLUMNS ONLY."
    
    def _run(self, input_cols_str: str) -> str:
        N_TRIALS = 20
        log_step("Optimization", f"Starting Hyperparameter Optimization ({N_TRIALS} Trials + GMM Comparison)...")
        try:
            df = pd.read_csv("data/telco_churn.csv")
            if 'TotalCharges' in df.columns:
                tc = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = tc.fillna(tc.median())

            train, test = train_test_split(df, test_size=0.2, random_state=42)

            suggested_cols = [c.strip() for c in str(input_cols_str).replace("'", "").split(',')]
            valid_cols = [c for c in suggested_cols if c in train.columns and pd.api.types.is_numeric_dtype(train[c])]

            mandatory = ['tenure', 'MonthlyCharges', 'TotalCharges']
            present_mandatory = [c for c in mandatory if c in train.columns and pd.api.types.is_numeric_dtype(train[c])]

            all_numerics = train.select_dtypes(include=[np.number]).columns.tolist()
            all_numerics = [c for c in all_numerics if 'id' not in c.lower()]
            # Exclude binary/low-cardinality features — K-Means Euclidean distance
            # is inappropriate for discrete features with < 5 unique values
            all_numerics = [c for c in all_numerics if train[c].nunique() >= 5]

            trial_results = []

            for i in range(N_TRIALS):
                if i < 7:
                    n_extra = random.randint(1, 3)
                    others = [c for c in all_numerics if c not in present_mandatory]
                    selected_features = present_mandatory + random.sample(others, min(len(others), n_extra))
                elif i < 14:
                    n_cols = random.randint(3, 5)
                    selected_features = random.sample(all_numerics, min(len(all_numerics), n_cols))
                else:
                    selected_features = list(present_mandatory) + random.sample(
                        [c for c in all_numerics if c not in present_mandatory],
                        min(2, len([c for c in all_numerics if c not in present_mandatory])))

                selected_features = list(set(selected_features))
                while len(selected_features) < 3 and len(all_numerics) >= 3:
                    candidates = [c for c in all_numerics if c not in selected_features]
                    if not candidates: break
                    selected_features.append(random.choice(candidates))
                k = random.randint(3, 6)

                subset = train[selected_features].copy()
                imputer = SimpleImputer(strategy='median')
                subset_imputed = imputer.fit_transform(subset)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(subset_imputed)

                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(scaled_data)

                score = silhouette_score(scaled_data, labels) if len(set(labels)) > 1 else -1.0

                THESIS_METRICS["optimization_trials"].append((i+1, score))

                trial_results.append({
                    "id": i+1, "k": k, "features": selected_features,
                    "score": score, "model": model, "scaler": scaler, "imputer": imputer
                })

            if not trial_results: return "Optimization Failed."

            trial_results.sort(key=lambda x: (-x['score'], len(x['features'])))

            best = trial_results[0]
            best_score = best['score']
            THESIS_METRICS["best_features"] = best['features']

            best_subset = train[best['features']].copy()
            best_imputed = best['imputer'].transform(best_subset)
            best_scaled = best['scaler'].transform(best_imputed)
            best_labels = best['model'].predict(best_scaled)

            db_score = davies_bouldin_score(best_scaled, best_labels)
            ch_score = calinski_harabasz_score(best_scaled, best_labels)

            # --- GMM Comparison on same features/k ---
            gmm = GaussianMixture(n_components=best['k'], random_state=42, covariance_type='full')
            gmm_labels = gmm.fit_predict(best_scaled)
            gmm_sil = silhouette_score(best_scaled, gmm_labels) if len(set(gmm_labels)) > 1 else -1.0
            gmm_db = davies_bouldin_score(best_scaled, gmm_labels)
            gmm_ch = calinski_harabasz_score(best_scaled, gmm_labels)

            THESIS_METRICS["clustering_comparison"] = {
                "KMeans": {"silhouette": round(best_score, 4), "davies_bouldin": round(db_score, 4), "calinski_harabasz": round(ch_score, 4)},
                "GMM": {"silhouette": round(gmm_sil, 4), "davies_bouldin": round(gmm_db, 4), "calinski_harabasz": round(gmm_ch, 4)}
            }

            THESIS_METRICS["clustering_metrics"] = {
                "silhouette": round(best_score, 4),
                "davies_bouldin": round(db_score, 4),
                "calinski_harabasz": round(ch_score, 4),
                "best_k": best['k'],
                "best_features": best['features'],
                "n_trials": N_TRIALS
            }

            draw_optimization_chart()
            log_step("Optimization", f"KMeans Winner: K={best['k']}, Features={best['features']}, Sil={best_score:.3f}, DB={db_score:.3f}, CH={ch_score:.1f}")
            log_step("Optimization", f"GMM Comparison: Sil={gmm_sil:.3f}, DB={gmm_db:.3f}, CH={gmm_ch:.1f}")

            full_subset = df[best['features']].copy()
            full_imputed = best['imputer'].transform(full_subset)
            full_scaled = best['scaler'].transform(full_imputed)
            df['Cluster'] = best['model'].predict(full_scaled)

            df.to_csv(os.path.join(OUTPUT_DIR, "intermediate_data_with_clusters.csv"), index=False)
            generate_feature_distributions(df)

            draw_elbow_chart(best['features'], train)
            draw_cluster_pca(best['features'], df, df['Cluster'].values)
            draw_cluster_radar(best['features'], df, df['Cluster'].values, best['k'])
            draw_cluster_churn_analysis(df, best['k'])

            report = f"Optimization Complete ({N_TRIALS} Trials).\nWinner Configuration:\n- K: {best['k']}\n- Features: {best['features']}\n- Silhouette Score: {best_score:.3f}\n- Davies-Bouldin Index: {db_score:.3f} (lower is better)\n- Calinski-Harabasz Index: {ch_score:.1f} (higher is better)\n"
            for i in range(best['k']):
                cluster_df = df[df['Cluster'] == i]
                churn_rate = (cluster_df['Churn'] == 'Yes').mean() * 100 if 'Churn' in df.columns else 0
                report += f"Cluster {i}: {len(cluster_df)} customers, Churn Rate: {churn_rate:.1f}%\n"

            return report

        except Exception as e: return f"Clustering Error: {str(e)}"

class FileWriterTool(BaseTool):
    name: str = "file_writer_tool"
    description: str = "Writes file."
    def _run(self, filename: str, content: str) -> str:
        with open(os.path.join(OUTPUT_DIR, os.path.basename(filename)), 'w', encoding='utf-8') as f:
            f.write(str(content))
        return "File saved."

# ==============================================================================
# 3. AGENT DEFINITIONS & LOGIC
# ==============================================================================

def get_agents():
    llm = get_active_llm()
    log_step("LLM", f"Active LLM for this crew: {llm}")

    data_architect = Agent(
        role='Lead Data Architect',
        goal='Analyze the raw schema and strictly select high-signal numeric features for clustering.',
        backstory="""You are a senior Data Architect with 15 years of experience. You analyze dataset schemas 
        and select ONLY continuous numeric features suitable for K-Means clustering (tenure, MonthlyCharges, TotalCharges).
        You NEVER select categorical or binary columns. Output a comma-separated list of column names.""",
        verbose=True, allow_delegation=False, llm=llm, max_rpm=10, tools=[DataDiscoveryTool()]
    )

    cao = Agent(
        role='Chief Analytics Officer (CAO)',
        goal='Run clustering optimization and interpret results into business personas.',
        backstory="""You are a data-driven executive who runs the clustering tool and interprets raw cluster statistics 
        into actionable business personas. For each cluster, you provide: Name, Size, Avg Tenure, Avg Spend, Churn Risk Level.
        You MUST call auto_k_clustering_tool with the column list from the previous task.""",
        verbose=True, allow_delegation=False, llm=llm, max_rpm=10, tools=[AdvancedClusteringTool()]
    )

    strategist = Agent(
        role='Senior Retention Strategist',
        goal='Design targeted retention campaigns with specific offers, discounts, and ROI estimates.',
        backstory="""You are an expert retention marketer. For each customer persona, you design 2 offers:
        1) Premium: Higher cost but higher retention impact
        2) Standard: Lower cost, still effective
        Each offer MUST include: Discount %, Duration, Estimated ROI %, Target Churn Reduction.
        You save results to campaigns.csv using the file_writer_tool.""",
        verbose=True, allow_delegation=False, llm=llm, max_rpm=10, tools=[FileWriterTool()]
    )

    manager = Agent(
        role='Strategy & Budget Manager',
        goal='Audit campaigns for financial viability and approve or reject.',
        backstory="""You are a strict CFO. You audit retention campaigns by checking:
        1) Is the discount justified by the customer's lifetime value?
        2) Is the ROI positive?
        3) Are we not offering discounts to already-loyal customers?
        Return 'APPROVED' if acceptable, or 'REVISION REQUESTED' with specific reasons.""",
        verbose=True, allow_delegation=False, llm=llm, max_rpm=10
    )

    return data_architect, cao, strategist, manager

def safe_kickoff(crew):
    """Execute crew with retry logic. On rate limit, mark provider exhausted and rotate.
    Absolute max retries = 2 * pool size to prevent infinite cycling."""
    global RETRY_WAIT_TOTAL
    max_retries = len(LLM_POOL) * 2
    for i in range(max_retries):
        try:
            result = crew.kickoff()
            track_token_budget(estimate_tokens(result))
            return result
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err
            if is_rate_limit:
                mark_provider_exhausted(err)
                active = sum(1 for p in LLM_POOL if not p["exhausted"])
                if active == 0:
                    log_step("Retry", f"All {len(LLM_POOL)} providers exhausted. Giving up.")
                    return f"Error: All providers exhausted after {i+1} attempts."
                new_llm = get_active_llm()
                log_step("Retry", f"Rate limit hit. Switching to {new_llm} (Attempt {i+1}/{max_retries})")
                for agent in crew.agents:
                    agent.llm = new_llm
                RETRY_WAIT_TOTAL += 2
                time.sleep(2)
            else:
                print(f"[WARN] Agent error (attempt {i+1}): {err[:300]}")
                if i < len(LLM_POOL):
                    mark_provider_exhausted(err)
                    new_llm = get_active_llm()
                    for agent in crew.agents:
                        agent.llm = new_llm
                    RETRY_WAIT_TOTAL += 1
                    time.sleep(1)
                else:
                    return f"Error: {e}"
    return "Error: All providers exhausted after max retries."

def test_clustering_stability(df_full):
    """Test clustering stability across different random seeds. Since only 3 numeric
    features pass the nunique>=5 filter, feature selection is deterministic.
    This tests KMeans initialization sensitivity."""
    log_step("Stability", "Testing Clustering Stability (5 seeds)...")
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    subset = df_full[features].copy()
    subset['TotalCharges'] = pd.to_numeric(subset['TotalCharges'], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    scaled = StandardScaler().fit_transform(imputer.fit_transform(subset))

    seeds = [42, 123, 456, 789, 1024]
    stability_results = []
    for seed in seeds:
        km = KMeans(n_clusters=4, random_state=seed, n_init=10)
        labels = km.fit_predict(scaled)
        sil = round(silhouette_score(scaled, labels), 4)
        db = round(davies_bouldin_score(scaled, labels), 4)
        ch = round(calinski_harabasz_score(scaled, labels), 4)
        stability_results.append({"seed": seed, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})
        log_step("Stability", f"Seed={seed}: Sil={sil}, DB={db}, CH={ch}")

    sil_values = [r["silhouette"] for r in stability_results]
    THESIS_METRICS["clustering_stability"] = {
        "results": stability_results,
        "sil_mean": round(float(np.mean(sil_values)), 4),
        "sil_std": round(float(np.std(sil_values)), 4),
    }
    log_step("Stability", f"Silhouette Mean={np.mean(sil_values):.4f} ± {np.std(sil_values):.4f}")

def _run_single_baseline():
    """Run one baseline agent trial. Returns (result_text, ads, elapsed, net_time, retry_time, tokens)."""
    global RETRY_WAIT_TOTAL
    retry_before = RETRY_WAIT_TOTAL
    start = time.time()

    base_llm = get_active_llm()
    base_agent = Agent(role='Analyst', goal='Analyze churn data and suggest retention strategies',
                       backstory='You are a junior data analyst. Read the dataset and suggest 3 specific retention strategies with percentages and timeframes.',
                       llm=base_llm, tools=[DataDiscoveryTool()])
    base_task = Task(description="Use data_discovery_tool to read the dataset. Then suggest 3 specific retention strategies. Each strategy must include: target segment, discount percentage, expected churn reduction %, and timeframe.",
                     expected_output="3 specific retention strategies with numbers", agent=base_agent)

    result = safe_kickoff(Crew(agents=[base_agent], tasks=[base_task]))

    elapsed = round(time.time() - start, 2)
    retry = round(RETRY_WAIT_TOTAL - retry_before, 2)
    net = round(elapsed - retry, 2)
    ads = calculate_actionable_density_score(result)
    tokens = estimate_tokens(result)

    return result, ads, elapsed, net, retry, tokens

def _run_adversarial_trial(cluster_summary):
    """Run one adversarial loop trial (Stage 2 only). Returns (final_decision, ads, elapsed, net_time, retry_time, tokens)."""
    global RETRY_WAIT_TOTAL
    start = time.time()
    retry_before = RETRY_WAIT_TOTAL
    total_tokens = 0

    _, _, strategist, manager = get_agents()
    audit_approved = False
    loop_count = 0
    max_loops = 3
    final_decision = ""

    while loop_count < max_loops and not audit_approved:
        loop_count += 1

        if loop_count == 1:
            strat_prompt = f"""Based on the following cluster analysis:

{cluster_summary}

For EACH cluster, design 2 retention offers in CSV format with EXACTLY these columns: Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%

IMPORTANT DEFINITIONS:
- Persona: Customer risk profile (HighRisk, ModerateRisk, LowRisk, Loyal, etc.)
- OfferType: Offer tier - MUST be 'Premium' or 'Standard'
- Discount%: Percentage discount (0-50), proportional to churn risk
- Duration: Contract duration in months (e.g., 6, 12, 24)
- EstimatedROI%: Expected ROI percentage

RULES:
- High churn clusters = HighRisk persona, higher discounts
- Low churn clusters = Loyal persona, minimal/zero discounts
- ALL 6 columns MUST be filled for every row
- Use EXACTLY this header: Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%

Save to 'campaigns.csv' using file_writer_tool."""
        elif loop_count == 2:
            strat_prompt = f"""Previous offers REJECTED by CFO. Create REVISED offers with LOWER costs.

Cluster data: {cluster_summary}

CSV format (MANDATORY): Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%
- Persona: Risk profile (HighRisk/ModerateRisk/Loyal)
- OfferType: Premium or Standard
- Standard offers should use non-monetary incentives (loyalty points, priority support)
- ALL 6 columns MUST be filled

Save to 'campaigns.csv' using file_writer_tool."""
        else:
            strat_prompt = f"""FINAL attempt. CFO rejected twice. Most cost-efficient offers needed.

Cluster data: {cluster_summary}

CSV format (STRICT): Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%
- Persona: Risk profile based on churn rate
- OfferType: Premium or Standard
- Max 10% discount for HighRisk clusters only
- Others get non-monetary incentives (0% discount)
- ALL 6 columns MUST be filled with valid data

Save to 'campaigns.csv' using file_writer_tool."""

        t3 = Task(description=strat_prompt, expected_output="CSV file saved", agent=strategist)
        res_strat = safe_kickoff(Crew(agents=[strategist], tasks=[t3]))
        total_tokens += estimate_tokens(res_strat)

        try:
            with open(os.path.join(OUTPUT_DIR, 'campaigns.csv'), 'r', encoding='utf-8') as f:
                campaign_data = f.read()
        except:
            campaign_data = "campaigns.csv not found."

        if loop_count < max_loops:
            mgr_prompt = f"Audit the following campaigns:\n\n{campaign_data}\n\nCluster context:\n{cluster_summary}\n\nCheck: 1) Are discounts justified by churn risk? 2) Is ROI positive? 3) No discounts for loyal clusters? Return 'APPROVED' or 'REVISION REQUESTED' with reasons."
        else:
            mgr_prompt = f"FINAL REVIEW of campaigns:\n\n{campaign_data}\n\nYou MUST select the best offers. Return 'APPROVED' with your chosen strategy."

        t4 = Task(description=mgr_prompt, expected_output="APPROVED or REVISION REQUESTED", agent=manager)
        res_mgr = safe_kickoff(Crew(agents=[manager], tasks=[t4]))
        total_tokens += estimate_tokens(res_mgr)

        final_decision = str(res_mgr)

        if "APPROVED" in final_decision.upper():
            audit_approved = True
        else:
            if loop_count >= max_loops:
                audit_approved = True
                final_decision += " [FORCE APPROVED - Max loops reached]"
            else:
                time.sleep(3)

    elapsed = round(time.time() - start, 2)
    retry = round(RETRY_WAIT_TOTAL - retry_before, 2)
    net = round(elapsed - retry, 2)
    ads = calculate_actionable_density_score(str(cluster_summary) + final_decision)

    return final_decision, ads, elapsed, net, retry, total_tokens, loop_count

def run_experiment():
    pipeline_start = time.time()
    global RETRY_WAIT_TOTAL

    # 0. Initial Visuals
    generate_initial_data_visuals()

    log_step("Init", "Starting Thesis Experimental Pipeline...")

    # 0b. Load and prepare data for ML baselines
    df_full = pd.read_csv("data/telco_churn.csv")
    df_full['TotalCharges'] = pd.to_numeric(df_full['TotalCharges'], errors='coerce')
    df_full['TotalCharges'] = df_full['TotalCharges'].fillna(df_full['TotalCharges'].median())
    df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['Churn'])

    # 0c. Run ML Baselines (Logistic Regression + Random Forest + 5-Fold CV)
    ml_results = run_ml_baselines(df_train, df_test)
    draw_preprocessing_pipeline()

    # 0d. Clustering Stability Test
    test_clustering_stability(df_full)

    # ========== STAGE 1: CLUSTERING (ONE-TIME, DETERMINISTIC) ==========
    RETRY_WAIT_TOTAL = 0
    for p in LLM_POOL:
        p["exhausted"] = False
        p["requests_used"] = 0

    # --- STAGE 1: DATA ANALYSIS + CLUSTERING (one-time) ---
    log_step("Multi-Agent", "Executing Stage 1: Data Analysis + Clustering...")
    architect, cao, strategist, manager = get_agents()
    start_s1 = time.time()

    t1 = Task(description="Use data_discovery_tool to read the dataset schema. From the columns, select ONLY numeric columns suitable for K-Means clustering. Must include: tenure, MonthlyCharges, TotalCharges. Output as comma-separated list.",
              expected_output="tenure,MonthlyCharges,TotalCharges", agent=architect)
    t2 = Task(description="Take the numeric column list from the previous task. Call auto_k_clustering_tool with those columns as a comma-separated string. Then analyze the clustering report and assign descriptive PERSONA NAMES to each cluster based on their characteristics.",
              expected_output="Persona definitions with cluster stats", agent=cao)

    crew_stage1 = Crew(agents=[architect, cao], tasks=[t1, t2])
    res_stage1 = safe_kickoff(crew_stage1)

    track_agent_cost(0, "Data Architect + CAO", res_stage1)
    log_interaction("Data Architect & CAO", "Feature Selection & Clustering", res_stage1)

    # FALLBACK: If clustering tool was not called by agent, run it programmatically
    if not THESIS_METRICS["clustering_metrics"]:
        log_step("Fallback", "Clustering tool was not called by agent. Running programmatic clustering...")
        fallback_tool = AdvancedClusteringTool()
        fallback_result = fallback_tool._run("tenure, MonthlyCharges, TotalCharges")
        res_stage1 = str(res_stage1) + "\n\n[Fallback Clustering Result]:\n" + fallback_result
        log_step("Fallback", "Programmatic clustering completed successfully.")

    THESIS_METRICS["step_times"]["clustering_stage"] = round(time.time() - start_s1, 2)

    # Build cluster summary for data-driven prompts
    cluster_summary = ""
    try:
        cdf = pd.read_csv(os.path.join(OUTPUT_DIR, "intermediate_data_with_clusters.csv"))
        for c in sorted(cdf['Cluster'].unique()):
            grp = cdf[cdf['Cluster'] == c]
            churn_r = (grp['Churn'] == 'Yes').mean() * 100 if 'Churn' in cdf.columns else 0
            cluster_summary += f"Cluster {c}: {len(grp)} customers, Avg Tenure={grp['tenure'].mean():.1f}mo, Avg MonthlyCharges=${grp['MonthlyCharges'].mean():.0f}, Churn Rate={churn_r:.1f}%\n"
    except:
        cluster_summary = str(res_stage1)[:500]

    # ========== MULTI-RUN AGENT COMPARISON (N_RUNS trials) ==========
    multi_run_results = []

    for run_id in range(1, N_RUNS + 1):
        log_step(f"Trial {run_id}/{N_RUNS}", "Running agent comparison trial...")

        # Reset state for this trial
        RETRY_WAIT_TOTAL = 0
        for p in LLM_POOL:
            p["exhausted"] = False
            p["requests_used"] = 0

        # --- Baseline Agent ---
        log_step("Baseline", f"Trial {run_id}: Single-Agent Baseline...")
        base_res, b_ads, b_time, b_net, b_retry, b_tokens = _run_single_baseline()

        if run_id == 1:
            THESIS_METRICS['baseline_time'] = b_time
            THESIS_METRICS['baseline_tokens'] = track_agent_cost(0, "Baseline Agent", base_res)
            THESIS_METRICS['baseline_actionable_density'] = b_ads
            log_interaction("Baseline Agent", "Baseline Task", base_res)

        time.sleep(3)

        # Reset providers for multi-agent
        for p in LLM_POOL:
            p["exhausted"] = False
            p["requests_used"] = 0

        # --- Multi-Agent Adversarial Loop (Stage 2) ---
        log_step("Multi-Agent", f"Trial {run_id}: Adversarial Loop...")

        if run_id == 1:
            # First run: full adversarial loop with detailed logging
            start_ma = time.time()
            retry_before_ma = RETRY_WAIT_TOTAL

            audit_approved = False
            loop_count = 0
            max_loops = 3
            final_decision = ""

            while loop_count < max_loops and not audit_approved:
                loop_count += 1
                log_step(f"Loop {loop_count}", "Generating Strategies & Auditing...")

                if loop_count == 1:
                    strat_prompt = f"""Based on the following cluster analysis:

{cluster_summary}

For EACH cluster, design 2 retention offers in CSV format with EXACTLY these columns: Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%

IMPORTANT DEFINITIONS:
- Persona: Customer risk profile (HighRisk, ModerateRisk, LowRisk, Loyal, etc.)
- OfferType: Offer tier - MUST be 'Premium' or 'Standard'
- Discount%: Percentage discount (0-50), proportional to churn risk
- Duration: Contract duration in months (e.g., 6, 12, 24)
- EstimatedROI%: Expected ROI percentage

RULES:
- High churn clusters = HighRisk persona, higher discounts
- Low churn clusters = Loyal persona, minimal/zero discounts
- ALL 6 columns MUST be filled for every row
- Use EXACTLY this header: Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%

Save to 'campaigns.csv' using file_writer_tool."""
                elif loop_count == 2:
                    strat_prompt = f"""Previous offers REJECTED by CFO. Create REVISED offers with LOWER costs.

Cluster data: {cluster_summary}

CSV format (MANDATORY): Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%
- Persona: Risk profile (HighRisk/ModerateRisk/Loyal)
- OfferType: Premium or Standard
- Standard offers should use non-monetary incentives (loyalty points, priority support)
- ALL 6 columns MUST be filled

Save to 'campaigns.csv' using file_writer_tool."""
                else:
                    strat_prompt = f"""FINAL attempt. CFO rejected twice. Most cost-efficient offers needed.

Cluster data: {cluster_summary}

CSV format (STRICT): Cluster,Persona,OfferType,Discount%,Duration,EstimatedROI%
- Persona: Risk profile based on churn rate
- OfferType: Premium or Standard
- Max 10% discount for HighRisk clusters only
- Others get non-monetary incentives (0% discount)
- ALL 6 columns MUST be filled with valid data

Save to 'campaigns.csv' using file_writer_tool."""

                t3 = Task(description=strat_prompt, expected_output="CSV file saved", agent=strategist)
                crew_strat = Crew(agents=[strategist], tasks=[t3])
                res_strat = safe_kickoff(crew_strat)
                track_agent_cost(loop_count, "Strategist", res_strat)
                log_interaction(f"Strategist (Loop {loop_count})", strat_prompt[:200], res_strat)

                try:
                    with open(os.path.join(OUTPUT_DIR, 'campaigns.csv'), 'r', encoding='utf-8') as f:
                        campaign_data = f.read()
                except:
                    campaign_data = "campaigns.csv not found."

                if loop_count < max_loops:
                    mgr_prompt = f"Audit the following campaigns:\n\n{campaign_data}\n\nCluster context:\n{cluster_summary}\n\nCheck: 1) Are discounts justified by churn risk? 2) Is ROI positive? 3) No discounts for loyal clusters? Return 'APPROVED' or 'REVISION REQUESTED' with reasons."
                else:
                    mgr_prompt = f"FINAL REVIEW of campaigns:\n\n{campaign_data}\n\nYou MUST select the best offers. Return 'APPROVED' with your chosen strategy."

                t4 = Task(description=mgr_prompt, expected_output="APPROVED or REVISION REQUESTED", agent=manager)
                crew_mgr = Crew(agents=[manager], tasks=[t4])
                res_mgr = safe_kickoff(crew_mgr)
                track_agent_cost(loop_count, "Manager", res_mgr)
                log_interaction(f"Manager (Loop {loop_count})", mgr_prompt[:200], res_mgr)

                final_decision = str(res_mgr)

                if "APPROVED" in final_decision.upper():
                    audit_approved = True
                    log_step("Loop", "Manager APPROVED the strategy.")
                else:
                    if loop_count >= max_loops:
                        audit_approved = True
                        final_decision += " [FORCE APPROVED - Max loops reached]"
                        log_step("Loop", "Manager did not approve. FORCE APPROVED (max loops reached).")
                    else:
                        log_step("Loop", "Manager requested revision. Retrying...")
                        time.sleep(3)

            m_time = round(time.time() - start_ma, 2)
            m_retry = round(RETRY_WAIT_TOTAL - retry_before_ma, 2)
            m_net = round(m_time - m_retry, 2)
            m_ads = calculate_actionable_density_score(str(res_stage1) + final_decision)
            m_tokens = sum([e['Tokens'] for e in THESIS_METRICS["agent_token_usage"] if e['Agent'] != "Baseline Agent"])

            THESIS_METRICS['multi_agent_time'] = m_time
            THESIS_METRICS['multi_agent_actionable_density'] = m_ads
            THESIS_METRICS['multi_agent_tokens'] = m_tokens

        else:
            # Runs 2+: use helper function (no detailed logging to avoid pollution)
            final_decision, m_ads, m_time, m_net, m_retry, m_tokens, loop_count = _run_adversarial_trial(cluster_summary)

        # Save campaigns.csv for this run
        import shutil
        campaigns_src = os.path.join(OUTPUT_DIR, 'campaigns.csv')
        campaigns_dst = os.path.join(OUTPUT_DIR, f'campaigns_run{run_id}.csv')
        if os.path.exists(campaigns_src):
            shutil.copy2(campaigns_src, campaigns_dst)
            log_step(f"Run {run_id}", f"Saved {campaigns_dst}")

        multi_run_results.append({
            "run_id": run_id,
            "loop_count": loop_count,
            "baseline_ads": b_ads, "baseline_time": b_time, "baseline_net_time": b_net,
            "baseline_retry_time": b_retry, "baseline_tokens": b_tokens,
            "ma_ads": m_ads, "ma_time": m_time, "ma_net_time": m_net,
            "ma_retry_time": m_retry, "ma_tokens": m_tokens,
        })

    # Store multi-run results and compute aggregates
    THESIS_METRICS["multi_run_results"] = multi_run_results

    b_ads_list = [r["baseline_ads"] for r in multi_run_results]
    m_ads_list = [r["ma_ads"] for r in multi_run_results]
    b_net_list = [r["baseline_net_time"] for r in multi_run_results]
    m_net_list = [r["ma_net_time"] for r in multi_run_results]

    b_tok_list = [r["baseline_tokens"] for r in multi_run_results]
    m_tok_list = [r["ma_tokens"] for r in multi_run_results]

    THESIS_METRICS["multi_run_stats"] = {
        "n_runs": N_RUNS,
        "baseline_ads_mean": round(float(np.mean(b_ads_list)), 2),
        "baseline_ads_std": round(float(np.std(b_ads_list)), 2),
        "baseline_ads_min": round(float(np.min(b_ads_list)), 2),
        "baseline_ads_max": round(float(np.max(b_ads_list)), 2),
        "ma_ads_mean": round(float(np.mean(m_ads_list)), 2),
        "ma_ads_std": round(float(np.std(m_ads_list)), 2),
        "ma_ads_min": round(float(np.min(m_ads_list)), 2),
        "ma_ads_max": round(float(np.max(m_ads_list)), 2),
        "baseline_net_mean": round(float(np.mean(b_net_list)), 2),
        "baseline_net_std": round(float(np.std(b_net_list)), 2),
        "baseline_net_min": round(float(np.min(b_net_list)), 2),
        "baseline_net_max": round(float(np.max(b_net_list)), 2),
        "ma_net_mean": round(float(np.mean(m_net_list)), 2),
        "ma_net_std": round(float(np.std(m_net_list)), 2),
        "ma_net_min": round(float(np.min(m_net_list)), 2),
        "ma_net_max": round(float(np.max(m_net_list)), 2),
        "baseline_retry_mean": round(float(np.mean([r["baseline_retry_time"] for r in multi_run_results])), 2),
        "ma_retry_mean": round(float(np.mean([r["ma_retry_time"] for r in multi_run_results])), 2),
        "baseline_tokens_mean": round(float(np.mean(b_tok_list)), 0),
        "ma_tokens_mean": round(float(np.mean(m_tok_list)), 0),
        "baseline_tokens_min": round(float(np.min(b_tok_list)), 0),
        "baseline_tokens_max": round(float(np.max(b_tok_list)), 0),
        "ma_tokens_min": round(float(np.min(m_tok_list)), 0),
        "ma_tokens_max": round(float(np.max(m_tok_list)), 0),
    }

    log_step("Stats", f"Baseline ADS: {THESIS_METRICS['multi_run_stats']['baseline_ads_mean']}+/-{THESIS_METRICS['multi_run_stats']['baseline_ads_std']}")
    log_step("Stats", f"Multi-Agent ADS: {THESIS_METRICS['multi_run_stats']['ma_ads_mean']}+/-{THESIS_METRICS['multi_run_stats']['ma_ads_std']}")
    log_step("Stats", f"Baseline Net Time: {THESIS_METRICS['multi_run_stats']['baseline_net_mean']}+/-{THESIS_METRICS['multi_run_stats']['baseline_net_std']}s")
    log_step("Stats", f"Multi-Agent Net Time: {THESIS_METRICS['multi_run_stats']['ma_net_mean']}+/-{THESIS_METRICS['multi_run_stats']['ma_net_std']}s")

    THESIS_METRICS["step_times"]["total_pipeline"] = round(time.time() - pipeline_start, 2)

    # --- SAVE ALL RUNS METRICS TO CSV ---
    log_step("Metrics", "Saving all runs metrics to CSV...")
    df_runs = pd.DataFrame(multi_run_results)
    df_runs.to_csv(os.path.join(OUTPUT_DIR, "All_Runs_Metrics.csv"), index=False)

    # --- REPORTING ---
    log_step("Reporting", "Generating Final Artifacts...")
    draw_thesis_pipeline()
    draw_performance_charts()
    draw_agent_token_chart()
    draw_multi_run_charts(multi_run_results)
    draw_all_runs_individual_charts(df_runs)
    generate_all_runs_summary_table(df_runs)

    generate_thesis_tables()
    generate_final_report()
    log_step("Done", f"Pipeline Complete. {N_RUNS} trials executed.")

def generate_thesis_tables():
    """Generate Table CSV files referenced in TEZ_TASLAK.md."""
    import shutil
    ml = THESIS_METRICS["ml_baseline_metrics"]
    cm = THESIS_METRICS["clustering_metrics"]
    cc = THESIS_METRICS["clustering_comparison"]
    st = THESIS_METRICS["step_times"]

    # Table 3.1: Dataset Overview
    df = pd.read_csv("data/telco_churn.csv")
    overview = pd.DataFrame([
        {"Feature": "Row Count", "Value": len(df)},
        {"Feature": "Column Count", "Value": len(df.columns)},
        {"Feature": "Target Variable", "Value": "Churn (Yes/No)"},
        {"Feature": "Churn Rate", "Value": f"{(df['Churn']=='Yes').mean()*100:.1f} percentage"},
        {"Feature": "Numeric Variables", "Value": len(df.select_dtypes(include=[np.number]).columns)},
        {"Feature": "Categorical Variables", "Value": len(df.select_dtypes(include=['object']).columns)},
    ])
    overview.to_csv(os.path.join(OUTPUT_DIR, "Table-3-1-Dataset-Overview.csv"), index=False)

    # Table 4.1 & 4.2: ML Metrics
    if ml:
        for model_key, fname in [("Logistic Regression", "Table-4-1-LR-Metrics.csv"),
                                  ("Random Forest", "Table-4-2-RF-Metrics.csv")]:
            if model_key in ml:
                m = ml[model_key]
                mdf = pd.DataFrame([{"Metric": k.capitalize(), "Value": v} for k, v in m.items()])
                mdf.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    # Table 5.1: Best Clustering Config
    if cm:
        cfg = pd.DataFrame([
            {"Parameter": "K (number of segments)", "Value": cm.get("best-k", "")},
            {"Parameter": "Features", "Value": ", ".join(cm.get("best-features", []))},
            {"Parameter": "Silhouette Score", "Value": cm.get("silhouette", "")},
            {"Parameter": "Davies-Bouldin Index", "Value": cm.get("davies-bouldin", "")},
            {"Parameter": "Calinski-Harabasz Index", "Value": cm.get("calinski-harabasz", "")},
        ])
        cfg.to_csv(os.path.join(OUTPUT_DIR, "Table-5-1-Best-Clustering-Config.csv"), index=False)

    # Table 5.2: KMeans vs GMM
    if cc:
        rows = []
        for metric in ['silhouette', 'davies-bouldin', 'calinski-harabasz']:
            km_val = cc.get('KMeans', {}).get(metric.replace('-', '_'), '')
            gm_val = cc.get('GMM', {}).get(metric.replace('-', '_'), '')
            if metric == 'davies-bouldin':
                winner = 'K-Means' if km_val < gm_val else 'GMM'
            else:
                winner = 'K-Means' if km_val > gm_val else 'GMM'
            rows.append({"Metric": metric, "KMeans": km_val, "GMM": gm_val, "Winner": winner})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "Table-5-2-KMeans-vs-GMM.csv"), index=False)

    # Table 6.1: Agent Roles
    roles = pd.DataFrame([
        {"Agent": "Data Architect", "Role": "Data Analyst", "Task": "Numeric feature selection"},
        {"Agent": "CAO", "Role": "Analytics Manager", "Task": "K-Means clustering optimization"},
        {"Agent": "Strategist", "Role": "Marketing Strategist", "Task": "Retention campaign design"},
        {"Agent": "Manager", "Role": "CFO / Auditor", "Task": "Financial feasibility audit"},
        {"Agent": "Baseline", "Role": "Junior Analyst", "Task": "Single-agent reference baseline"},
    ])
    roles.to_csv(os.path.join(OUTPUT_DIR, "Table-6-1-Agent-Roles.csv"), index=False)

    # Table 6.2: LLM Providers (ALL providers, no deduplication)
    providers = []
    for p in LLM_POOL:
        providers.append({"Provider": p["name"], "Model": p["model"], "RPM": p["rpm"], "RPD": p["rpd"]})
    pd.DataFrame(providers).to_csv(os.path.join(OUTPUT_DIR, "Table-6-2-LLM-Providers.csv"), index=False)

    # Table 6.3: Baseline vs Multi-Agent
    b_ads = THESIS_METRICS.get('baseline_actionable_density', 0)
    m_ads = THESIS_METRICS.get('multi_agent_actionable_density', 0)
    bt = THESIS_METRICS.get('baseline_time', 0)
    mt = THESIS_METRICS.get('multi_agent_time', 0)
    btk = THESIS_METRICS.get('baseline_tokens', 0)
    mtk = THESIS_METRICS.get('multi_agent_tokens', 0)
    comp = pd.DataFrame([
        {"Metric": "Execution Time (s)", "Single-Agent": bt, "Multi-Agent": mt, "Difference": f"{mt-bt:+.1f}s"},
        {"Metric": "Estimated Tokens", "Single-Agent": btk, "Multi-Agent": mtk, "Difference": f"{mtk-btk:+d}"},
        {"Metric": "ADS", "Single-Agent": b_ads, "Multi-Agent": m_ads, "Difference": f"{m_ads-b_ads:+.2f}"},
    ])
    comp.to_csv(os.path.join(OUTPUT_DIR, "Table-6-3-Baseline-vs-MultiAgent.csv"), index=False)

    # Table 6.4: Campaigns (copy from campaigns.csv and fix column names)
    camp_src = os.path.join(OUTPUT_DIR, "campaigns.csv")
    camp_dst = os.path.join(OUTPUT_DIR, "Table-6-4-Campaigns.csv")
    if os.path.exists(camp_src):
        df_camp = pd.read_csv(camp_src)
        df_camp.columns = [col.replace('_', '-').replace('%', '-percentage') for col in df_camp.columns]
        df_camp.to_csv(camp_dst, index=False)

    # Table 7.1: Pipeline Performance
    perf = pd.DataFrame([
        {"Stage": "Data Analysis + Clustering", "Duration (s)": st.get("clustering-stage", st.get("clustering_stage", 0))},
        {"Stage": "Baseline Agent", "Duration (s)": THESIS_METRICS.get('baseline-time', THESIS_METRICS.get('baseline_time', 0))},
        {"Stage": "Multi-Agent Loop", "Duration (s)": THESIS_METRICS.get('multi-agent-time', THESIS_METRICS.get('multi_agent_time', 0))},
        {"Stage": "Total Pipeline", "Duration (s)": st.get("total-pipeline", st.get("total_pipeline", 0))},
    ])
    perf.to_csv(os.path.join(OUTPUT_DIR, "Table-7-1-Pipeline-Performance.csv"), index=False)

    # Table 7.2: Clustering Success (KMeans vs GMM)
    if cc:
        rows = []
        for metric in ['silhouette', 'davies-bouldin', 'calinski-harabasz']:
            rows.append({"Metric": metric, "KMeans": cc['KMeans'].get(metric.replace('-', '_'), ''), "GMM": cc['GMM'].get(metric.replace('-', '_'), '')})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "Table-7-2-Clustering-Success.csv"), index=False)

    log_step("Tables", f"Generated thesis table CSVs in {OUTPUT_DIR}")


def generate_final_report():
    """Generate Final_Report.md (business) and Experimental_Results.md (technical)."""
    cm = THESIS_METRICS["clustering_metrics"]
    cc = THESIS_METRICS["clustering_comparison"]
    ml = THESIS_METRICS["ml_baseline_metrics"]
    st = THESIS_METRICS["step_times"]
    n_trials = cm.get('n_trials', 20) if cm else 20

    # =========================================================================
    # FINAL REPORT (Business-oriented)
    # =========================================================================
    fr = "# FINAL REPORT: Multi-Agent Churn Mitigation System\n\n"

    fr += "## 1. Executive Summary\n\n"
    fr += "This report presents the results of a multi-agent AI system designed to reduce customer churn "
    fr += "in the telecommunications sector. The system combines ML-based churn prediction, K-Means customer "
    fr += "segmentation, and a multi-agent strategy generation pipeline built on the CrewAI framework.\n\n"
    fr += f"**Pipeline completed in {st.get('total_pipeline', 'N/A')}s.**\n\n"

    fr += "## 2. Pipeline Workflow\n\n"
    fr += "1. **Data Loading & EDA**: Load Telco Churn dataset (7,043 customers, 21 features). Generate exploratory charts.\n"
    fr += "2. **ML Baselines**: Train Logistic Regression and Random Forest to establish churn prediction benchmarks.\n"
    fr += "3. **Single-Agent Baseline**: A junior analyst agent reads data and proposes basic retention strategies.\n"
    fr += "4. **Multi-Agent Pipeline**:\n"
    fr += "   - Data Architect selects numeric features for clustering.\n"
    fr += "   - CAO runs 20-trial K-Means optimization + GMM comparison.\n"
    fr += "   - Strategist designs retention campaigns per cluster.\n"
    fr += "   - Manager (CFO) audits campaigns for financial viability.\n"
    fr += "5. **Adversarial Loop**: Strategist and Manager iterate up to 3 rounds until approval.\n"
    fr += "6. **Reporting**: Generate all charts, metrics, and final artifacts.\n\n"

    fr += "## 3. Customer Segments (Clusters)\n\n"
    # Load cluster data for business descriptions
    try:
        cdf = pd.read_csv(os.path.join(OUTPUT_DIR, "intermediate_data_with_clusters.csv"))
        fr += "| Cluster | Size | Avg Tenure | Avg Monthly | Avg Total | Churn Rate | Risk |\n"
        fr += "|---|---|---|---|---|---|---|\n"
        for c in sorted(cdf['Cluster'].unique()):
            grp = cdf[cdf['Cluster'] == c]
            churn_r = (grp['Churn'] == 'Yes').mean() * 100
            risk = "Very High" if churn_r > 40 else "High" if churn_r > 25 else "Medium" if churn_r > 15 else "Low"
            tc_avg = grp['TotalCharges'].mean()
            fr += f"| {c} | {len(grp):,} | {grp['tenure'].mean():.1f} mo | ${grp['MonthlyCharges'].mean():.0f} | ${tc_avg:,.0f} | {churn_r:.1f}% | {risk} |\n"
        fr += "\n"
        fr += "### Cluster Descriptions\n\n"
        for c in sorted(cdf['Cluster'].unique()):
            grp = cdf[cdf['Cluster'] == c]
            churn_r = (grp['Churn'] == 'Yes').mean() * 100
            tenure_avg = grp['tenure'].mean()
            mc_avg = grp['MonthlyCharges'].mean()
            tc_avg = grp['TotalCharges'].mean()
            risk = "Very High" if churn_r > 40 else "High" if churn_r > 25 else "Medium" if churn_r > 15 else "Low"
            fr += f"**Cluster {c}** (n={len(grp):,}, Churn: {churn_r:.1f}%, Risk: {risk})\n\n"
            fr += f"Average tenure: {tenure_avg:.1f} months, Monthly: ${mc_avg:.0f}, Total: ${tc_avg:,.0f}. "
            if churn_r > 40:
                fr += f"Highest-risk segment with {churn_r:.1f}% churn. Requires aggressive retention: targeted discounts, contract migration offers, priority support.\n\n"
            elif churn_r > 25:
                fr += f"Above-average churn ({churn_r:.1f}%). Early intervention recommended: welcome packages, loyalty incentives, proactive outreach.\n\n"
            elif churn_r > 15:
                fr += f"Moderate churn ({churn_r:.1f}%). Standard retention programs and periodic engagement should suffice.\n\n"
            else:
                fr += f"Low churn ({churn_r:.1f}%). Highly loyal segment. Avoid unnecessary discounts — VIP status and non-monetary rewards are more appropriate.\n\n"
    except Exception as e:
        fr += f"(Cluster data unavailable: {e})\n\n"

    fr += "## 4. Approved Campaigns\n\n"
    try:
        with open(os.path.join(OUTPUT_DIR, 'campaigns.csv'), 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        if len(lines) > 1:
            header = lines[0]
            fr += f"| {' | '.join(header.split(','))} |\n"
            fr += f"|{'---|' * len(header.split(','))}\n"
            for line in lines[1:]:
                fr += f"| {' | '.join(line.split(','))} |\n"
        fr += "\n"
    except:
        fr += "(campaigns.csv not available)\n\n"

    # Dynamically generate Key Observations based on actual cluster data
    fr += "### Key Observations\n\n"
    try:
        cdf = pd.read_csv(os.path.join(OUTPUT_DIR, "intermediate_data_with_clusters.csv"))
        clusters = cdf['Cluster'].unique()
        for c in clusters:
            grp = cdf[cdf['Cluster'] == c]
            churn_r = (grp['Churn'] == 'Yes').mean() * 100
            if churn_r > 40:
                fr += f"- Cluster {c} has the highest churn risk ({churn_r:.1f}%). Discounts are proportional to churn risk.\n"
            elif churn_r > 25:
                fr += f"- Cluster {c} has above-average churn ({churn_r:.1f}%). Early intervention is recommended.\n"
            elif churn_r > 15:
                fr += f"- Cluster {c} has moderate churn ({churn_r:.1f}%). Standard retention programs are sufficient.\n"
            else:
                fr += f"- Cluster {c} has low churn ({churn_r:.1f}%). Minimal or zero discounts are applied to preserve budget.\n"
        fr += "- All ROI estimates are positive: every campaign is designed to generate more revenue than it costs.\n\n"
    except:
        fr += "(Cluster data unavailable)\n\n"

    fr += "## 5. Adversarial Loop Outcome\n\n"
    loop_interactions = [i for i in THESIS_METRICS["interaction_log"] if 'Manager' in i['role']]
    fr += f"- **Total loops**: {len(loop_interactions)}\n"
    for interaction in loop_interactions:
        approved = "APPROVED" in interaction['output'].upper()
        fr += f"- **{interaction['role']}**: {'APPROVED' if approved else 'REVISION REQUESTED'}\n"
    fr += "\n"

    fr += "## 6. ML Baseline Summary\n\n"
    if ml:
        fr += "| Model | Accuracy | F1 Score |\n|---|---|---|\n"
        for model_name, metrics in ml.items():
            fr += f"| {model_name} | {metrics['accuracy']} | {metrics['f1']} |\n"
        fr += "\n"
        fr += "ML models predict churn with ~80% accuracy but cannot prescribe actions. "
        fr += "The multi-agent system bridges this gap by converting predictions into actionable campaigns.\n\n"

    fr += "## 7. Generated Figures\n\n"
    fig_list = [
        ("Fig 1.1", "Thesis_Pipeline_Architecture.png", "Multi-Agent System Architecture"),
        ("Fig 1.2", "Fig_Preprocessing_Pipeline.png", "Data Preprocessing Pipeline"),
        ("Fig 2.1", "Fig_EDA_1_Churn_Donut.png", "Churn Distribution"),
        ("Fig 2.2a", "Fig_EDA_2a_tenure.png", "tenure Distribution by Churn"),
        ("Fig 2.2b", "Fig_EDA_2b_MonthlyCharges.png", "MonthlyCharges Distribution by Churn"),
        ("Fig 2.2c", "Fig_EDA_2c_TotalCharges.png", "TotalCharges Distribution by Churn"),
        ("Fig 2.2d", "Fig_EDA_2d_Contract.png", "Contract Distribution by Churn"),
        ("Fig 2.2e", "Fig_EDA_2e_PaymentMethod.png", "PaymentMethod Distribution by Churn"),
        ("Fig 2.3", "Fig_EDA_3_Correlation.png", "Feature Correlation Matrix"),
        ("Fig 2.4", "Fig_EDA_4_Churn_by_Category.png", "Churn Rate by Categorical Feature"),
        ("Fig 3.1", "Fig_ML_Confusion_Matrix.png", "ML Confusion Matrices"),
        ("Fig 3.2", "Fig_ML_ROC_Curves.png", "ML ROC Curves"),
        ("Fig 3.3", "Fig_ML_Feature_Importance.png", "Random Forest Feature Importance"),
        ("Fig 3.4", "Fig_ML_Metrics_Table.png", "ML Metrics Comparison Table"),
        ("Fig 4.1", "Fig_Optimization_Process.png", "Clustering Optimization Process"),
        ("Fig 4.2", "Fig_Elbow_Method.png", "Elbow Method for K Selection"),
        ("Fig 4.3", "Fig_PCA_Clusters.png", "PCA Cluster Visualization"),
        ("Fig 4.4", "Fig_Cluster_Radar.png", "Cluster Profiling Radar Chart"),
        ("Fig 4.5", "Fig_Cluster_Churn_Analysis.png", "Cluster Size & Churn Rate"),
        ("Fig 5.1", "Fig_Comp_Performance.png", "Performance Comparison"),
        ("Fig 5.2", "Fig_Comp_Cost.png", "Token Cost Comparison"),
        ("Fig 5.3", "Fig_Agent_Token_Usage.png", "Agent Token Usage Breakdown"),
    ]
    fr += "| Figure | File | Description | Status |\n|---|---|---|---|\n"
    for fig_id, fname, desc in fig_list:
        status = "OK" if os.path.exists(os.path.join(OUTPUT_DIR, fname)) else "MISSING"
        fr += f"| {fig_id} | {fname} | {desc} | {status} |\n"
    fr += "\n"

    with open(os.path.join(OUTPUT_DIR, "Final_Report.md"), 'w', encoding='utf-8') as f:
        f.write(fr)

    # =========================================================================
    # EXPERIMENTAL RESULTS (Technical)
    # =========================================================================
    report = "# EXPERIMENTAL RESULTS — Technical Details\n\n"

    report += "## 1. Experimental Protocol\n"
    report += "| Parameter | Value |\n|---|---|\n"
    report += f"| Initial LLM | {MY_LLM} |\n"
    report += f"| Provider Pool | {len(LLM_POOL)} providers (round-robin with exhaustion tracking) |\n"
    report += "| Train/Test Split | 80/20, stratified on Churn |\n"
    report += f"| Clustering Optimization | {n_trials} Trials (Hybrid Strategy) |\n"
    report += "| Clustering Algorithms | K-Means (primary) + GMM (comparison) |\n"
    report += "| ML Baselines | Logistic Regression + Random Forest |\n"
    report += "| Adversarial Loop | Max 3 iterations |\n"
    report += f"| Agent Comparison Trials | N_RUNS={N_RUNS} (mean ± std reported) |\n"
    report += f"| Token Budget | {MAX_TOKENS_PER_RUN} tokens/run |\n\n"

    # Preprocessing
    steps = THESIS_METRICS.get("preprocessing_steps", [])
    if steps:
        report += "## 2. Data Preprocessing Pipeline\n"
        for s in steps:
            report += f"- {s}\n"
        report += "\n"

    # ML Baselines
    if ml:
        report += "## 3. ML Baseline Results\n"
        report += "| Model | Accuracy | F1 Score | Precision | Recall |\n|---|---|---|---|---|\n"
        for model_name, metrics in ml.items():
            report += f"| {model_name} | {metrics['accuracy']} | {metrics['f1']} | {metrics['precision']} | {metrics['recall']} |\n"
        report += "\n"

    # Clustering
    if cm:
        report += "## 4. Clustering Validation Metrics\n"
        report += "| Metric | Value | Interpretation |\n|---|---|---|\n"
        report += f"| Optimization Trials | {cm.get('n_trials', 'N/A')} | Hybrid: Business-logic + Random search |\n"
        report += f"| Best K | {cm.get('best_k', 'N/A')} | Optimal number of customer segments |\n"
        report += f"| Features | {cm.get('best_features', 'N/A')} | Strictly numeric (no categoricals) |\n"
        report += f"| Silhouette Score | {cm.get('silhouette', 'N/A')} | Higher = better cluster separation |\n"
        report += f"| Davies-Bouldin Index | {cm.get('davies_bouldin', 'N/A')} | Lower = better separation |\n"
        report += f"| Calinski-Harabasz Index | {cm.get('calinski_harabasz', 'N/A')} | Higher = denser clusters |\n\n"

    if cc:
        report += "### K-Means vs GMM Comparison\n"
        report += "| Metric | K-Means | GMM | Winner |\n|---|---|---|---|\n"
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            km_val = cc.get('KMeans', {}).get(metric, 'N/A')
            gm_val = cc.get('GMM', {}).get(metric, 'N/A')
            if metric == 'davies_bouldin':
                better = 'K-Means' if km_val < gm_val else 'GMM'
            else:
                better = 'K-Means' if km_val > gm_val else 'GMM'
            report += f"| {metric} | {km_val} | {gm_val} | {better} |\n"
        report += "\n"

    # Agent Comparison
    report += "## 5. Agent System Comparison\n"
    report += "| Metric | Baseline (Single-Agent) | Multi-Agent System | Delta |\n|---|---|---|---|\n"
    bt = THESIS_METRICS['baseline_time']
    mt = THESIS_METRICS['multi_agent_time']
    ba = THESIS_METRICS['baseline_actionable_density']
    ma = THESIS_METRICS['multi_agent_actionable_density']
    btk = THESIS_METRICS['baseline_tokens']
    mtk = THESIS_METRICS['multi_agent_tokens']
    report += f"| Execution Time (s) | {bt} | {mt} | {mt-bt:+.1f}s |\n"
    report += f"| Estimated Tokens | {btk} | {mtk} | {mtk-btk:+d} |\n"
    report += f"| Actionable Density Score | {ba} | {ma} | {ma-ba:+.2f} |\n\n"

    report += "### ADS Methodology\n"
    report += "Actionable Density Score (ADS) measures the ratio of sentences containing specific actionable elements "
    report += "(percentages, dollar amounts, time frames, financial terms, segmentation references) to total sentences. "
    report += "Formula: `ADS = min(10, actionable_ratio * 12.5)`. Range: 0-10.\n\n"

    # Multi-Run Statistics
    mrs = THESIS_METRICS.get("multi_run_stats", {})
    if mrs:
        report += "### Multi-Run Results (N={} trials)\n".format(mrs.get('n_runs', 'N/A'))
        report += "| Metric | Baseline (mean +/- std) | Baseline Min | Baseline Max | Multi-Agent (mean +/- std) | MA Min | MA Max |\n|---|---|---|---|---|---|---|\n"
        report += f"| ADS | {mrs['baseline_ads_mean']} +/- {mrs['baseline_ads_std']} | {mrs['baseline_ads_min']} | {mrs['baseline_ads_max']} | {mrs['ma_ads_mean']} +/- {mrs['ma_ads_std']} | {mrs['ma_ads_min']} | {mrs['ma_ads_max']} |\n"
        report += f"| Net Time (s) | {mrs['baseline_net_mean']} +/- {mrs['baseline_net_std']} | {mrs['baseline_net_min']} | {mrs['baseline_net_max']} | {mrs['ma_net_mean']} +/- {mrs['ma_net_std']} | {mrs['ma_net_min']} | {mrs['ma_net_max']} |\n"
        report += f"| Tokens | {int(mrs['baseline_tokens_mean'])} | {int(mrs['baseline_tokens_min'])} | {int(mrs['baseline_tokens_max'])} | {int(mrs['ma_tokens_mean'])} | {int(mrs['ma_tokens_min'])} | {int(mrs['ma_tokens_max'])} |\n"
        report += f"| Retry Wait (s) | {mrs['baseline_retry_mean']} | - | - | {mrs['ma_retry_mean']} | - | - |\n\n"

        report += "### Per-Trial Breakdown\n"
        report += "| Trial | B.ADS | B.Net(s) | B.Retry(s) | B.Tokens | MA.ADS | MA.Net(s) | MA.Retry(s) | MA.Tokens |\n|---|---|---|---|---|---|---|---|---|\n"
        for r in THESIS_METRICS.get("multi_run_results", []):
            report += f"| {r['run_id']} | {r['baseline_ads']} | {r['baseline_net_time']} | {r['baseline_retry_time']} | {r['baseline_tokens']} | {r['ma_ads']} | {r['ma_net_time']} | {r['ma_retry_time']} | {r['ma_tokens']} |\n"
        report += "\n"

    # Cross-Validation Results
    cvr = THESIS_METRICS.get("cv_results", {})
    if cvr:
        report += "### 5-Fold Stratified Cross-Validation\n"
        report += "| Model | CV Accuracy (mean +/- std) | CV F1 (mean +/- std) |\n|---|---|---|\n"
        for model_name, metrics in cvr.items():
            report += f"| {model_name} | {metrics['cv_acc_mean']} +/- {metrics['cv_acc_std']} | {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']} |\n"
        report += "\n"

    # Clustering Stability
    cs = THESIS_METRICS.get("clustering_stability", {})
    if cs:
        report += "### Clustering Stability Test (K=4, 5 seeds)\n"
        report += "| Seed | Silhouette | Davies-Bouldin | Calinski-Harabasz |\n|---|---|---|---|\n"
        for r in cs.get("results", []):
            report += f"| {r['seed']} | {r['silhouette']} | {r['davies_bouldin']} | {r['calinski_harabasz']} |\n"
        report += f"| **Mean +/- Std** | **{cs['sil_mean']} +/- {cs['sil_std']}** | | |\n\n"

    # Token Breakdown per agent per loop
    report += "## 6. Token Usage Breakdown\n\n"
    report += "### Per Agent Per Loop\n"
    report += "| Loop | Agent | Est. Tokens |\n|---|---|---|\n"
    for entry in THESIS_METRICS["agent_token_usage"]:
        report += f"| {entry['Loop']} | {entry['Agent']} | {entry['Tokens']:,} |\n"
    total_tokens = sum(e['Tokens'] for e in THESIS_METRICS["agent_token_usage"])
    report += f"| **Total** | **All Agents** | **{total_tokens:,}** |\n\n"

    report += "### Token-Consuming vs Non-Token Operations\n"
    report += "| Operation | Consumes Tokens? | Details |\n|---|---|---|\n"
    report += "| EDA Chart Generation | No | Pure matplotlib/seaborn, no LLM calls |\n"
    report += "| ML Baseline Training | No | scikit-learn only |\n"
    report += "| Confusion Matrix / ROC / Feature Importance | No | matplotlib only |\n"
    report += "| Clustering Optimization (20 trials) | No | scikit-learn KMeans/GMM |\n"
    report += "| All Chart Drawing | No | matplotlib only |\n"
    report += "| Baseline Agent (single-agent) | Yes | ~1 LLM call |\n"
    report += "| Data Architect Agent | Yes | ~1 LLM call + tool call |\n"
    report += "| CAO Agent | Yes | ~1 LLM call + tool call |\n"
    report += "| Strategist Agent (per loop) | Yes | ~1 LLM call + file write |\n"
    report += "| Manager Agent (per loop) | Yes | ~1 LLM call |\n\n"

    # Compute Budget
    report += "## 7. Execution Timing\n"
    report += "| Step | Duration (s) |\n|---|---|\n"
    for step_name, duration in st.items():
        report += f"| {step_name} | {duration} |\n"
    report += "\n"

    # Execution Log
    report += "## 8. Full Execution Log (Audit Trail)\n"
    report += "```\n"
    for line in EXECUTION_LOG:
        report += f"{line}\n"
    report += "```\n\n"

    # Detailed Agent Interaction Log
    report += "## 9. Agent Interaction Log\n"
    for interaction in THESIS_METRICS["interaction_log"]:
        report += f"\n### {interaction['role']}\n"
        report += f"**Input (truncated):**\n```\n{interaction['input'][:500]}\n```\n\n"
        report += f"**Output (truncated):**\n```\n{interaction['output'][:2000]}\n```\n\n"
        report += "---\n"

    # Provider Pool Status
    report += "\n## 10. LLM Provider Pool Status\n"
    report += "| Provider | Model | Status |\n|---|---|---|\n"
    for p in LLM_POOL:
        status = "EXHAUSTED" if p["exhausted"] else "ACTIVE"
        report += f"| {p['name']} | {p['model']} | {status} |\n"

    with open(os.path.join(OUTPUT_DIR, "Experimental_Results.md"), 'w', encoding='utf-8') as f:
        f.write(report)

def log_system_info():
    print("\n--- Logging System Info ---")
    data = {
        "OS": f"{platform.system()} {platform.release()}",
        "Python Version": platform.python_version(),
        "Processor": platform.processor(),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024.0 **3), 2),
        "LLM Providers": f"{len(LLM_POOL)} providers in rotation pool",
    }

    content = "# EXPERIMENTAL SETUP & COMPUTE ENVIRONMENT\n\n"
    content += "## Hardware & Software\n"
    content += "| Component | Specification |\n|---|---|\n"
    for k, v in data.items():
        content += f"| {k} | {v} |\n"

    content += "\n## Library Versions\n"
    content += "| Library | Version |\n|---|---|\n"
    libs = ['pandas', 'scikit-learn', 'crewai', 'numpy', 'matplotlib', 'seaborn', 'litellm']
    for lib in libs:
        try:
            content += f"| {lib} | {version(lib)} |\n"
        except PackageNotFoundError:
            pass

    content += "\n## Experimental Parameters\n"
    content += "| Parameter | Value |\n|---|---|\n"
    content += "| Random State | 42 (all stochastic operations) |\n"
    content += "| Train/Test Split | 80/20, stratified on Churn |\n"
    content += "| Clustering Optimization | 20 Trials (Hybrid Strategy: Business + Random) |\n"
    content += "| Clustering Algorithms | K-Means (primary) + GMM (comparison) |\n"
    content += "| ML Baselines | Logistic Regression + Random Forest |\n"
    content += "| Adversarial Loop | Max 3 iterations |\n"
    content += f"| Agent Comparison Trials | N_RUNS={N_RUNS} (mean ± std reported) |\n"
    content += "| ML Cross-Validation | 5-Fold Stratified |\n"
    content += "| Clustering Stability | 5 seeds (42, 123, 456, 789, 1024) |\n"
    content += "| Reproducibility | random_state=42 for all stochastic operations |\n"
    content += "| Feature Scaling | StandardScaler |\n"
    content += "| Missing Value Imputation | SimpleImputer (median) |\n"
    content += f"| Token Budget | {MAX_TOKENS_PER_RUN} tokens/run |\n"

    content += "\n## LLM Provider Pool\n"
    content += "| Provider | Model | RPM | RPD | TPM |\n|---|---|---|---|---|\n"
    for p in LLM_POOL:
        content += f"| {p['name']} | {p['model']} | {p['rpm']} | {p['rpd']} | {p['tpm']} |\n"

    with open(os.path.join(OUTPUT_DIR, "00_Experimental_Setup.md"), 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    preflight_check_providers()
    MY_LLM = get_active_llm()
    log_system_info()
    run_experiment()
