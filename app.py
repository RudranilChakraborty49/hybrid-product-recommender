
import torch, pickle, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import gradio as gr

device = "cpu"   # HF free tier is CPU; our model is tiny so this is fine

# --- model definition (must match what we trained) ---
class ResidualFusionGNN(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
        self.text_proj = nn.Linear(in_dim, out_dim)
    def forward(self, x, edge_index):
        h = self.dropout(F.relu(self.conv1(x, edge_index)))
        h = self.conv2(h, edge_index)
        return h + self.text_proj(x)

# --- load artifacts ---
data = torch.load("graph_data.pt", weights_only=False, map_location=device)
with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)
texts = meta["texts"]

model = ResidualFusionGNN().to(device)
model.load_state_dict(torch.load("residual_fusion_model.pt", map_location=device))
model.eval()

# --- precompute the two pathways for explainability ---
with torch.no_grad():
    x, ei = data.x.to(device), data.edge_index.to(device)
    h = model.dropout(F.relu(model.conv1(x, ei)))
    h = model.conv2(h, ei)
    t = model.text_proj(x)
    full = F.normalize(h + t, dim=-1)
    h_n = F.normalize(h, dim=-1)
    t_n = F.normalize(t, dim=-1)

def recommend(query_idx, topk=8):
    scores = full[query_idx] @ full.T
    scores[query_idx] = -float("inf")
    top = torch.argsort(scores, descending=True)[:topk].tolist()
    out = []
    for ti in top:
        tc = abs((t_n[query_idx] @ t_n[ti]).item())
        gc = abs((h_n[query_idx] @ h_n[ti]).item())
        tot = tc + gc + 1e-9
        out.append((texts[ti][:70], scores[ti].item(),
                    100*tc/tot, 100*gc/tot))
    return out

def bar(pct, color):
    w = max(2, int(pct * 1.6))
    return (f"<span style='display:inline-block;background:{color};width:{w}px;"
            f"height:11px;border-radius:3px;vertical-align:middle;margin-left:6px'></span>")

choices = [(texts[i][:60], i) for i in range(len(texts))]

def demo_fn(idx):
    if idx is None:
        return "Pick a product."
    recs = recommend(idx)
    out = f"### Because you viewed:\n**{texts[idx][:90]}**\n\n---\n### Recommended\n\n"
    for i,(title,sc,tp,gp) in enumerate(recs,1):
        out += (f"**{i}. {title}**  \n&nbsp;&nbsp;similarity **{sc:.3f}**  \n"
                f"&nbsp;&nbsp;📝 text {tp:.0f}% {bar(tp,'#4C72B0')}  \n"
                f"&nbsp;&nbsp;🔗 co-purchase {gp:.0f}% {bar(gp,'#DD8452')}\n\n")
    return out

with gr.Blocks(title="Hybrid Product Recommender") as demo:
    gr.Markdown("# 🛒 Hybrid GNN + Transformer Product Recommender\n"
                "Combines a **co-purchase graph** (GNN) with **product text** (BERT). "
                "Each result shows which signal drove it.\n\n"
                "*Split shows which signal leaned heavier, not a hard causal split.*")
    dd = gr.Dropdown(choices=choices, label="Select a product", filterable=True, value=1000)
    dd.change(demo_fn, inputs=dd, outputs=gr.Markdown())
    btn = gr.Button("Get Recommendations", variant="primary")
    out = gr.Markdown()
    btn.click(demo_fn, inputs=dd, outputs=out)

demo.launch()
