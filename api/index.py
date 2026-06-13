"""
Vercel serverless function wrapper for the Streamlit app.
Note: This is a simplified API endpoint. Full Streamlit deployment requires additional configuration.
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "message": "TacticalNet API - World Cup 2026 Match Predictor",
            "status": "active",
            "endpoints": {
                "/api": "This info page",
                "/api/predict": "POST endpoint for match predictions (body: {team_a, team_b, form_a, form_b, ppda_a, ppda_b, tilt_a, tilt_b, tactic_a, tactic_b})"
            },
            "note": "For full interactive dashboard, consider deploying to Streamlit Cloud or using Docker on platforms like Railway/Render"
        }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
        return
    
    def do_POST(self):
        if self.path == '/api/predict' or self.path == '/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                import torch
                import torch.nn.functional as F
                from torch_geometric.data import Data
                import numpy as np
                
                # Import model
                from modules import TacticalNet
                
                # Parse request
                data = json.loads(body.decode())
                
                # Build simplified prediction (demo mode)
                def build_demo_graph(form_score, num_players=11, feature_dim=64):
                    rng = np.random.default_rng(seed=int(form_score * 100))
                    x = torch.tensor(
                        rng.normal(loc=form_score / 10.0, scale=0.1, size=(num_players, feature_dim)),
                        dtype=torch.float32,
                    )
                    src = [i for i in range(num_players) for j in range(num_players) if i != j]
                    dst = [j for i in range(num_players) for j in range(num_players) if i != j]
                    edge_index = torch.tensor([src, dst], dtype=torch.long)
                    batch = torch.zeros(num_players, dtype=torch.long)
                    return Data(x=x, edge_index=edge_index, batch=batch)
                
                def build_style_vector(ppda, field_tilt, tactic):
                    tactic_map = {
                        "Tiki-Taka": [0.9, 0.2, 0.8, 0.3],
                        "High Press": [0.7, 0.8, 0.6, 0.4],
                        "Low Block": [0.2, 0.1, 0.3, 0.5],
                        "Counter-Attack": [0.3, 0.6, 0.4, 0.7],
                        "Gegenpressing": [0.8, 0.9, 0.7, 0.3],
                        "Possession": [0.85, 0.3, 0.75, 0.2],
                    }
                    base = tactic_map.get(tactic, [0.5, 0.5, 0.5, 0.5])
                    pressing_norm = 1.0 - (ppda - 3.0) / 17.0
                    tilt_norm = field_tilt / 100.0
                    vec = [
                        base[0] * 0.6 + tilt_norm * 0.4,
                        base[1] * 0.6 + pressing_norm * 0.4,
                        base[2],
                        base[3],
                    ]
                    return torch.tensor([vec], dtype=torch.float32)
                
                # Extract parameters
                form_a = data.get('form_a', 7.5)
                form_b = data.get('form_b', 6.0)
                ppda_a = data.get('ppda_a', 8.0)
                ppda_b = data.get('ppda_b', 14.0)
                tilt_a = data.get('tilt_a', 60.0)
                tilt_b = data.get('tilt_b', 40.0)
                tactic_a = data.get('tactic_a', 'Tiki-Taka')
                tactic_b = data.get('tactic_b', 'Low Block')
                team_a = data.get('team_a', 'Team A')
                team_b = data.get('team_b', 'Team B')
                
                # Build graphs
                data_a = build_demo_graph(form_a)
                data_b = build_demo_graph(form_b)
                style_a = build_style_vector(ppda_a, tilt_a, tactic_a)
                style_b = build_style_vector(ppda_b, tilt_b, tactic_b)
                
                # Load model
                model = TacticalNet(player_feature_dim=64, hidden_dim=128, style_latent_dim=4)
                model.eval()
                
                # Predict
                with torch.no_grad():
                    logits = model(data_a, data_b, style_a, style_b)
                    probs = F.softmax(logits, dim=-1).squeeze().numpy()
                
                result = {
                    "team_a": team_a,
                    "team_b": team_b,
                    "probabilities": {
                        "team_a_win": float(probs[0]),
                        "draw": float(probs[1]),
                        "team_b_win": float(probs[2])
                    },
                    "configuration": {
                        "team_a": {
                            "form": form_a,
                            "tactic": tactic_a,
                            "ppda": ppda_a,
                            "field_tilt": tilt_a
                        },
                        "team_b": {
                            "form": form_b,
                            "tactic": tactic_b,
                            "ppda": ppda_b,
                            "field_tilt": tilt_b
                        }
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {
                    "error": str(e),
                    "message": "Prediction failed. Ensure all required parameters are provided."
                }
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
