#!/usr/bin/env python3
"""
Calcul de la matrice de divergence KL pour les 4 donateurs S8.
===============================================================

Ce script calcule les KL croisés entre les 4 donateurs en utilisant
la fonction kl_direction de s5_common.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from lab_common import ToyCorpus, LAB, set_seed
from s5_common import kl_direction, topk_overlap, columns_mycelia, load_dense

SEQ = 8
CORPUS_PATH = LAB / "corpus_toy.txt"
DONORS_DIR = LAB / "s8" / "donors"
DIV_FILE = LAB / "s8" / "divergence.json"

# Lecture des donneurs
def load_all_donors():
    """Charge les 4 donateurs depuis les fichiers .pt"""
    donateurs = {}
    for tag in ['A', 'B', 'C', 'D']:
        path = DONORS_DIR / f"{tag}.pt"
        if not path.exists():
            print(f"[!] {tag} non trouve")
            continue
        blob = torch.load(str(path), weights_only=False)
        
        # Vérifier le type avant de charger
        state_dict = blob.get('state_dict', {})
        config = blob.get('config', {})
        
        # Determine type: dense vs NUMENIA brain
        has_embed = 'embed.weight' in state_dict
        has_edge_index = 'edge_index' in blob or '_edge_mask' in blob
        has_clock_or_ne = 'clock' in blob or 'ne' in blob
        
        is_dense = has_embed
        is_brain = (not has_embed) and (has_edge_index or has_clock_or_ne)
        
        if is_dense:
            # Donneur DENSE
            from micro_dense import DenseTransformer
            config = blob.get('config', {})
            vocab_size = blob.get('vocab_size', 312)
            
            # Vérifier les clés de state_dict
            expected_keys = {'mask', 'embed.weight', 'pos.weight', 'head.weight', 'head.bias', 'ln_f.weight', 'ln_f.bias'}
            for i in range(config.get('layers', 2)):
                for suffix in ['.ln1.weight', '.ln1.bias', '.qkv.weight', '.qkv.bias', 
                              '.proj.weight', '.proj.bias', '.ln2.weight', '.ln2.bias',
                              '.fc1.weight', '.fc1.bias', '.fc2.weight', '.fc2.bias']:
                    expected_keys.add(f'blocks.{i}{suffix}')
            
            actual_keys = set(state_dict.keys())
            if expected_keys - actual_keys:
                print(f"[!] {tag}: cles manquantes dans state_dict: {expected_keys - actual_keys}")
                print(f"   cles attendues: {sorted(expected_keys)[:10]}...")
                print(f"   cles actuelles: {sorted(actual_keys)[:10]}...")
                # Continuer quand meme
            
            model = DenseTransformer(
                vocab=vocab_size,
                dim=config.get('dim', 16),
                layers=config.get('layers', 2),
                heads=config.get('heads', 2),
                mlp=config.get('mlp', 768),
                seq=config.get('seq', 8),
            )
            
            # Filtrer le state_dict pour les cles attendues
            filtered_sd = {k: v for k, v in state_dict.items() if k in expected_keys or k.startswith('blocks.')}
            if len(filtered_sd) < len(expected_keys) * 0.8:
                print(f"[!] {tag}: trop de cles manquantes, abandon")
                continue
            
            model.load_state_dict(filtered_sd, strict=False)
            model.eval()
            donateurs[tag] = model
            
        elif is_brain:
            # Donneur NUMENIA (brain) - charge via repaired_mycelia
            from repaired_mycelia import load_brain
            brain = load_brain(path)
            donateurs[tag] = brain
        else:
            print(f"[!] {tag} format inconnu (cles: {list(blob.keys())[:10]})")
    
    return donateurs

def forward_model(model, x, generator=None):
    """Forward un modèle (dense ou NUMENIA) et retourne les logits [batch, seq-1, vocab]
    
    Retourne None si le modèle ne peut pas faire de prédiction.
    """
    with torch.no_grad():
        if hasattr(model, 'forward') and not hasattr(model, '_edge_mask'):
            # Dense: forward avec mask
            out = model(x)
            # Retourne les logits (évite les tuples)
            if isinstance(out, tuple):
                logits = out[0] if len(out) > 0 else out
            elif isinstance(out, dict):
                logits = out.get('logits', out)
            else:
                logits = out
            logits = logits[:, 1:]  # skip position 0 (pas de contexte)
            return logits
        
        elif hasattr(model, 'parts') and not hasattr(model, '_edge_mask'):
            # Dense comme dans s5_distill
            _, logits = model.parts(x)
            logits = logits[:, 1:]
            return logits
        
        elif hasattr(model, '_edge_mask'):
            # NUMENIA: forward avec état
            state = model._fresh_state(x.size(0)) if hasattr(model, '_fresh_state') else None
            if state is not None:
                out = model(x, state=state, grow=False)
            else:
                out = model(x, grow=False)
            # Retourne les logits
            if isinstance(out, tuple):
                logits = out[0] if len(out) > 0 else out
            elif isinstance(out, dict):
                logits = out.get('logits', out)
            else:
                logits = out
            logits = logits[:, 1:]
            return logits
        
        else:
            # Fallback
            try:
                out = model(x)
                if isinstance(out, tuple):
                    logits = out[0] if len(out) > 0 else out
                else:
                    logits = out
                logits = logits[:, 1:]
                return logits
            except Exception as e:
                print(f"[!] Erreur forward model: {e}")
                return None


def compute_kl_matrix(donateurs, corpus, num_samples=50, batch_size=64):
    """Calculer la matrice KL (KL(A||B)) entre donateurs"""
    matrix = {}
    top3_matrix = {}
    refs = {}
    all_stats = {}
    
    # Calculer les logits pour chaque donneur sur un ensemble de contextes
    generator = torch.Generator().manual_seed(42)
    
    for tag, model in donateurs.items():
        all_logits = []
        kept_tokens: list[int] = []
        
        # Utiliser des batches du corpus
        for _ in range(num_samples):
            x, _ = corpus.batch(batch_size, generator=generator)
            
            logits = forward_model(model, x)
            if logits is not None:
                all_logits.append(logits)
        
        # Concaténer tous les logits
        if all_logits:
            big_logits = torch.cat(all_logits, dim=0)
            refs[tag] = big_logits
            
            # Calcul des statistiques
            flat = big_logits.reshape(-1, big_logits.shape[-1])
            ps = F.softmax(flat, dim=-1)
            logps = F.log_softmax(flat, dim=-1)
            
            # Entropie moyenne
            entropy = -(ps * logps).sum(dim=-1).mean()
            max_prob = ps.max(dim=-1).values.mean()
            confidence = (ps.max(dim=-1).values > 0.1).float().mean()
            
            all_stats[tag] = {
                "avg_entropy": round(float(entropy), 4),
                "avg_max_prob": round(float(max_prob), 4),
                "top1_confidence": round(float(confidence), 4),
                "vocab_size": flat.shape[-1],
                "total_contextes": big_logits.shape[0] * big_logits.shape[1],
            }
        else:
            refs[tag] = None
            all_stats[tag] = {"error": "pas de logits generes"}
    
    # Matrice KL (KL(A || B) = attendu en A, attendu sur B)
    tags = list(donateurs.keys())
    for i, tag_a in enumerate(tags):
        for j, tag_b in enumerate(tags):
            if i >= j:
                continue
            
            logits_a = refs.get(tag_a)
            logits_b = refs.get(tag_b)
            
            if logits_a is None or logits_b is None:
                key = f"{tag_a}_vers_{tag_b}"
                key2 = f"{tag_b}_vers_{tag_a}"
                matrix[key] = None
                matrix[key2] = None
                top3_matrix[key] = None
                top3_matrix[key2] = None
                continue
            
            # KL(A || B)
            try:
                kl_ab = kl_direction(logits_a, logits_b)
                kl_ba = kl_direction(logits_b, logits_a)
                key = f"{tag_a}_vers_{tag_b}"
                key2 = f"{tag_b}_vers_{tag_a}"
                matrix[key] = round(float(kl_ab), 4)
                matrix[key2] = round(float(kl_ba), 4)
                
                # Top-3 overlap
                flat_a = logits_a.reshape(-1, logits_a.shape[-1])
                flat_b = logits_b.reshape(-1, logits_b.shape[-1])
                top3 = topk_overlap(flat_a, flat_b)
                top3_matrix[key] = {
                    "identiques": round(float(top3["identiques"]), 4),
                    "recouvrement": round(float(top3["recouvrement"]), 4),
                }
                top3_matrix[key2] = {
                    "identiques": round(float(top3["identiques"]), 4),
                    "recouvrement": round(float(top3["recouvrement"]), 4),
                }
            except Exception as e:
                print(f"[!] KL {tag_a} vs {tag_b}: {e}")
                key = f"{tag_a}_vers_{tag_b}"
                key2 = f"{tag_b}_vers_{tag_a}"
                matrix[key] = None
                matrix[key2] = None
    
    return matrix, top3_matrix, all_stats

def compute_plafond_kl(donateurs, corpus):
    """Calculer un plafond de KL: KL inter-famille (dense vs NUMENIA pool24)
    
    Le plafond represente la difference maximale attendue entre deux architectures
    differentes. Pour S8, on compare:
    - A, B, C (dense) vs D (NUMENIA pool24): KL > plafond
    - KL entre donateurs de meme famille < plafond
    """
    dense_tags = [t for t in donateurs if t != 'D']
    mycelia_tags = ['D'] if 'D' in donateurs else []
    
    plafond = {}
    
    for dt in dense_tags:
        for mt in mycelia_tags:
            key = f"{dt}_vers_{mt}"
            # On ne peut pas facilement calculer KL inter-architecture,
            # mais on peut utiliser la difference de perte comme reference
            # pour estimer
            plafond[key] = 0.858  # Reference S5
    
    return plafond

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    
    print("="*60)
    print("Calcul matrice de divergence KL — S8")
    print("="*60)
    
    # Chargement des donateurs
    print("\n[1] Chargement des donateurs...")
    donateurs = load_all_donors()
    print(f"  {len(donateurs)} donateurs charges: {list(donateurs.keys())}")
    
    # Corpus
    print("\n[2] Preparation du corpus...")
    corpus = ToyCorpus(seq_len=SEQ, path=CORPUS_PATH)
    print(f"  Vocab: {corpus.vocab_size}, phrases: {len(corpus.lines)}")
    
    # KL matrix
    print("\n[3] Calcul de la matrice KL...")
    matrix, top3_matrix, stats = compute_kl_matrix(donateurs, corpus, num_samples=args.samples)
    
    print("\n  MATRICE KL (KL(A||B)):")
    tags = list(donateurs.keys())
    print("      ", "  ".join(f"{t:>10}" for t in tags))
    for i, tag_a in enumerate(tags):
        row = [f"{tag_a:>10}"]
        for j, tag_b in enumerate(tags):
            if i >= j:
                row.append(" " * 10)
            else:
                key = f"{tag_a}_vers_{tag_b}"
                val = matrix.get(key, "N/A")
                row.append(f"{str(val):>10}")
        print("  ".join(row))
    
    print("\n  TOP-3 IDENTIQUES:")
    for tag_a in tags:
        for tag_b in tags:
            if tag_a >= tag_b:
                continue
            key = f"{tag_a}_vers_{tag_b}"
            val = top3_matrix.get(key)
            if val:
                print(f"    {tag_a} vs {tag_b}: identiques={val['identiques']}, recouvrement={val['recouvrement']}")
    
    print("\n  STATISTIQUES DES DONNEURS:")
    for tag, s in stats.items():
        print(f"    {tag}: {s}")
    
    # Plafond
    print("\n[4] Plafond KL (reference S5):")
    plafond = compute_plafond_kl(donateurs, corpus)
    for key, val in plafond.items():
        print(f"    {key}: {val}")
    
    # Sauvegarde
    print("\n[5] Sauvegarde...")
    result = {
        "donneurs": list(donateurs.keys()),
        "kl_matrix": matrix,
        "top3_matrix": top3_matrix,
        "stats": stats,
        "plafond": plafond,
        "samples": args.samples,
    }
    
    DIV_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Fichier: {DIV_FILE}")
    
    print("\n" + "="*60)
    print("Calcul termine")
    print("="*60)
    
    # Verifier la divergence
    print("\n[6] Verifications:")
    for tag_a in tags:
        for tag_b in tags:
            if tag_a >= tag_b:
                continue
            key = f"{tag_a}_vers_{tag_b}"
            kl = matrix.get(key)
            if kl is not None:
                if kl < 0.1:
                    print(f"  [!!] KL({tag_a}||{tag_b}) = {kl} < 0.1 : jumeaux detectes!")
                elif kl > 0.5:
                    print(f"  [OK] KL({tag_a}||{tag_b}) = {kl} : bien diverge")
                else:
                    print(f"  [...] KL({tag_a}||{tag_b}) = {kl} : modere")
    
    # Identite cards
    print("\n[7] Cartes d'identite...")
    for tag, model in donateurs.items():
        card = {
            "tag": tag,
            "label": {
                "A": "dense (lignee S5)",
                "B": "dense variant (graine+arch)",
                "C": "dense ordre melange (contextualite)",
                "D": "NUMENIA pool24 donneur",
            }.get(tag, f"donneur-{tag}"),
            "type": "dense" if hasattr(model, 'vocab') else "brain",
            "config": {},
            "stats": stats.get(tag, {}),
        }
        
        # Set card type based on actual model attributes
        is_dense = hasattr(model, 'vocab')
        is_brain = hasattr(model, 'ne') or hasattr(model, 'n_nodes')
        
        if is_dense:
            card["type"] = "dense"
            # Dense
            if hasattr(model, 'blocks'):
                n_layers = len(model.blocks)
                first_block = model.blocks[0]
                card["config"] = {
                    "vocab": model.vocab,
                    "dim": model.dim,
                    "layers": n_layers,
                    "heads": first_block.heads,
                    "mlp": first_block.fc1.out_features,
                    "seq": model.seq,
                }
            else:
                # Brain detected as dense - use fallback
                card["type"] = "brain"
                card["config"] = {
                    "n_nodes": getattr(model, 'n_nodes', None),
                    "ne": int(model.ne) if hasattr(model, 'ne') else None,
                    "max_edges": getattr(model, 'max_edges', None),
                    "two_speeds": getattr(model, 'two_speeds', None),
                }
        elif is_brain:
            card["type"] = "brain"
            card["config"] = {
                "n_nodes": getattr(model, 'n_nodes', None),
                "ne": int(model.ne) if hasattr(model, 'ne') else None,
                "max_edges": getattr(model, 'max_edges', None),
                "two_speeds": getattr(model, 'two_speeds', None),
            }
        else:
            card["type"] = "unknown"
            print(f"[!] {tag}: type inconnu (cles model: {[k for k in dir(model) if not k.startswith('_')][:10]})")
            card["config"] = {
                "n_nodes": getattr(model, 'n_nodes', None),
                "ne": int(model.ne) if hasattr(model, 'ne') else None,
                "max_edges": getattr(model, 'max_edges', None),
                "two_speeds": getattr(model, 'two_speeds', None),
            }
        
        card_path = DONORS_DIR / f"identity_{tag}.json"
        card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Fichier: {card_path}")
    
    print("\n" + "="*60)
    print("FIN")
    print("="*60)

if __name__ == "__main__":
    main()
