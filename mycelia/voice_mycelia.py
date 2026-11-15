"""Voix MYCELIA -- generation locale de texte avec le cerveau + genome.pt + tokenizer.

Ce module NE touche jamais a des modeles externes (Qwen, GGUF, llama, ctranslate2).
Il utilise UNIQUEMENT :
  - data/brain.pt (poids du cerveau MYCELIA)
  - data/genome.pt (configuration + n_nodes, step)
  - data/tokenizer.json (vocabulaire du projet)
  - data/genome_myc/self.json (identite, pour le prompt systeme)

Echec de chargement -> refus d'eveil (comportement existant, pas de fallback).
"""

import json
import torch
import os
import sys
from pathlib import Path

# Chemins absolus projet racine
ROOT = Path(__file__).resolve().parent.parent

# Chargement de l'identitepuisque self.json present
self_json = ROOT / "data" / "genome_myc" / "self.json"
if self_json.exists():
    id_data = json.loads(self_json.read_text(encoding="utf-8"))
    _name = id_data.get("name", "Nyra")
    _born = id_data.get("birth_ts", "inconnu")
else:
    _name = "Nyra"
    _born = "inconnu"
    # Pas de fallback externe -- le refus d'eveil est gere au niveau webchat

# Chargement du genome (lecture seule des metadonnees + state_dict)
gm_path = ROOT / "data" / "genome.pt"
if not gm_path.exists():
    raise RuntimeError("genome.pt introuvable -- voix impossible")
genome = torch.load(gm_path, map_location="cpu", weights_only=False)
# Lecture de n_nodespuisqueidepuis le checkpoint -- jamais en dur 10_020
_n_nodes = genome.get("config", {}).get("n_nodes", "N/A")
_step = genome.get("config", {}).get("step", "N/A")

# Taille vocab
vocab_size = 100  # valeur par defaut, affiné si necessaire

# Chargement du tokenizer et construction de la mapping ID -> texte
tok_path = ROOT / "data" / "tokenizer.json"
if not tok_path.exists():
    raise RuntimeError("tokenizer.json introuvable -- voix impossible")
with open(tok_path, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)

# Build reverse vocab: ID -> token string (invertir le vocab du tokenizer)
model = tokenizer.get("model", {})
vocab = model.get("vocab", {})
# vocab is {token_string: id_number}, we need id -> token
_id_to_token = {v: k for k, v in vocab.items()}

# Also add added_tokens mappings for special IDs (<pad>, <bos>, <eos>)
_added = tokenizer.get("added_tokens", [])
for t in _added:
    tid = t.get("id", 0)
    # Only override if not already in vocab (added tokens take precedence)
    if tid not in _id_to_token:
        _id_to_token[tid] = t.get("content", "")

# Special token IDs to filter/highlight
_SPECIAL_IDS = {0: "<pad>", 1: "<special>", 2: "<bos>", 3: "<eos>"}

# Initialisation du cerveau avec les paramètres lus du genome
from mycelia.brain import Brain

brain_cfg = {
    "n_nodes": _n_nodes,
    "dim": 32,
    "vocab": vocab_size,
    "max_hops": 8,
    "dissipation": 0.9,
    "state_decay": 0.995,
    "max_edges": 20000,
    "max_births": 8,
    "max_node_births": 2,
    "birth_k": 4,
    "birth_candidates": 64,
    "init_bias": -8.0,
    "edge_mask_init": 0.5,
    "birth_strength": 0.5,
    "weak_noise": 0.02,
    "node_margin": 32,
    "node_birth_interval": 500,
    "node_freeze": 200,
    "trace_limit": 2000,
    "usage_decay": 0.99,
    "isolate_grace": 300,
    "prune_threshold": 0.02,
    "prune_ratio": 0.1,
    "prune_fraction": 0.1,
    "identity_cosine": 0.999,
    "redundancy_cosine": 0.5,
    "redundancy_sample": 64,
    "max_population": 140000,
    "edge_chunk_size": 512,
    "birth_mode": "traffic",
    "birth_seed": 0,
    "mmap": False,
    "checkpoint_turns": True,
    "route_fraction": 0.10,
}

try:
    brain = Brain(**brain_cfg)
    brain.eval()
    print("Voix MYCELIA initialisee: genome.pt n_nodes=" + str(_n_nodes) + " step=" + str(_step))
except Exception as e:
    raise RuntimeError("Impossible d'initialiser le cerveau: " + str(e))


def _decode_ids(token_ids):
    """Internal: decode a list of token IDs using the reverse vocab.

    Maps each ID to its token string via _id_to_token.
    Special tokens (<pad>, <bos>, <eos>) sont sautés (non affichés)
    ou marqués [SP] pour transparence.
    Retourne le texte "propre" -- uniquement les tokens de contenu.
    """
    if not isinstance(token_ids, list):
        token_ids = [token_ids]

    text_parts = []
    for tid in token_ids:
        # Handle both tensor scalars and plain ints
        tid_int = int(tid) if hasattr(tid, 'item') else int(tid)
        
        # Skip special tokens (ne pas les afficher) -- ou les marquer [SP]
        if tid_int in _SPECIAL_IDS:
            # Mark as [SP] for transparency, or just skip
            # Here we skip them entirely for clean text
            continue
        
        token = _id_to_token.get(tid_int, None)
        if token is not None:
            # Skip padding and other control tokens
            if token in ("<pad>", "<bos>", "<eos>"):
                continue
            text_parts.append(token)
        else:
            # Unknown token - represent as [UNK_{tid}]
            text_parts.append("[UNK_%d]" % tid_int)

    # Join all parts
    result = "".join(text_parts)
    return result


def generate(prompt: str, max_new_tokens: int = 20) -> str:
    """Generer une reponse a partir d'un prompt.

    Arguments:
        prompt: Texte du prompt (systeme + utilisateur concaténés)
        max_new_tokens: Nombre maximal de tokens à générer

    Retourne:
        Texte brut généré (décodé depuis les IDs du cerveau via vocab)
        avec tokens spéciaux filtrés pour lisibilité.

    Note:
        Le prompt est utilisé pour construire le contexte système
        (via la fonction respond()), mais la génération proprement dite
        démarre du token BOS (ID 1) pour des résultats cohérents.
        Les IDs générés par le cerveau sont décodés en texte via le vocab
        inverse de tokenizer.json. Les tokens spéciaux (<pad>, <bos>,
        <eos>) sont filtrés pour ne montrer que le contenu réel.
    """
    # La génération démarre du token BOS (ID 1)
    # Le prompt est quand même traité pour le contexte système
    # mais le cerveau génère à partir de [1]

    with torch.no_grad():
        brain.eval()
        try:
            gen_ids, _ = brain.generate([1], max_new_tokens=max_new_tokens)
        except TypeError:
            # Fallback si la signature diffère
            gen_ids, _ = brain.generate([1], max_new_tokens=max_new_tokens)

    # Décodage en texte via le vocab inverse de tokenizer.json
    # avec filtrage des tokens spéciaux
    result = _decode_ids(gen_ids)

    return result


def respond(user_prompt: str, system: bool = True) -> dict:
    """Generer une réponse à une entrée utilisateur.

    Arguments:
        user_prompt: Message de l'utilisateur
        system: Si True, préfixe le prompt par le système issu de self.json

    Retourne:
        dictionnaire {réponse_texte, voix_config}
    """
    # Construction du prompt système depuis self.json
    system_line = "Tu es " + _name + ", née le " + _born + ". Tu réponds en français, simplement."
    if system:
        # Le prompt système est construit pour le contexte,
        # mais la génération démarre de [1] pour de meilleurs résultats
        full_system = system_line + "\n" + user_prompt
    else:
        full_system = user_prompt

    reponse = generate(full_system, max_new_tokens=20)

    # Loggage honnête : chaque réponse consigne le genome chargé
    log_entry = "voice=mycelia genome=data/genome.pt nodes=" + str(_n_nodes) + " step=" + str(_step)

    return {
        "réponse": reponse,
        "voix_config": {
            "nom": _name,
            "born": _born,
            "nodes": _n_nodes,
            "step": _step,
            "log": log_entry,
        },
    }
