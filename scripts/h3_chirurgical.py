import sys, gc, torch
sys.path.insert(0, "/home/dieux/mycelia")
from mycelia.config import load_config
from mycelia.train import Trainer
from mycelia.brain import Brain

# Brain avec n_nodes=1000 (paramètres par défaut pour le reste)
brain = Brain(n_nodes=1000)

# Hook post-forward pour capturer la sortie
captured = {}

def forward_post_hook(module, input, output):
    captured['out'] = output.detach()

handle = brain.register_forward_post_hook(forward_post_hook)

trainer = Trainer(config=load_config(), data="toy", device="cpu", scale=4, batch_size=2, seq_len=16)
trainer.brain = brain
trainer.step = 10007

print("=== H3 CHIRURGICAL: gc.get_referrers chain ===")
for step in range(1, 11):
    trainer.step_once()
    if step in (1, 10):
        t = captured.get('out')
        if t is not None:
            print(f"\nStep {step}: shape={t.shape}")
            referrers = gc.get_referrers(t)
            print(f"  Nb références: {len(referrers)}")
            # Chaîne de référence simplifiée : suivre réf[0] profondeur 5
            current = referrers[0] if referrers else None
            depth = 0
            chain_parts = []
            visited = {id(t)}
            while current and depth < 5:
                depth += 1
                cid = id(current)
                if cid in visited:
                    chain_parts.append(f"niveau{depth}: CYCLE (id={cid})")
                    break
                visited.add(cid)
                try:
                    tname = type(current).__name__
                    if hasattr(current, '__class__'):
                        tname = f"{current.__class__.__module__}.{current.__class__.__name__}"
                    chain_parts.append(f"niveau{depth}: {tname} (id={cid})")
                except:
                    chain_parts.append(f"niveau{depth}: {type(current).__name__} (id={cid})")
                try:
                    refs = gc.get_referrers(current)
                    current = refs[0] if refs else None
                except:
                    current = None
            print("  Chaîne complète :")
            for p in chain_parts:
                print(f"  {p}")
        else:
            print(f"Step {step}: Aucun tenseur capturé")
    captured.clear()

print("\n=== FIN de la capture ===")
handle.remove()