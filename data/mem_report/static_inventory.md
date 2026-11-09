# Inventaire statique du cerveau (532 nœuds, 4 000 arêtes, vocab 10 000)

Tout [mesuré] : tailles lues sur les tenseurs après un pas d'entraînement
(les moments AdamW n'existent qu'après le premier pas).

| # | tenseur | type | shape | dtype | device | octets | Mo | grad | persistant | bloc |
|---:|---|---|---|---|---|---:|---:|:--:|:--:|---|
| 1 | `bank.b1_w` | parameter | 558x64x32 | torch.float32 | cuda:0 | 4571136 | 4.359 | oui | oui | state_dict.node_bank |
| 2 | `bank.b2_w` | parameter | 558x32x64 | torch.float32 | cuda:0 | 4571136 | 4.359 | oui | oui | state_dict.node_bank |
| 3 | `optim.?.exp_avg` | optimizer | 532x64x32 | torch.float32 | cuda:0 | 4358144 | 4.156 | non | oui | optimizer |
| 4 | `optim.?.exp_avg_sq` | optimizer | 532x64x32 | torch.float32 | cuda:0 | 4358144 | 4.156 | non | oui | optimizer |
| 5 | `optim.?.exp_avg` | optimizer | 532x32x64 | torch.float32 | cuda:0 | 4358144 | 4.156 | non | oui | optimizer |
| 6 | `optim.?.exp_avg_sq` | optimizer | 532x32x64 | torch.float32 | cuda:0 | 4358144 | 4.156 | non | oui | optimizer |
| 7 | `bank.q_w` | parameter | 558x32x32 | torch.float32 | cuda:0 | 2285568 | 2.180 | oui | oui | state_dict.node_bank |
| 8 | `bank.k_w` | parameter | 558x32x32 | torch.float32 | cuda:0 | 2285568 | 2.180 | oui | oui | state_dict.node_bank |
| 9 | `optim.?.exp_avg` | optimizer | 532x32x32 | torch.float32 | cuda:0 | 2179072 | 2.078 | non | oui | optimizer |
| 10 | `optim.?.exp_avg_sq` | optimizer | 532x32x32 | torch.float32 | cuda:0 | 2179072 | 2.078 | non | oui | optimizer |
| 11 | `optim.?.exp_avg` | optimizer | 532x32x32 | torch.float32 | cuda:0 | 2179072 | 2.078 | non | oui | optimizer |
| 12 | `optim.?.exp_avg_sq` | optimizer | 532x32x32 | torch.float32 | cuda:0 | 2179072 | 2.078 | non | oui | optimizer |
| 13 | `embedding.weight` | parameter | 10000x32 | torch.float32 | cuda:0 | 1280000 | 1.221 | oui | oui | state_dict.vocab |
| 14 | `output_head.weight` | parameter | 10000x32 | torch.float32 | cuda:0 | 1280000 | 1.221 | oui | oui | state_dict.vocab |
| 15 | `optim.embedding.weight.exp_avg` | optimizer | 10000x32 | torch.float32 | cuda:0 | 1280000 | 1.221 | non | oui | optimizer |
| 16 | `optim.embedding.weight.exp_avg_sq` | optimizer | 10000x32 | torch.float32 | cuda:0 | 1280000 | 1.221 | non | oui | optimizer |
| 17 | `optim.output_head.weight.exp_avg` | optimizer | 10000x32 | torch.float32 | cuda:0 | 1280000 | 1.221 | non | oui | optimizer |
| 18 | `optim.output_head.weight.exp_avg_sq` | optimizer | 10000x32 | torch.float32 | cuda:0 | 1280000 | 1.221 | non | oui | optimizer |
| 19 | `attr._last_out` | attribute tensor | 530x8x32 | torch.float32 | cuda:0 | 542720 | 0.518 | oui | non | other |
| 20 | `edge_index` | buffer | 2x20000 | torch.int64 | cuda:0 | 320000 | 0.305 | non | oui | state_dict.wiring |
| 21 | `bank.b1_b` | parameter | 558x64 | torch.float32 | cuda:0 | 142848 | 0.136 | oui | oui | state_dict.node_bank |
| 22 | `optim.?.exp_avg` | optimizer | 532x64 | torch.float32 | cuda:0 | 136192 | 0.130 | non | oui | optimizer |
| 23 | `optim.?.exp_avg_sq` | optimizer | 532x64 | torch.float32 | cuda:0 | 136192 | 0.130 | non | oui | optimizer |
| 24 | `_edge_keys[4010 pairs]` | host set | - | int | cpu | 128320 | 0.122 | non | non | not saved (derived) |
| 25 | `_edge_mask` | parameter | 20000 | torch.float32 | cuda:0 | 80000 | 0.076 | oui | oui | state_dict.wiring |
| 26 | `edge_usage` | buffer | 20000 | torch.float32 | cuda:0 | 80000 | 0.076 | non | oui | state_dict.wiring |
| 27 | `optim._edge_mask.exp_avg` | optimizer | 20000 | torch.float32 | cuda:0 | 80000 | 0.076 | non | oui | optimizer |
| 28 | `optim._edge_mask.exp_avg_sq` | optimizer | 20000 | torch.float32 | cuda:0 | 80000 | 0.076 | non | oui | optimizer |
| 29 | `bank.gate_w` | parameter | 558x1x32 | torch.float32 | cuda:0 | 71424 | 0.068 | oui | oui | state_dict.node_bank |
| 30 | `bank.b2_b` | parameter | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | oui | oui | state_dict.node_bank |
| 31 | `bank.ln_w` | parameter | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | oui | oui | state_dict.node_bank |
| 32 | `bank.ln_b` | parameter | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | oui | oui | state_dict.node_bank |
| 33 | `bank.q_b` | parameter | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | oui | oui | state_dict.node_bank |
| 34 | `bank.k_b` | parameter | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | oui | oui | state_dict.node_bank |
| 35 | `mean_out` | buffer | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | non | oui | state_dict.life |
| 36 | `attr.mean_out` | attribute tensor | 558x32 | torch.float32 | cuda:0 | 71424 | 0.068 | non | non | other |
| 37 | `optim.?.exp_avg` | optimizer | 532x1x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 38 | `optim.?.exp_avg_sq` | optimizer | 532x1x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 39 | `optim.?.exp_avg` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 40 | `optim.?.exp_avg_sq` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 41 | `optim.?.exp_avg` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 42 | `optim.?.exp_avg_sq` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 43 | `optim.?.exp_avg` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 44 | `optim.?.exp_avg_sq` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 45 | `optim.?.exp_avg` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 46 | `optim.?.exp_avg_sq` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 47 | `optim.?.exp_avg` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 48 | `optim.?.exp_avg_sq` | optimizer | 532x32 | torch.float32 | cuda:0 | 68096 | 0.065 | non | oui | optimizer |
| 49 | `output_head.bias` | parameter | 10000 | torch.float32 | cuda:0 | 40000 | 0.038 | oui | oui | state_dict.vocab |
| 50 | `optim.output_head.bias.exp_avg` | optimizer | 10000 | torch.float32 | cuda:0 | 40000 | 0.038 | non | oui | optimizer |
| 51 | `optim.output_head.bias.exp_avg_sq` | optimizer | 10000 | torch.float32 | cuda:0 | 40000 | 0.038 | non | oui | optimizer |
| 52 | `node_born_step` | buffer | 558 | torch.int64 | cuda:0 | 4464 | 0.004 | non | oui | state_dict.life |
| 53 | `node_frozen_until` | buffer | 558 | torch.int64 | cuda:0 | 4464 | 0.004 | non | oui | state_dict.life |
| 54 | `bank.gate_b` | parameter | 558x1 | torch.float32 | cuda:0 | 2232 | 0.002 | oui | oui | state_dict.node_bank |
| 55 | `optim.?.exp_avg` | optimizer | 532x1 | torch.float32 | cuda:0 | 2128 | 0.002 | non | oui | optimizer |
| 56 | `optim.?.exp_avg_sq` | optimizer | 532x1 | torch.float32 | cuda:0 | 2128 | 0.002 | non | oui | optimizer |
| 57 | `attr.last_out_acc` | attribute tensor | 8x32 | torch.float32 | cuda:0 | 1024 | 0.001 | non | non | other |
| 58 | `node_alive` | buffer | 558 | torch.bool | cuda:0 | 558 | 0.001 | non | oui | state_dict.life |
| 59 | `is_io` | buffer | 558 | torch.bool | cuda:0 | 558 | 0.001 | non | oui | state_dict.life |
| 60 | `readout_norm.weight` | parameter | 32 | torch.float32 | cuda:0 | 128 | 0.000 | oui | oui | state_dict.vocab |
| 61 | `readout_norm.bias` | parameter | 32 | torch.float32 | cuda:0 | 128 | 0.000 | oui | oui | state_dict.vocab |
| 62 | `optim.readout_norm.weight.exp_avg` | optimizer | 32 | torch.float32 | cuda:0 | 128 | 0.000 | non | oui | optimizer |
| 63 | `optim.readout_norm.weight.exp_avg_sq` | optimizer | 32 | torch.float32 | cuda:0 | 128 | 0.000 | non | oui | optimizer |
| 64 | `optim.readout_norm.bias.exp_avg` | optimizer | 32 | torch.float32 | cuda:0 | 128 | 0.000 | non | oui | optimizer |
| 65 | `optim.readout_norm.bias.exp_avg_sq` | optimizer | 32 | torch.float32 | cuda:0 | 128 | 0.000 | non | oui | optimizer |
| 66 | `optim._edge_mask.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 67 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 68 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 69 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 70 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 71 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 72 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 73 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 74 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 75 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 76 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 77 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 78 | `optim.?.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 79 | `optim.embedding.weight.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 80 | `optim.output_head.weight.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 81 | `optim.output_head.bias.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 82 | `optim.readout_norm.weight.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 83 | `optim.readout_norm.bias.step` | optimizer | - | torch.float32 | cpu | 4 | 0.000 | non | oui | optimizer |
| 84 | `cemetery[0 nodes]` | host tensors | - | float32 | cpu | 0 | 0.000 | non | oui | cemetery |

**Somme des tenseurs** : 48.44 Mo (50795484 octets), dont 47.73 Mo persistants.

**`torch.cuda.memory_allocated()`** : 391.75 Mo — **checkpoint classe L** : 21.78 Mo (compressé par torch.save).

