# Fermeture des compteurs de nœuds — la question est tranchée

**6 novembre 2026.** Le run v5 affichait `nœuds 520`, `morts-nœuds 64`,
`nœuds_nés 8`, alors que `512 + 8 − 64 = 456`. Le §5 de
`docs/PLATEAU-croissance.md` laissait la question ouverte avec deux hypothèses.
Elle est résolue, et la réponse n'est ni l'une ni l'autre : **il manquait un
terme au relevé, pas à l'arithmétique.**

## 1. La réponse, lue dans le log

`logs_toy_v5_2000_batch16.txt`, relevés successifs (extraits) :

| pas | nœuds vivants | morts-nœuds | cimetière |
|---|---:|---:|---:|
| 100 | 514 | 0 | 0 |
| 300 | 514 | 46 | 16 |
| 500 | 518 | 57 | 1 |
| 1 100 | 518 | 63 | 0 |
| 2 000 | **520** | **64** | **0** |

Deux faits sufisent :

1. `nœuds` est bien le compte **vivant** (`train.py` L475 :
   `"nodes": int(self.brain.node_alive.sum())`), pas la taille du pool : à la
   fin, les 520 slots sont tous vivants.
2. Le cimetière est **vide** à la fin alors que 64 enterrements ont eu lieu :
   chaque nœud enterré est donc revenu. `revive_node`
   (`mycelia/lifecycle.py`) remet `node_alive` à vrai **sans décrémenter
   `node_deaths`**, et rien n'imprimait le nombre de retours.

## 2. L'identité qui se ferme

```
vivants = slots − (morts − ressuscités)
```

soit, au pas 2 000 : `512 + 8 = 520` slots, `520 − (64 − 64) = 520` vivants.
La soustraction `512 + 8 − 64 = 456` était fausse parce qu'elle oubliait les
64 résurrections. C'est cohérent avec `cimetière 0` et avec le fait que
`kill_isolated_nodes` n'enterre un nœud qu'une fois par vie : un nœud ressuscité
puis re-condamné compte deux morts et un retour de plus.

## 3. Ce qui a été corrigé, et ce qui ne l'est pas

- **Ajouté** : `Brain.node_revivals`, incrémenté dans `revive_node`, exposé dans
  les métriques (`train.py`) et imprimé dans la ligne de vitals
  (`ressuscités 64`). L'instrument se ferme maintenant tout seul.
- **Non modifié** : `node_deaths` reste un compteur **d'événements**, pas de
  nœuds distincts. C'est le bon choix — l'extinction progressive s'intéresse au
  nombre de morts survenues — mais cela signifie qu'il ne faut jamais lire
  `morts-nœuds` comme « nombre de nœuds aujourd'hui morts ». C'est écrit ici
  pour que personne ne le relise de travers.
- **Test** : `tests/test_instrumentation.py::test_node_counters_close` vérifie
  l'identité `alive == slots − morts + ressuscités` sur un cerveau réel, après
  un enterrement et deux résurrections.

## 4. Ce que cela change pour le plateau

Rien : `docs/PLATEAU-croissance.md` §1-§4 ne dépendent pas de ce compteur. Seul
son §5 était ouvert, et il est corrigé par ce document. Le chiffre à retenir
pour la suite est que **520 nœuds vivants** sur 861 slots reste valable ; c'est
le « 456 » qui n'a jamais existé.
