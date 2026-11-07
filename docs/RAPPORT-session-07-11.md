# Session du 7 novembre 2026 — parallélisation refusée, forme corrigée, comparaison entamée

**7 novembre 2026.** Consigne : trancher la parallélisation du front d'onde par lecture
de code (bloquant), corriger cinq points de forme, produire la première preuve
comparative colonie contre Transformer, observer le régime texte puis lancer un run
conditionnel, livrer un verdict. Étiquette de chaque chiffre : **[mesuré]**,
**[estimé]**, **[supposé]**.

## 1. Verdict en une phrase

La parallélisation du front d'onde est **refusée par lecture de code** — la membrane du
destinataire est la clé de routage, pas une mémoire passive —, les cinq corrections de
forme sont livrées et vérifiées, la première comparaison a **démasqué un bug réel du
baseline** (il apprenait l'identité), et ni l'observation ni le run texte n'ont été
lancés parce que le GPU était occupé à 97 % par un jeu et que la machine a épuisé son
fichier de pagination.

## 2. Étape par étape

### Étape 1 — la membrane influe-t-elle sur le routage ? **FAITE, verdict NON**

Trois questions, trois réponses lues dans `mycelia/brain.py` (détail complet :
`docs/PARALLELISATION-front-onde.md`).

| Question | Réponse | Preuve |
|---|---|---|
| (a) l'émission lit-elle la membrane ? | **non**, seulement ce qui vient d'arriver | L591-592 `x = wave` ; `wave` = injection (L583) ou `inbox` du tour précédent (L663) |
| (b) la porte / le routage dépend-il de la membrane ? | **OUI, du destinataire** | L639 `key_c = self.bank.keys(state[recv_c], recv_c)` ; L647-649 le score multiplie le message |
| (c) couplage entre positions hors boucle ? | pas de masque, causalité structurelle, mais la membrane est **portée** | L573-575 prédiction avant injection ; L660-663 `state` mis à jour puis réutilisé à `t+1` |

La première prémisse de l'avis externe est donc vraie, **la deuxième est fausse**, la
troisième vraie mais insuffisante. Le « scan » proposé ne s'applique pas : la même
membrane alimente une sigmoïde qui multiplie le message avant accumulation, donc la
récurrence n'est pas associative (`state_h = state_{h-1}·0,995 + Σ out_e·σ(q_e·K(state_{h-1}))·m_e·0,9`).

Conséquence appliquée sans exception : **aucun drapeau `wave_parallel`, aucun
`test_wave_parallel_equivalence`, aucun passage au banc, zéro ligne de code touchée.**
Paralléliser aurait demandé de geler la clé sur l'émission — c'est-à-dire de retirer à
F2 le routage dépendant de l'état. Refusé.

### Étape 2 — corrections de forme : **FAITE** (4 mesurées, 1 tranchée par le log)

**2.1 Init de sortie — mesuré.** `OUTPUT_INIT_STD = 0.02`, `output_head` initialisé par
un `torch.Generator` dédié.

| | perte initiale | ppl | critère `< 9,3024` |
|---|---:|---:|---|
| avant (défaut PyTorch) | **9,2838** | 10 763 | non |
| après (σ = 0,02) | **9,2289** | 10 187 | **oui** |
| uniforme théorique `ln 10 000` | 9,2103 | 10 000 | — |

En conditions réelles, premier relevé du run comparatif : **9,1856 au pas 50**
[mesuré]. Tests : `test_initial_loss_near_uniform`.
Ce que la correction a failli casser : `normal_()` consommait le RNG global et décalait
le tirage de candidats, ce qui faisait échouer `test_revive_from_cemetery_via_growth` —
détecté par la suite, corrigé par un générateur dédié, consigné comme piège n°8 de
`CONTRIBUTING.md`. `test_resume_continuity` n'a pas été touché et reste vert à 1e-6 : il
compare deux reprises du même état, pas deux initialisations.

**2.2 `pytest.ini` — mesuré.** `testpaths = tests`. `pytest -q` nu passe de « collecte
cassée par `archive/v2-1633` (`ImportError: Config`) » à **43 tests collectés en 1,00 s**.

**2.3 Compteurs de nœuds — tranché, et ce n'était aucune des deux hypothèses.**
`520` est bien le nombre de nœuds **vivants** (`train.py` L475). Le cimetière est vide
alors que 64 enterrements ont eu lieu : il y a eu **64 résurrections**, que rien
n'imprimait. L'identité qui se ferme est

```
vivants = slots − (morts − ressuscités)      520 − (64 − 64) = 520
```

`512 + 8 − 64 = 456` oubliait les résurrections ; **ce nombre n'a jamais existé**.
Ajouté : `Brain.node_revivals`, incrémenté dans `revive_node`, métrique et ligne de
vitals `ressuscités`, test `test_node_counters_close`, doc `docs/COMPTEURS-noeuds.md`,
§5 de `docs/PLATEAU-croissance.md` corrigé. `node_deaths` reste un compteur
d'**événements** — écrit noir sur blanc pour qu'il ne soit plus relu de travers.

**2.4 README — fait.** Pièges et consignes d'agent (08/10, « Section 0 bis », « the
designer ») déplacés dans `CONTRIBUTING.md` (10 pièges) ; EN + FR déclarés faisant foi,
les six autres traductions marquées **gelées** ; scale 3 marqué **feuille de route** ;
pourquoi scale 1 (302 nœuds, didactique) < scale 0 (512) expliqué ; compte de tests
corrigé (32 → 43) ; commande `pytest tests` → `pytest -q`.

**2.5 Git — fait.** Tag annoté `v5` (`Toy v5, 2 000 pas : 5 433 arêtes, 520 nœuds, VRAM
5 702 Mo`) ; `archive/` et les six ZIP sortis du suivi (`git rm --cached`, fichiers
conservés sur disque) ; `.gitignore` : `*.zip`, `archive/`. Arbre suivi : **56
fichiers**, exactement le périmètre du ZIP.

### Étape 3 — première preuve comparative : **FAITE À MOITIÉ (250 pas sur 500)**

**3.1 Le baseline était faux — mesuré.** Premier lancement : perte **0,0000**, ppl de
validation **1,000 dès le pas 50**. Parfait, donc faux. Cause lue dans le code :
`TextDataset.sample_batch` renvoie `(x, x)`. C'est juste pour la colonie, qui lit sa
prédiction *avant* d'injecter (prouvé par `test_causality`), et faux pour un Transformer
causal dont le masque `triu(diagonal=1)` laisse `t` s'attentionner elle-même : il
apprenait l'identité. Corrigé par le décalage `logits[:, :-1]` contre `y[:, 1:]`. Sans ce
correctif, la comparaison aurait « prouvé » que la colonie perd contre un tricheur.

**3.2 Les deux courbes — mesuré.**

| pas | colonie perte | colonie ppl | arêtes | Transformer perte |
|---:|---:|---:|---:|---:|
| 50 | 9,1856 | 9 755 | 413 | 21,8981 |
| 100 | 9,0859 | 8 830 | 812 | 16,9997 |
| 150 | 8,9880 | 8 006 | 1 212 | 12,8336 |
| 200 | 8,8547 | 7 007 | 1 612 | 12,5231 |
| 250 | 8,6679 | 5 813 | 2 011 | 10,2979 |

Transformer, après **500** pas : perte d'entraînement 7,9369, **ppl de validation
6 060,503** (`baseline_ppl_500.txt`, 1 680 896 paramètres) [mesuré].
La colonie s'est arrêtée au pas **250** : la courbe n'est donc pas terminée.

**3.3 Débit — mesuré mais contaminé.** 3 832 → 8 242 ms/pas entre les pas 50 et 250,
pendant que `SkyrimSE.exe` occupait la carte : ces chiffres décrivent *colonie + jeu*,
jamais la colonie seule, et ne doivent pas être comparés aux 1 464 ms/pas du banc du
6 novembre (carte libre). Aucune courbe PNG : le tableau markdown est la source, comme
la consigne l'autorisait.

### Étape 4 — observation puis run texte : **BLOQUÉE, rien n'a été lancé**

Cause mesurée, pas supposée : `nvidia-smi` relève **7 929 Mo / 8 192 et 90 %
d'utilisation** par `SkyrimSE.exe` (PID 32564, processus étranger à ce travail, jamais
touché). Un run seq 128 / batch 16 dans cet état part en OOM dès les premières
activations. Aggravation mesurée dans la session : `OSError [WinError 1455]` au
chargement de `cublas64_12.dll` — la machine a épuisé son fichier de pagination, ce qui a
bloqué **jusqu'à la relance de pytest** et n'a été résolu qu'en arrêtant mon propre run.

Donc : observation 300 pas **non lancée**, seuils **non appliqués** (décider sur une ETA
non mesurée est exactement l'erreur déjà payée le 5 novembre), baseline 10 000 pas non
lancée (sa condition de départ était le lancement du run texte), `max_births` 8 → 4 non
appliqué.

Seul point d'ancrage non contaminé pour seq 128 — `logs_text.txt`, run du 3 novembre,
carte libre : 4 633 → 7 326 ms/pas. **Estimé** : 3 000 pas ≈ **3,9–6,1 h**, 10 000 pas
≈ 13–20 h. Le coût de `tbptt 4` n'est **pas** mesuré [supposé : il réduit la mémoire du
pas sans nécessairement réduire son temps].

### Étape 5 — verdict : **FAITE**

Verdict **Palier 1 : non**, sans ambiguïté, sur les échantillons du §5.

| Grandeur (pas 250) | Mesuré |
|---|---|
| arêtes | 2 011 (1 999 naissances, 0 mort) |
| nœuds vivants | 514 (512 initiaux + 2 nés d'un message) |
| morts-nœuds / morts-arêtes | 0 / 0 |
| E* prédit | ≈ 1 000 × naissances/pas ≈ 5 000 arêtes — non atteint |

À 250 pas la table n'a pas fini de croître : le plateau documenté n'est pas encore
observable, et rien ici ne le contredit.

## 3. Verdict de parallélisation, résumé

**Non parallélisable sans changement de sémantique.** Le tableau « banc avant-après »
est **sans objet** : il n'y a pas d'après, la consigne interdisant de forcer la
parallélisation au prix de la sémantique. Ce qui reste vrai : le batch est déjà
vectorisé, et l'unique axe séquentiel est `positions × tours` ; le seul levier mesuré
pour ce coût demeure `tbptt 4` (2 186 ms/pas, 1 240 Mo à 2 200 arêtes, 5 novembre).

## 4. Courbe comparative — les deux perplexités côte à côte, et ce qu'elle montre

```
colonie      ppl  5 813   (pas 250, run arrêté là)          [mesuré]
Transformer  ppl  6 060   (validation, après 500 pas)       [mesuré]
hasard uniforme  10 000
```

Elles ne sont **pas à budget égal** (250 pas contre 500) : on ne peut donc pas les
mettre en rapport. Ce que la comparaison montre honnêtement :

- le Transformer descend plus vite à ce budget (10,30 contre 8,67 au même pas 250) ;
- la colonie part beaucoup plus haut (9,19 contre 21,90) mais paie chaque pas dix fois
  plus cher ;
- 500 pas sur une tâche que le Transformer apprend en minutes ne sont pas un régime de
  croissance : **aucun vainqueur ne peut être déclaré**, et aucun avantage structurel ne
  peut être ni confirmé ni infirmé ici.

Le résultat le plus solide de cette étape n'est pas la courbe : c'est le bug du
baseline, qu'aucun test unitaire n'aurait attrapé.

## 5. Échantillons produits, en clair (42 tokens, amorce `Il était une fois`)

Les `�` et `\b` sont des artefacts de décodage d'un BPE à octets, nettoyés seulement à
l'affichage. Le plan demandait les pas 100 / 300 / 500 : le run s'étant arrêté au pas
250, **ces trois pas n'existent pas** et les échantillons disponibles sont ceux de 50 à
250.

**Pas 50** — `Il était une foisadmorage bul vivaient attiraitait ferméaboyerève illuminantâce sembler bi fauteuil odeurastiques profond exauça rougit donnantblè hors fruitier tourbillonner carr apparu étroitchèrent dedans tourbillon allu suffprise Framboiseteaux aperçut brusquement toupindre offrirent bruy`

**Pas 100** — `Il était une fois enveloppa quant comp cabossé réconforter remerciéuoué Jojo planterlat levaitenroulaoudouN Un Croa ro échap ponts plantèrent Basilearmon voilàUtilise BongoumesTiti fois Martine voiturette]iendrait peign Quelques permettrait pes amitié apais courageuse séparerinalement`

**Pas 150** — `Il était une fois luge crinière repassé faisantouver fermier sourire Al Zizi fâcheux igloo demandeBien souhaitait quartier Cher pensent vaches Gast Dupont ryth rais glaces amer empl fissurangeantassiette lettresersonne rencontreettait mélangées poussi Fl œil création réchauffa Dansjet impressionnés`

**Pas 200** — `Il était une fois passerait Fin regardermienvelop ballon réun rouges réisous pleuv meilleursaraignéeb tandis chanta éclat| Pao leçons coeur ramasser rassuré exam peint déc ard Coin bouillir avançaitCoin baignamusa traceD géantesoignoncheurherbesesses objet`

**Pas 250** — `Il était une fois rencontraraient sœursifférent détestait brisé propreté brique perles attendait lèvres cerisesotes franavocatitait emport soudainè insista réfrigérateur cur réunissaient Pierre lionceauordball termina aimons Griquer déb adorée soucier pédaleronge glissa Peux avalé ouverte essayer`

**Verdict Palier 1 : non.** L'exemple qui le prouve est le dernier : « Il était une fois
rencontraraient sœursifférent détestait brisé propreté brique perles attendait lèvres
cerisesotes franavocatitait emport soudainè insista ». Des mots français, pas de
syntaxe, des morphèmes collés (`sœursifférent`, `cerisesotes`, `franavocatitait`).
Aucune phrase grammaticale et compréhensible. C'est le résultat attendu à 250 pas sur
24 M de caractères, et ce n'est pas un résultat sur l'architecture.

## 6. Ce qui n'a PAS été fait, et pourquoi (le STOP est un résultat)

| Non fait | Raison |
|---|---|
| parallélisation (drapeau, test, banc) | verdict NON : la clé de routage lit la membrane du destinataire |
| observation 300 pas seq 128 / batch 16 / tbptt 4 | GPU à 7 929/8 192 Mo par un jeu ; OOM garanti |
| seuils automatiques (ETA < 48 h, 48–96 h, > 96 h) | aucune ETA mesurée en régime texte : les appliquer serait décider sur une estimation |
| `max_births` 8 → 4 | dépend des seuils ci-dessus ; déplacer E* est un choix scientifique non tranché |
| run comparatif 500 pas complet | arrêté au pas 250 : page épuisée (`WinError 1455`), checkpoint reprenable |
| baseline 10 000 pas | conditionnée au lancement du run texte, qui n'a pas eu lieu |
| extension à 10 000 pas | **accord humain requis**, et rien à étendre sans run de 3 000 pas |
| courbe PNG | le plan autorisait le tableau markdown, qui est la source reproductible |
| ms/pas propres de la colonie en régime texte | carte partagée : toute mesure y serait contaminée |

## 7. Livraison

- **Tests** : `pytest -q` nu → **43 passed, 1 warning en 119,80 s** [mesuré], après arrêt
  de mon propre run (quelques minutes plus tôt, la même commande échouait en `WinError 1455`
  faute de mémoire — blocage machine, pas échec de code).
- **Commits** (séparés par nature) : `2c91236` feat — sortie au niveau uniforme +
  fermeture des compteurs ; `c8067f0` chore — sortie d'`archive/` et des ZIP du suivi ;
  `293bfa0` docs — comparaison, échantillons, bilan ; `65ca610` docs — arrêt du run et
  sa cause. Tag annoté **`v5`**.
- **Arbre** : `git status` propre, 56 fichiers suivis.
- **ZIP complet horodaté** : `mycelia-v5-2026-11-07.zip`, reconstruit depuis
  `git ls-files` (jamais un patch), 56 entrées.

## 8. Prochaine action recommandée, chiffrée, et ce qu'il faut trancher

1. **Libérer la carte** (fermer le jeu, ou travailler sur une autre machine). C'est le
   seul vrai blocage ; aucun hyperparamètre ne le contourne.
2. **Reprendre** `python mycelia/train.py --data text --resume data/genome_comparaison.pt`
   pour finir les 250 pas restants — quelques dizaines de minutes [estimé].
3. **Observation** : `configs/config_observation.yaml` (seq 128, batch 16, tbptt 4),
   300 pas, puis ETA bornée **mesurée** → 3 000 pas coûterait 3,9–6,1 h [estimé sur
   seq 128 / tbptt 16, carte libre] → sous le seuil de 48 h, donc lancement.
4. **Baseline 10 000 pas** en parallèle dès ce lancement.

À trancher humainement, et par personne d'autre :

- **faut-il attendre une carte libre, ou déplacer le travail sur une machine dédiée ?**
  Tout le reste en dépend ;
- **accepte-t-on `tbptt 4`** pour le texte ? Il divise la mémoire par ~2,4 mais raccourcit
  la mémoire longue du modèle — c'est un choix de sémantique, pas de performance ;
- si l'ETA mesurée dépassait 96 h, **accepte-t-on de déplacer E\*** en passant
  `max_births` de 8 à 4 (E* ≈ 2 700) ? C'est un changement de propriété documentée.
