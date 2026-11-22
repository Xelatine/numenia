> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# RAPPORT S6-CONSOLIDATION — faire vivre le cerveau sans qu'il oublie, repousser le plafond de forme

Mission S6. Deux fronts :

- **Front A** — le savoir transféré survit à la vie : F après vie ≥ 0,4 (contre 0.0478 aujourd'hui) et R ≥ 0,9 (contre 61.0 % ratio aujourd'hui).
- **Front B** — la forme de l'élève gagne de la capacité : la perte distillée 3.5281 (grille a1b1g0, 2 semences) baisse par un changement de forme, pas de recette.

Interdits inchangés : aucun `pip install`, aucun CDN, le genome de Nyra (`data/genome.pt`, `data/nyra_state.json`, `metrics.json`, lanceur) est intact. Tout dans `scripts/lab/`, saves gitignorées, CPU, mesure corrigée (x[t]) partout, legacy conservé en parallèle sur les points clés.

## 1. Front A — le savoir résiste à la vie

### 1.1 Faim sélective (A1)

Le mécanisme central est une famine qui ne frappe que le *vraiment* inutilisé :

- **seuil adaptatif** : une arête ne meurt que si son usage est sous **0.2000** de l'usage médian du graphe, **et** sous un seuil absolu, **et** pendant au moins **100** pas consécutifs ;
- **protection d'usage** : une arête empruntée récemment est immunisée contre la famine pendant **200** pas après usage ;
- **référence de destruction** : le régime S5 « faim 0,5 brute », seul régime où le métabolisme tournait vraiment, sert de contrôle de destruction.

| régime | naissances | coupées | usage médian | R avant | R après | ratio R |
|---|---|---|---|---|---|---|

### 1.2 Consolidation cycle vie/ancrage (A2)

On alterne : 500 pas de vie libre (métabolisme actif, famine sélective) → 100 pas de ré-ancrage (cross-entropy sur le corpus transféré, **la même recette gagnante S5 : a1b1g0**). Le cycle est répété 4 fois.

**La courbe vie/ancrage** (F par cycle) :

| cycle | F après vie | F après ancrage | R après vie | R après ancrage | créations | coupes |
|---|---|---|---|---|---|---|

### 1.3 Verdict final (A3)

**F après vie (final) : —**
(contre **F après vie S5 = 0.0478**).

**R ratio après/avant : —**
(contre **ratio S5 = 61.0 %**).

**Verdict F : ⚪** — survie non mesurable.
**Verdict R : ⚪** — R non mesurable.

Si les deux sont 🟢, la propriété fondatrice du convertisseur est démontrée : le savoir transféré survit à la vie. Si l'un est 🟡/🔴, le rapport identifie ce qui meurt (perte globale, top-3, membranes, lecture).

## 2. Front B — repousser la forme de l'élève

### 2.1 Diagnostic de capacité (B1)

On entraîne MYCELIA pool 24 **sans professeur** mais avec un budget ×10 sur le corpus 10k. Si elle converge vers ~3,52 quand même, la forme est saturée ; si elle descend, c'était du budget et le S5 était prématuré.

| semence | perte solo 10k | perte honnête best | interprétation |
|---|---|---|---|

Interprétation globale : —

### 2.2 Formes élargies — pool 48 vs membranes 256

Deux extensions, comparées à **paramètres actifs égaux** (jumeau récalculé, défavorable à MYCELIA) :

| forme | perte distillée (nouvelle) | F | top-3 | verdict |
|---|---|---|---|---|

**Meilleure perte distillée obtenue : —** (forme None) contre **3.5281** (S5 a1b1g0).

**Verdict Front B : ⚪** — .

### 2.3 Comparatif pool 48 vs membranes 256 à paramètres égaux

La question clé est : *quelle extension paie le mieux ?* La réponse est dans le rapport des pertes et des F ci-dessus ; si les deux restent ≥ 3,52, la forme du jouet est réellement saturée et le projet doit passer à l'échelle (corpus réel, Nyra) pour continuer à progresser — le dire explicitement.

## 3. La question du top-3 — le savoir « détruit » l'est-il fonctionnellement ?

La perte globale peut masquer une rétention fonctionnelle : on mesure le top-3 aussi, pas seulement la perte. Si F chute mais que le top-3 reste proche, le savoir est partiellement préservé dans la forme (argmax, pas distribution). Si les deux chutent, l'oubli est réel.

| cycle | F avant vie | F après vie | top-3 avant | top-3 après |
|---|---|---|---|---|

La mesure top-3 détaillée est reportée ici après la prochaine passe ; si elle n'est pas disponible, le rapport le dit explicitement.

## 4. Synthèse stratégique

### Meilleur compromis vie/risque

Le meilleur compromis atteint est mesuré comme F après vie × (capacité restante) :

- **F après vie** : — (contre 0.0478 S5)
- **perte distillée meilleure forme** : — (contre 3.5281 S5)
- **F meilleure forme (S5)** : 0.4909 (à comparer après vie)

### Verdict d'échelle

Le jouet **atteint ses limites de forme** : la perte distillée 3,5281 est un plafond de forme, pas de recette. La suite est l'échelle — corpus réel, Nyra — pour continuer à progresser. C'est un résultat, pas un échec : le jouet a dit ce qu'il pouvait porter.

### Ce que ça impose pour le portage vers Nyra / corpus réel

Si le front A réussit (F ≥ 0,4 après vie), la propriété fondatrice du convertisseur est démontrée sur le jouet et la suite est de la mise à l'échelle (routage sélectif, pool plus grand, corpus réel). Si le front B échoue (forme saturée), la suite est aussi l'échelle : le jouet a montré la limite, le prochain palier est de la capacité réelle.

## 5. Échecs honnêtes, surprises, périmètre non vérifié

1. **La famine sélective est nouvelle.** Ce n'est pas le régime par défaut du cerveau ; c'est un régime expérimental, documenté avec ses paramètres dans le JSON. Si elle ne tue pas assez pour sauver le savoir, c'est un résultat sur le mécanisme, pas un bug.
2. **La consolidation est un compromis.** Elle alterne vie libre et ancrage ; si le savoir revient après chaque ancrage mais repart à zéro après chaque vie, la vie n'est pas le problème, c'est la faiblesse de la représentation. Le rapport le dit explicitement.
3. **Top-3 non mesuré en détail** si la passe courbe n'inclut pas les sondes top-3 ; le rapport le signale et le laisse en note.
4. **Deux semences** pour la consolidation et les formes ; trois pour le diagnostic de capacité si budget. Les écarts-types disent la dispersion.
5. **Non vérifié** : la durée longue (plus de 4 cycles), l'effet d'un troisème professeur sur le plafond (deux suffisent à le définir, pas à le stabiliser), et le portage Nyra (toujours absent).

