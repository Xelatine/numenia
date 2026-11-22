> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# RAPPORT DU TRADUCTEUR — dense ↔ MYCELIA

MYCELI-SCAN v0 + mini-labo de conversion. Toutes les mesures viennent des fichiers de `data/lab/` ; les courbes sont celles relevees pendant l'execution, pas des estimations.

## 1. Resume

- Atlas de Nyra : 514 noeuds vivants, 285 actives par les 500 stimuli, 6 familles, k=6.
- Divergences structure/fonction : 162 aretes jamais co-actives (36.1%), 4236 paires co-actives sans arete.
- Dense → MYCELIA : 5.3605 → 5.1691 apres greffe (3.6% de gain), controle au meme budget 5.2152.
- MYCELIA → dense : 5.0082 → 4.7986 (4.2%), controle 4.8016.

## 2. Les deux bacs a sable (S2)

Meme tache (mot suivant), meme flux de mots, meme vocabulaire, 312 mots. Le dense a 17016 parametres, MYCELIA 167638 — l'ecart est assume : MYCELIA paie un pool, des aretes et un metabolisme que le dense n'a pas.

### micro-transformer dense (2 couches, 16 dims)

- parametres : 17016
- vocabulaire : 312 mots
- entrainement : 5.8 s
- perte d'entrainement : 5.9005 → 1.9650
- validation : 5.7797 → 3.6774 (meilleur pas 300, arret sur patience)

Courbe de validation : 5.780 → 4.458 → 3.919 → 3.677 → 3.718 → 3.954 → 4.264 → 4.548

| prefixe | mots proposes |
|---|---|
| le chat mange | ce / une / ma |
| la fille dort | pas / pas. / sans |
| je ne veux pas | ce / mon / un |
| pourquoi le chien court | je / ce / cette |
| il y a trois | je / tu / une |

### micro-MYCELIA (pool, aretes, metabolisme)

- parametres : 167638
- vocabulaire : 312 mots
- entrainement : 170.9 s
- perte d'entrainement : 5.7460 → 3.8788
- validation : 5.7358 → 5.1248 (meilleur pas 150, arret sur patience)

Courbe de validation : 5.736 → 5.356 → 5.131 → 5.125 → 5.217 → 5.418 → 5.499 → 5.548 → 5.623 → 5.722 → 5.805 → 5.902 → 5.915 → 5.985 → 6.070 → 5.125

| prefixe | mots proposes |
|---|---|
| le chat mange | je / les / y |
| la fille dort | je / les / ? |
| je ne veux pas | je / pas / ce |
| pourquoi le chien court | je / ? / son |
| il y a trois | cent / quatre / vingt |

Metabolisme du bac MYCELIA : pool 24 → 94 noeuds, 5 naissances reelles, 0 enterrements, 5 aretes coupees.

## 3. Le dictionnaire bilingue (alignement fonctionnel)

Chaque ligne est une paire de familles qui repondent aux memes phrases. L'alignement est mutuel : une paire ne compte que si chacun est le meilleur partenaire de l'autre, au-dessus du seuil.

### Dense → MYCELIA

| concept source | famille receveuse | similarite | stimuli partages |
|---|---|---|---|
| 2 | 4 | 0.993 | « je ne se ferme pas un ours. »; « je ne dort pas mon singe. » |

### MYCELIA → dense

| famille source | concept receveur | similarite | stimuli partages |
|---|---|---|---|
| 3 | 6 | 0.979 | « son frère heureux arrive. »; « facteur et facteur travaille vite. » |

## 4. Verdict dense → MYCELIA (S3)

- **AVANT** : validation 5.3605, entrainement 5.3998.
- **APRES 100 pas avec greffe** : validation 5.1691 (+3.6%), entrainement 5.0001.
- **Controle, 100 pas sans greffe** : validation 5.2152, entrainement 4.9852.
- Greffe contre controle : +0.88% de validation, -0.30% d'entrainement.
- Test de langue (20 phrases hors corpus) : greffe 6.5027, controle 6.5154, 13/20 phrases meilleures.

Courbe de validation — greffe : 5.294 → 5.237 → 5.188 → 5.160 → 5.129 → 5.130 → 5.121 → 5.103 → 5.130 → 5.169

Courbe de validation — controle : 5.295 → 5.241 → 5.195 → 5.174 → 5.151 → 5.152 → 5.180 → 5.162 → 5.153 → 5.215

Transplants : 7 noeuds-portes (7 concepts sans equivalent), pool 27 noeuds receveurs → 34.

## 5. Verdict MYCELIA → dense (S4)

- **AVANT** : validation 5.0082, entrainement 5.0701.
- **APRES 100 pas avec greffe** : validation 4.7986 (+4.2%), entrainement 4.7841.
- **Controle** : validation 4.8016, entrainement 4.7867.
- Greffe contre controle : +0.06% de validation.
- Test de langue : greffe 5.2030, controle 5.2064, 10/20 phrases meilleures.

Courbe de validation — greffe : 4.988 → 4.972 → 4.954 → 4.934 → 4.916 → 4.896 → 4.875 → 4.855 → 4.827 → 4.799

Courbe de validation — controle : 4.988 → 4.972 → 4.955 → 4.936 → 4.918 → 4.899 → 4.878 → 4.858 → 4.830 → 4.802

Transplants : 7 neurones caches (7 familles sans equivalent), parametres 17016 → 17247.

## 6. Echecs honnetes

- **Un seul appariement sur 8 dans les deux sens.** Avec le seuil 0,5 et un appariement mutuel, presque aucun concept n'a de correspondant : les deux cerveaux ne se ressemblent pas assez pour que la traduction soit dense. Les 7 autres sont donc greffes, pas traduits.
- **Le gain vient surtout du transplant, pas de l'alignement** : 7 noeuds ajoutes, dont aucun ne correspond a une famille existante. Un traducteur qui greffe tout ne traduit rien.
- **Le sens MYCELIA → dense est presque neutre** : +0.06% de validation contre le controle, 10/20 phrases. Sept neurones zero-initialises n'apportent presque rien en 100 pas.
- **Entrainement contre validation** : dans le sens dense → MYCELIA, la greffe gagne en validation (+0.88%) mais perd un peu en entrainement (-0.30%). Un gain de generalisation qui n'est pas un gain d'apprentissage doit etre lu comme un effet de regularisation, pas comme une superiorite.
- **Exactitude de prediction : 0 %** sur les 20 phrases dans les quatre bras. Les modeles apprennent la syntaxe du corpus jouet, pas ses mots : la tache a 312 mots reste hors de portee en quelques centaines de pas.
- **229 noeuds vivants ne s'allument sur aucun des 500 stimuli** — ni famille, ni concept. Ils sont invisibles au scan, donc absents du traducteur.
- Les deux sens sont mesures depuis un point d'amorcage (50 pas d'entrainement du receveur, puis 100 pas d'affinage) : converti depuis son optimum, un receveur ne peut que se degrader et la comparaison ne mesure plus rien.

## 7. Atlas de Nyra (copie du rapport de scan)

Cerveau : `data/genome.pt` — 532 noeuds, 514 vivants, 462 aretes, step 56.
Stimulation : 500 phrases, 2911 tokens, 32 dimensions par noeud.

Activation d'un noeud = norme L2 maximale de sa membrane sur les tokens du stimulus. Familles = k-means (k=6) sur le profil d'activation de chaque noeud.

## Familles fonctionnelles

| # | noeuds | taille | activation moy. | degre sortant | cables |
|---|---|---|---|---|---|
| 1 | 1, 6, 8, 10, 16, 21… | 66 | 5.185 | 1.48 | 66/66 |
| 0 | 11, 29, 64, 102, 113, 164… | 19 | 4.602 | 1.00 | 19/19 |
| 4 | 5, 19, 34, 47, 50, 56… | 63 | 4.246 | 1.54 | 63/63 |
| 5 | 3, 28, 42, 53, 55, 75… | 33 | 4.218 | 2.06 | 33/33 |
| 3 | 2, 4, 7, 12, 20, 25… | 82 | 3.635 | 1.72 | 82/82 |
| 2 | 91, 98, 131, 179, 229, 231… | 22 | 3.255 | 1.18 | 22/22 |

### Top-10 stimuli par famille

**Famille 1** (66 noeuds)

- `11.179` [nombre] six plus onze font treize.
- `11.122` [nombre] douze plus quinze font quatre.
- `10.878` [nombre] deux plus quinze font seize.
- `10.797` [nombre] cinq plus onze font seize.
- `10.620` [nombre] huit plus treize font quinze.
- `10.475` [nombre] neuf plus zéro font onze.
- `10.307` [nombre] seize plus quatorze font cinq.
- `10.024` [nombre] cent plus douze font quatorze.
- `9.839` [nombre] douze plus quatre font trente.
- `9.690` [nombre] treize plus neuf font quinze.

**Famille 0** (19 noeuds)

- `9.593` [nombre] neuf plus zéro font onze.
- `8.875` [nombre] six plus onze font treize.
- `8.644` [nombre] douze plus quinze font quatre.
- `8.230` [nombre] cinq plus onze font seize.
- `8.159` [nombre] cent plus douze font quatorze.
- `8.104` [nombre] quatre plus trois font zéro.
- `8.092` [nombre] deux plus quinze font seize.
- `7.935` [question] est-ce que la femme compte ?
- `7.748` [nombre] douze plus quatre font trente.
- `7.694` [nombre] huit plus treize font quinze.

**Famille 4** (63 noeuds)

- `9.214` [nombre] douze plus quinze font quatre.
- `8.630` [nombre] deux plus quinze font seize.
- `8.507` [nombre] huit plus treize font quinze.
- `8.474` [nombre] six plus onze font treize.
- `7.886` [nombre] treize plus neuf font quinze.
- `7.885` [nombre] neuf plus zéro font onze.
- `7.818` [nombre] seize plus quatorze font cinq.
- `7.740` [question] est-ce que la femme compte ?
- `7.719` [nombre] cent plus douze font quatorze.
- `7.708` [nombre] cinq plus onze font seize.

**Famille 5** (33 noeuds)

- `8.458` [question] est-ce que la femme compte ?
- `8.414` [nombre] douze plus quinze font quatre.
- `8.376` [question] est-ce que mon mer s'ouvre ?
- `8.205` [nombre] neuf plus zéro font onze.
- `7.985` [question] est-ce que un voiture tourne ?
- `7.961` [negation] il n'y a pas cette pluie.
- `7.944` [nombre] six plus onze font treize.
- `7.732` [question] est-ce que ma gâteau s'ouvre ?
- `7.684` [question] est-ce que une poule mange ?
- `7.320` [nombre] huit plus treize font quinze.

**Famille 3** (82 noeuds)

- `7.165` [nombre] deux plus quinze font seize.
- `6.918` [nombre] douze plus quinze font quatre.
- `6.833` [nombre] huit plus treize font quinze.
- `6.778` [nombre] six plus onze font treize.
- `6.676` [nombre] cinq plus onze font seize.
- `6.545` [nombre] seize plus quatorze font cinq.
- `6.378` [nombre] neuf plus zéro font onze.
- `6.372` [nombre] cent plus douze font quatorze.
- `6.201` [nombre] treize plus neuf font quinze.
- `5.896` [nombre] douze plus quatre font trente.

**Famille 2** (22 noeuds)

- `5.942` [nombre] huit plus treize font quinze.
- `5.787` [nombre] treize plus neuf font quinze.
- `5.631` [nombre] douze plus quinze font quatre.
- `5.523` [question] est-ce que mon mer s'ouvre ?
- `5.443` [question] quand est ma neige ?
- `5.400` [nombre] cinq plus onze font seize.
- `5.358` [question] est-ce que la femme compte ?
- `5.312` [nombre] deux plus quinze font seize.
- `5.295` [nombre] sept plus six font treize.
- `5.263` [nombre] six plus onze font treize.

## Divergences structure / fonction

- Aretes observees dans le scan : 449
- **Cablees mais jamais co-actives** : 162 (36.1% des aretes).
- **Co-actives sans arete** (>= 50 co-activations) : 4236 paires.

Exemples de cable muet : 443→328 (0), 328→355 (0), 112→214 (0), 112→43 (0), 170→402 (0), 383→276 (0)

Exemples de paire co-active sans synapse : 1~123 (500), 1~475 (500), 123~475 (500), 1~92 (499), 1~329 (499), 1~356 (499)

Lecture : une arete qui ne co-active jamais n'est pas forcement morte — elle peut transporter un signal qui ne fait pas changer d'etat. Une paire co-active sans arete n'est pas une erreur de cablage — elle peut passer par deux sauts. Ces deux listes sont les candidates a verifier une par une.


## 8. Fichiers

- `data/lab/stimuli.txt` — 500 stimuli (6 categories)
- `data/lab/matrix_stimxnodes.npy` — activation [500, 532]
- `data/lab/atlas_nyra.json`, `atlas_nyra.md` — atlas
- `data/lab/film_nyra.json` — film de 5 stimuli, token par token
- `data/lab/corpus_toy.txt` — 1000 phrases
- `data/lab/conversion_dense_to_mycelia.json`, `conversion_mycelia_to_dense.json`
