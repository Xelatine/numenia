# ATLAS NYRA — MYCELI-SCAN v0

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
