#!/bin/bash
# lancement_donors.sh — Lancement du banc S8 donateurs
cd /g/mycelia
rm -f data/lab/logs/log_s8_donors.txt
python scripts/lab/s8_train_donors.py --all > data/lab/logs/log_s8_donors.txt 2>&1
echo "=== FIN DE LA TÂCHE ==="
