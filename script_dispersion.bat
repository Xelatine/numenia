@echo off
echo Demarrage du calcul de divergence S8...
cd /g/mycelia
python scripts/lab/s8_train_donors.py --divergence > data/lab/logs/log_s8_divergence.txt 2>&1
echo Calcul termine.
echo.
echo Tail du log:
type data/lab/logs/log_s8_divergence.txt
echo.
echo Fichiers dans s8/donors:
dir data/lab/s8/donors/*.pt
echo.
echo Fichiers dans s8:
dir data/lab/s8/*.json
