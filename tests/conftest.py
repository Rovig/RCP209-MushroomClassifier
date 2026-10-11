import sys
from pathlib import Path

# Les modules du projet sont à la racine du dépôt.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
