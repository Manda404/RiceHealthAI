# 🌾 RiceHealthAI

**RiceHealthAI** est un projet de recherche et d'intelligence artificielle dédié à la **détection automatique des maladies des feuilles de riz** à partir d’images.  
Le projet s’appuie sur le **dataset Mendeley Rice Leaf Disease** et explore l’usage de **l’apprentissage profond (Deep Learning)** pour améliorer la santé des cultures.

---

## 🧠 Objectif du projet

L'objectif principal est de développer un **modèle de classification d’images** capable d’identifier automatiquement les principales maladies du riz :

- **Bacterial Blight**
- **Blast**
- **Brown Spot**
- **Tungro**

Ce projet vise à contribuer à la recherche agricole et à la prévention précoce des maladies grâce à l’IA.

---

## 📦 Dataset

Le dataset utilisé provient de **Mendeley Data** et contient les classes suivantes :

| Maladie | Nombre d’images |
|----------|-----------------|
| Bacterial Blight | 1584 |
| Blast | 1440 |
| Brown Spot | 1600 |
| Tungro | 1308 |

- **Source** : [Mendeley Data – Rice Leaf Disease Dataset](https://data.mendeley.com/)  
- **Structure** :  
```bash
RiceHealthAI/
│
├── pyproject.toml                  # 📦 Configuration Poetry (dépendances, scripts, etc.)
├── poetry.lock
├── README.md
├── .gitignore
│
├── data/                           # Données brutes et dérivées (non versionnées)
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/                      # Expérimentations, EDA
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_experiments.ipynb
│
├── configs/                        # 🧩 Configs YAML centralisées
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── src/                            # 🧠 Code applicatif (core + interfaces)
│   ├── ricehealthai/
│   │   ├── __init__.py
│   │   │
│   │   ├── domain/                 # 💡 Couche "ENTITÉS" : logique métier pure
│   │   │   ├── __init__.py
│   │   │   ├── model_entity.py     # structure de données métier (image, label)
│   │   │   └── metrics_entity.py   # entité pour les résultats de performance
│   │   │
│   │   ├── use_cases/              # 🚀 Couche "APPLICATION" : logique de cas d’usage
│   │   │   ├── __init__.py
│   │   │   ├── train_model.py      # orchestrateur entraînement
│   │   │   ├── evaluate_model.py   # orchestrateur évaluation
│   │   │   └── preprocess_data.py  # orchestrateur data pipeline
│   │   │
│   │   ├── infrastructure/         # ⚙️ Couche "INFRA" : implémentations concrètes
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py      # chargement d’images, DataLoader PyTorch
│   │   │   ├── image_transformer.py# augmentations
│   │   │   ├── model_repository.py # sauvegarde/chargement du modèle
│   │   │   ├── registry_manager.py # gestion des modèles sauvegardés
│   │   │   └── logger.py           # logger structuré (loguru, etc.)
│   │   │
│   │   ├── adapters/               # 🔌 Couche "INTERFACE" : points d’entrée/sortie
│   │   │   ├── __init__.py
│   │   │   ├── cli/                # interface CLI (typer ou click)
│   │   │   │   ├── __init__.py
│   │   │   │   └── main.py         # CLI principale pour exécuter les cas d’usage
│   │   │   └── api/                # (optionnel) API FastAPI / Streamlit
│   │   │       ├── __init__.py
│   │   │       └── app.py
│   │   │
│   │   └── core/                   # 🔬 Fonctions transversales
│   │       ├── __init__.py
│   │       ├── settings.py         # lecture configs YAML
│   │       ├── utils.py            # helpers génériques
│   │       └── exceptions.py
│   │
│   └── scripts/
│       ├── train.py                # point d’entrée pour entraîner le modèle
│       ├── evaluate.py
│       └── preprocess.py
│
├── models/
│   ├── best_model.pth
│   └── label_encoder.pkl
│
├── logs/
│   ├── training.log
│   └── evaluation.log
│
├── tests/                          # 🧪 Tests unitaires et d’intégration
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_train_model.py
│   └── test_config_loader.py
│
└── docs/
    ├── architecture.md
    ├── design_decisions.md
    └── api_reference.md
```