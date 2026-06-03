#!/bin/bash

# Script de démarrage pour le bot Professeur Sandale
# Basé sur la structure du bot AM — personnalité remplacée par le Professeur Sandale
# Conçu pour Debian 12 (Proxmox LXC) - Version OpenAI

echo "Le Professeur Sandale s'initialise..."

SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
fi

echo "Vérification de Python et venv..."
if ! command -v python3 &> /dev/null || ! command -v pip &> /dev/null || ! dpkg -l | grep -q python3-venv; then
    echo "Installation des dépendances système..."
    $SUDO apt update
    $SUDO apt install python3 python3-venv python3-pip -y
fi

if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi

echo "Vérification et installation des paquets Python..."
venv/bin/pip install --upgrade pip
venv/bin/pip install discord.py openai python-dotenv

if [ ! -f ".env" ]; then
    echo ""
    echo "ATTENTION : Le fichier .env est absent."
    echo "Un modèle a été créé. Renseigne tes clés API, puis relance ce script."
    echo ""
    echo "DISCORD_TOKEN=\"TON_TOKEN_DISCORD_ICI\"" > .env
    echo "OPENAI_API_KEY=\"TA_CLE_API_OPENAI_ICI\"" >> .env
    exit 1
fi

echo "Lancement du Professeur Sandale..."
venv/bin/python bot_sandale.py
