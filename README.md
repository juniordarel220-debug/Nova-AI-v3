# NOVA AI V3.0

## Fichiers
- `app.py` : application Gradio
- `requirements.txt` : dépendances Python
- `render.yaml` : configuration de déploiement Render

## Lancement local
```bash
pip install -r requirements.txt
export GROQ_API_KEY="votre_cle"
python app.py
```

Sous Windows PowerShell :
```powershell
$env:GROQ_API_KEY="votre_cle"
python app.py
```

## Déploiement Render
1. Dépose ces fichiers dans un dépôt GitHub.
2. Sur Render, crée un nouveau **Web Service** lié au dépôt.
3. Build Command : `pip install -r requirements.txt`
4. Start Command : `python app.py`
5. Dans Environment, ajoute `GROQ_API_KEY` (ou `OPENAI_API_KEY`) avec ta clé.
6. Ne mets jamais la clé API dans `app.py`, GitHub ou un message public.

## Notes importantes
- Le modèle Groq sélectionné doit accepter le format de chat utilisé.
- L'analyse d'image dépend d'un modèle vision compatible. Les modèles texte seuls
  peuvent refuser les images : choisis un modèle vision compatible si disponible.
- La transcription audio dépend du support audio du fournisseur et du modèle.
- Les sessions Gradio sont isolées pendant leur durée de vie, mais l'historique
  n'est pas persistant après redémarrage. Le stockage persistant multi-utilisateur
  nécessite une base de données avec authentification.
- Le mode Ollama local n'est pas inclus dans ce déploiement Render : Render ne peut
  pas accéder au Ollama qui tourne sur ton téléphone ou ordinateur personnel.
