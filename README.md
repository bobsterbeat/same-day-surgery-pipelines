# Same Day Surgery Feedback Pipeline

A custom pipeline for Open WebUI to:

- Upload a PDF containing patient feedback
- Extract lines mentioning "same day surgery"
- Run sentiment analysis using Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english`
- Return a markdown-formatted summary with sentiment breakdown

## Requirements
Make sure the Docker container has these installed:
- `pdfplumber`
- `transformers`
- `torch`

## How to Use in Open WebUI
1. Go to **Admin → Pipelines → Add from GitHub**
2. Paste the **raw link** to this `.py` file
3. Restart Open WebUI container
4. Go to Pipelines tab and run it with a PDF input
