"""
title: Same Day Surgery Feedback Analyzer
description: Upload a PDF and extract sentiment from feedback about same-day surgery
version: 1.0
author: bobsterbeat
pipe_type: pipeline
"""

from pydantic import BaseModel, Field
from typing import List
import pdfplumber
from transformers import pipeline

class Input(BaseModel):
    pdf_file: str = Field(..., description="Path to the uploaded PDF file")

class Output(BaseModel):
    markdown: str

class Pipe:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def run(self, input: Input) -> Output:
        feedback_lines = []

        with pdfplumber.open(input.pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split("\n"):
                    if "same day surgery" in line.lower():
                        feedback_lines.append(line.strip())

        if not feedback_lines:
            return Output(markdown="❗ No 'same day surgery' feedback found.")

        result_lines = []
        pos, neg, neu = 0, 0, 0

        for line in feedback_lines:
            result = self.analyzer(line[:512])[0]
            label = result['label']
            score = round(result['score'], 3)

            if label == "POSITIVE":
                pos += 1
                label_display = f"✅ **Positive** ({score})"
            elif label == "NEGATIVE":
                neg += 1
                label_display = f"❌ **Negative** ({score})"
            else:
                neu += 1
                label_display = f"➖ Neutral ({score})"

            result_lines.append(f"- {label_display}: {line}")

        total = pos + neg + neu
        summary = (
            f"### Sentiment Breakdown\n"
            f"- Positive: {pos} ({round(100 * pos / total)}%)\n"
            f"- Negative: {neg} ({round(100 * neg / total)}%)\n"
            f"- Neutral: {neu} ({round(100 * neu / total)}%)\n\n"
            "### Feedback Extracted:\n"
        )

        return Output(markdown=summary + "\n".join(result_lines))
