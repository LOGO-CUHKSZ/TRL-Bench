#!/usr/bin/env python
"""
Table Question Answering using TAPAS (HuggingFace implementation).

TAPAS answers questions about tables by:
1. Selecting relevant cells from the table
2. Optionally applying an aggregation operator (COUNT, SUM, AVERAGE, etc.)

This script provides utilities for:
- Single question answering on a table
- Batch question answering
- Sequential (conversational) question answering

Usage:
    # Single question
    python tapas_qa.py \
        --table /path/to/table.csv \
        --question "What is the total revenue?"

    # Multiple questions
    python tapas_qa.py \
        --table /path/to/table.csv \
        --questions "What is the total?" "How many rows?" "What is the average?"

    # Conversational QA (SQA model)
    python tapas_qa.py \
        --table /path/to/table.csv \
        --question "What is his age?" \
        --model google/tapas-base-finetuned-sqa \
        --conversational

    # Batch mode with questions file
    python tapas_qa.py \
        --table /path/to/table.csv \
        --questions_file questions.txt \
        --output answers.json

References:
    - Paper: https://arxiv.org/abs/2004.02349
    - HuggingFace: https://huggingface.co/docs/transformers/model_doc/tapas
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import numpy as np
import pandas as pd

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TAPASQuestionAnswering:
    """
    TAPAS Question Answering using HuggingFace Transformers.

    Supports:
    - Cell selection (pointing to answer cells)
    - Aggregation operations (COUNT, SUM, AVERAGE, NONE)
    - Conversational QA (follow-up questions)
    """

    # Aggregation labels used by TAPAS
    AGGREGATION_LABELS = {
        0: "NONE",
        1: "SUM",
        2: "AVERAGE",
        3: "COUNT"
    }

    # Recommended models for different tasks
    RECOMMENDED_MODELS = {
        'wtq': 'google/tapas-base-finetuned-wtq',      # WikiTableQuestions
        'sqa': 'google/tapas-base-finetuned-sqa',      # Sequential QA
        'wikisql': 'google/tapas-base-finetuned-wikisql-supervised',
    }

    def __init__(
        self,
        model_name: str = 'google/tapas-base-finetuned-wtq',
        device: str = None,
        max_length: int = 512
    ):
        """
        Initialize TAPAS QA model.

        Args:
            model_name: HuggingFace model identifier
                - For general QA: google/tapas-base-finetuned-wtq
                - For conversational QA: google/tapas-base-finetuned-sqa
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length
        """
        try:
            from transformers import TapasTokenizer, TapasForQuestionAnswering
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.max_length = max_length
        self.model_name = model_name

        print(f"Loading TAPAS QA model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully on {device}")

    def _prepare_table(self, df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
        """Prepare DataFrame for TAPAS processing."""
        df = df.head(max_rows).copy()

        # Convert all values to strings
        for col in df.columns:
            df[col] = df[col].astype(str).fillna('')

        df.columns = [str(c) for c in df.columns]
        return df

    def _get_answer_coordinates(
        self,
        logits: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Get predicted cell coordinates from logits.

        Args:
            logits: Cell selection logits [seq_len]
            token_type_ids: Token type IDs [seq_len, 7]
            attention_mask: Attention mask [seq_len]

        Returns:
            List of (row, col) tuples for selected cells
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Get row and column IDs from token_type_ids
        # Index 2 = row_ids, Index 1 = column_ids
        row_ids = token_type_ids[:, 2]
        col_ids = token_type_ids[:, 1]

        # Find cells with probability > 0.5
        selected_mask = (probs > 0.5) & (attention_mask == 1) & (row_ids > 0)

        if not selected_mask.any():
            # If no cells selected, take the highest probability cell
            valid_mask = (attention_mask == 1) & (row_ids > 0)
            if valid_mask.any():
                valid_probs = probs.clone()
                valid_probs[~valid_mask] = -float('inf')
                best_idx = valid_probs.argmax().item()
                return [(row_ids[best_idx].item() - 1, col_ids[best_idx].item() - 1)]
            return []

        # Collect unique (row, col) pairs
        coordinates = set()
        for idx in selected_mask.nonzero(as_tuple=True)[0]:
            row = row_ids[idx].item() - 1  # Convert to 0-indexed
            col = col_ids[idx].item() - 1
            if row >= 0 and col >= 0:
                coordinates.add((row, col))

        return sorted(list(coordinates))

    def _get_answer_text(
        self,
        df: pd.DataFrame,
        coordinates: List[Tuple[int, int]],
        aggregation: str
    ) -> str:
        """
        Get the final answer text from coordinates and aggregation.

        Args:
            df: Input DataFrame
            coordinates: List of (row, col) cell coordinates
            aggregation: Aggregation operation

        Returns:
            Answer string
        """
        if not coordinates:
            return ""

        # Extract cell values
        values = []
        for row, col in coordinates:
            if row < len(df) and col < len(df.columns):
                val = df.iloc[row, col]
                values.append(str(val))

        if not values:
            return ""

        if aggregation == "NONE":
            return ", ".join(values)

        # Try to convert to numbers for aggregation
        numeric_values = []
        for v in values:
            try:
                # Handle common number formats
                clean_v = v.replace(',', '').replace('$', '').replace('%', '')
                numeric_values.append(float(clean_v))
            except ValueError:
                continue

        if not numeric_values:
            return ", ".join(values)

        if aggregation == "COUNT":
            return str(len(numeric_values))
        elif aggregation == "SUM":
            return str(sum(numeric_values))
        elif aggregation == "AVERAGE":
            return str(sum(numeric_values) / len(numeric_values))

        return ", ".join(values)

    def answer(
        self,
        table: Union[str, pd.DataFrame],
        question: str,
        max_rows: int = 100,
        return_details: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Answer a question about a table.

        Args:
            table: Path to CSV file or pandas DataFrame
            question: Question string
            max_rows: Maximum rows to process
            return_details: If True, return detailed dict instead of just answer

        Returns:
            Answer string, or dict with details if return_details=True
        """
        # Load table
        if isinstance(table, str):
            df = pd.read_csv(table, nrows=max_rows, dtype=str)
        else:
            df = table.copy()

        df = self._prepare_table(df, max_rows)

        # Tokenize
        inputs = self.tokenizer(
            table=df,
            queries=question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predictions
        logits = outputs.logits[0]  # Remove batch dim
        aggregation_logits = outputs.logits_aggregation[0] if outputs.logits_aggregation is not None else None

        # Get aggregation prediction
        if aggregation_logits is not None:
            agg_idx = aggregation_logits.argmax().item()
            aggregation = self.AGGREGATION_LABELS.get(agg_idx, "NONE")
        else:
            aggregation = "NONE"

        # Get cell coordinates
        coordinates = self._get_answer_coordinates(
            logits,
            inputs['token_type_ids'][0],
            inputs['attention_mask'][0]
        )

        # Get answer text
        answer_text = self._get_answer_text(df, coordinates, aggregation)

        if return_details:
            # Get selected cell values
            selected_cells = []
            for row, col in coordinates:
                if row < len(df) and col < len(df.columns):
                    selected_cells.append({
                        'row': row,
                        'column': col,
                        'column_name': df.columns[col],
                        'value': df.iloc[row, col]
                    })

            return {
                'answer': answer_text,
                'aggregation': aggregation,
                'coordinates': coordinates,
                'selected_cells': selected_cells,
                'question': question
            }

        return answer_text

    def answer_batch(
        self,
        table: Union[str, pd.DataFrame],
        questions: List[str],
        max_rows: int = 100,
        return_details: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Answer multiple questions about the same table.

        Args:
            table: Path to CSV file or pandas DataFrame
            questions: List of question strings
            max_rows: Maximum rows to process
            return_details: If True, return detailed dicts

        Returns:
            List of answers (strings or dicts)
        """
        results = []
        for question in questions:
            result = self.answer(
                table, question,
                max_rows=max_rows,
                return_details=return_details
            )
            results.append(result)
        return results

    def answer_conversational(
        self,
        table: Union[str, pd.DataFrame],
        questions: List[str],
        max_rows: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Answer sequential/conversational questions about a table.

        Each question can reference previous questions/answers.
        Best used with SQA-trained models.

        Args:
            table: Path to CSV file or pandas DataFrame
            questions: List of questions in conversational order
            max_rows: Maximum rows to process

        Returns:
            List of answer dicts with conversation context
        """
        results = []
        conversation_history = []

        for i, question in enumerate(questions):
            result = self.answer(
                table, question,
                max_rows=max_rows,
                return_details=True
            )

            result['turn'] = i + 1
            result['previous_questions'] = conversation_history.copy()
            results.append(result)

            conversation_history.append({
                'question': question,
                'answer': result['answer']
            })

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Answer questions about tables using TAPAS'
    )
    parser.add_argument('--table', type=str, required=True,
                        help='Path to CSV file')
    parser.add_argument('--question', type=str, default=None,
                        help='Single question to answer')
    parser.add_argument('--questions', type=str, nargs='+', default=None,
                        help='Multiple questions to answer')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='File with questions (one per line)')
    parser.add_argument('--model', type=str, default='google/tapas-base-finetuned-wtq',
                        help='HuggingFace model name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--max_rows', type=int, default=100,
                        help='Maximum rows to process')
    parser.add_argument('--conversational', action='store_true',
                        help='Use conversational (sequential) QA mode')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--details', action='store_true',
                        help='Show detailed output with cell coordinates')

    args = parser.parse_args()

    # Collect questions
    questions = []
    if args.question:
        questions.append(args.question)
    if args.questions:
        questions.extend(args.questions)
    if args.questions_file:
        with open(args.questions_file, 'r') as f:
            questions.extend([line.strip() for line in f if line.strip()])

    if not questions:
        parser.error("At least one question required (--question, --questions, or --questions_file)")

    # Load model
    qa = TAPASQuestionAnswering(
        model_name=args.model,
        device=args.device,
        max_length=512
    )

    # Answer questions
    print(f"\nTable: {args.table}")
    print(f"Questions: {len(questions)}")
    print("-" * 50)

    if args.conversational:
        results = qa.answer_conversational(
            args.table, questions, max_rows=args.max_rows
        )
    else:
        results = qa.answer_batch(
            args.table, questions,
            max_rows=args.max_rows,
            return_details=args.details
        )

    # Display results
    for i, (q, r) in enumerate(zip(questions, results)):
        print(f"\nQ{i+1}: {q}")
        if isinstance(r, dict):
            print(f"A{i+1}: {r['answer']}")
            if args.details:
                print(f"    Aggregation: {r['aggregation']}")
                print(f"    Cells: {r['coordinates']}")
        else:
            print(f"A{i+1}: {r}")

    # Save to file if requested
    if args.output:
        output_data = {
            'table': args.table,
            'model': args.model,
            'results': [
                {'question': q, 'answer': r if isinstance(r, str) else r}
                for q, r in zip(questions, results)
            ]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
