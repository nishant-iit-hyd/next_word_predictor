#!/usr/bin/env python3
"""
Next Word Predictor using N-grams
Based on probability calculations from bigrams and trigrams
"""

import re
import string
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


class TextPreprocessor:
    """Handles text cleaning and tokenization"""
    
    def __init__(self):
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing punctuation and normalizing whitespace
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = self.punctuation_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize cleaned text into words
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of words
        """
        if not text:
            return []
        
        words = text.split()
        # Filter out empty strings and very short words
        words = [word for word in words if len(word) > 0]
        
        return words
    
    def preprocess_text(self, raw_text: str) -> List[str]:
        """
        Complete preprocessing pipeline
        
        Args:
            raw_text: Raw input text
            
        Returns:
            List of cleaned and tokenized words
        """
        cleaned_text = self.clean_text(raw_text)
        tokens = self.tokenize(cleaned_text)
        return tokens


class NGramBuilder:
    """Builds n-gram models from preprocessed text"""
    
    def __init__(self):
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.trigrams = defaultdict(lambda: defaultdict(int))
        self.unigrams = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.total_words = 0
    
    def build_ngrams(self, tokens: List[str]) -> None:
        """
        Build bigram and trigram frequency tables
        
        Args:
            tokens: List of preprocessed tokens
        """
        if len(tokens) < 2:
            return
        
        self.total_words += len(tokens)
        
        # Build unigrams
        for token in tokens:
            self.unigrams[token] += 1
        
        # Build bigrams
        for i in range(len(tokens) - 1):
            word1, word2 = tokens[i], tokens[i + 1]
            self.bigrams[word1][word2] += 1
            self.bigram_counts[word1] += 1
        
        # Build trigrams
        for i in range(len(tokens) - 2):
            word1, word2, word3 = tokens[i], tokens[i + 1], tokens[i + 2]
            context = f"{word1} {word2}"
            self.trigrams[context][word3] += 1
            self.trigram_counts[context] += 1
    
    def get_ngram_stats(self) -> Dict[str, int]:
        """Get statistics about the n-gram models"""
        return {
            'total_words': self.total_words,
            'unique_words': len(self.unigrams),
            'bigram_contexts': len(self.bigrams),
            'trigram_contexts': len(self.trigrams)
        }


class ProbabilityCalculator:
    """Calculates probabilities for next word predictions"""
    
    def __init__(self, ngram_builder: NGramBuilder):
        self.ngram_builder = ngram_builder
    
    def calculate_bigram_probability(self, word1: str, word2: str) -> float:
        """
        Calculate P(word2|word1) using bigram model
        
        Args:
            word1: First word (context)
            word2: Second word (target)
            
        Returns:
            Conditional probability
        """
        if word1 not in self.ngram_builder.bigrams:
            return 0.0
        
        word1_count = self.ngram_builder.bigram_counts[word1]
        word1_word2_count = self.ngram_builder.bigrams[word1][word2]
        
        if word1_count == 0:
            return 0.0
        
        return word1_word2_count / word1_count
    
    def calculate_trigram_probability(self, context: str, word: str) -> float:
        """
        Calculate P(word|context) using trigram model
        
        Args:
            context: Two-word context (e.g., "hello world")
            word: Target word
            
        Returns:
            Conditional probability
        """
        if context not in self.ngram_builder.trigrams:
            return 0.0
        
        context_count = self.ngram_builder.trigram_counts[context]
        context_word_count = self.ngram_builder.trigrams[context][word]
        
        if context_count == 0:
            return 0.0
        
        return context_word_count / context_count
    
    def calculate_unigram_probability(self, word: str) -> float:
        """
        Calculate P(word) using unigram model (fallback)
        
        Args:
            word: Target word
            
        Returns:
            Probability based on frequency
        """
        if self.ngram_builder.total_words == 0:
            return 0.0
        
        word_count = self.ngram_builder.unigrams[word]
        return word_count / self.ngram_builder.total_words


class ContextManager:
    """Manages context queue for prediction"""
    
    def __init__(self, max_context_length: int = 3):
        self.context_queue = deque(maxlen=max_context_length)
    
    def update_context(self, new_word: str) -> None:
        """
        Add new word to context queue
        
        Args:
            new_word: Word to add to context
        """
        self.context_queue.append(new_word.lower())
    
    def get_trigram_context(self) -> Optional[str]:
        """
        Get current trigram context (last two words)
        
        Returns:
            Two-word context string or None if insufficient context
        """
        if len(self.context_queue) >= 2:
            return f"{self.context_queue[-2]} {self.context_queue[-1]}"
        return None
    
    def get_bigram_context(self) -> Optional[str]:
        """
        Get current bigram context (last word)
        
        Returns:
            Last word or None if no context
        """
        if len(self.context_queue) >= 1:
            return self.context_queue[-1]
        return None
    
    def clear_context(self) -> None:
        """Clear the context queue"""
        self.context_queue.clear()
    
    def get_context_words(self) -> List[str]:
        """Get current context as list of words"""
        return list(self.context_queue)


class NextWordPredictor:
    """Main next word predictor class"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.ngram_builder = NGramBuilder()
        self.probability_calculator = None
        self.context_manager = ContextManager()
        self.is_trained = False
    
    def train_from_file(self, file_path: str) -> None:
        """
        Train the model from a text file
        
        Args:
            file_path: Path to training text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            tokens = self.preprocessor.preprocess_text(text)
            self.ngram_builder.build_ngrams(tokens)
            
            print(f"Processed {file_path}: {len(tokens)} tokens")
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
    
    def train_from_files(self, file_paths: List[str]) -> None:
        """
        Train the model from multiple text files
        
        Args:
            file_paths: List of paths to training text files
        """
        for file_path in file_paths:
            self.train_from_file(file_path)
        
        # Initialize probability calculator after training
        self.probability_calculator = ProbabilityCalculator(self.ngram_builder)
        self.is_trained = True
        
        # Print training statistics
        stats = self.ngram_builder.get_ngram_stats()
        print("\nTraining completed!")
        print(f"Total words processed: {stats['total_words']}")
        print(f"Unique words: {stats['unique_words']}")
        print(f"Bigram contexts: {stats['bigram_contexts']}")
        print(f"Trigram contexts: {stats['trigram_contexts']}")
    
    def predict_next_word(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next words based on current context
        
        Args:
            n: Number of predictions to return
            
        Returns:
            List of (word, probability) tuples sorted by probability
        """
        if not self.is_trained:
            print("Error: Model is not trained yet.")
            return []
        
        predictions = defaultdict(float)
        
        # Try trigram prediction first
        trigram_context = self.context_manager.get_trigram_context()
        if trigram_context and trigram_context in self.ngram_builder.trigrams:
            for word, count in self.ngram_builder.trigrams[trigram_context].items():
                prob = self.probability_calculator.calculate_trigram_probability(trigram_context, word)
                predictions[word] = max(predictions[word], prob * 0.7)  # Weight trigrams higher
        
        # Try bigram prediction
        bigram_context = self.context_manager.get_bigram_context()
        if bigram_context and bigram_context in self.ngram_builder.bigrams:
            for word, count in self.ngram_builder.bigrams[bigram_context].items():
                prob = self.probability_calculator.calculate_bigram_probability(bigram_context, word)
                predictions[word] = max(predictions[word], prob * 0.5)  # Weight bigrams moderately
        
        # Fallback to unigram (most frequent words) if no context predictions
        if not predictions:
            for word, count in self.ngram_builder.unigrams.items():
                prob = self.probability_calculator.calculate_unigram_probability(word)
                predictions[word] = prob * 0.1  # Lower weight for unigrams
        
        # Sort predictions by probability (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:n]
    
    def update_context(self, word: str) -> None:
        """
        Update context with a new word
        
        Args:
            word: New word to add to context
        """
        self.context_manager.update_context(word)
    
    def get_context(self) -> List[str]:
        """Get current context words"""
        return self.context_manager.get_context_words()
    
    def clear_context(self) -> None:
        """Clear the current context"""
        self.context_manager.clear_context()
    
    def get_word_suggestions(self, predictions: List[Tuple[str, float]]) -> str:
        """
        Format predictions for display
        
        Args:
            predictions: List of (word, probability) tuples
            
        Returns:
            Formatted string for display
        """
        if not predictions:
            return "No predictions available."
        
        result = "Next word predictions:\n"
        for i, (word, prob) in enumerate(predictions, 1):
            result += f"{i}. \"{word}\" (probability: {prob:.4f})\n"
        
        return result


def main():
    """Simple command-line interface for the next word predictor"""
    print("Next Word Predictor")
    print("==================")
    
    # Initialize predictor
    predictor = NextWordPredictor()
    
    # Train on dataset files
    training_files = ['dataset/01.txt', 'dataset/02.txt']
    print("Training model...")
    predictor.train_from_files(training_files)
    
    print("\nType words to build context, then press Enter to get predictions.")
    print("Commands: 'predict' (get predictions), 'clear' (clear context), 'quit' (exit)")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                predictor.clear_context()
                print("Context cleared.")
                continue
            elif user_input.lower() == 'predict':
                predictions = predictor.predict_next_word()
                print(predictor.get_word_suggestions(predictions))
                continue
            elif user_input:
                # Add words to context
                words = predictor.preprocessor.preprocess_text(user_input)
                for word in words:
                    predictor.update_context(word)
                
                print(f"Current context: {' '.join(predictor.get_context())}")
                
                # Auto-predict after adding words
                predictions = predictor.predict_next_word()
                print(predictor.get_word_suggestions(predictions))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()