#!/usr/bin/env python3
"""
Test script for the Next Word Predictor
"""

from next_word_predictor import NextWordPredictor

def test_predictor():
    """Test the next word predictor with various examples"""
    print("Testing Next Word Predictor")
    print("===========================")
    
    # Initialize and train predictor
    predictor = NextWordPredictor()
    training_files = ['dataset/01.txt', 'dataset/02.txt']
    
    print("Training model...")
    predictor.train_from_files(training_files)
    
    # Test cases with different contexts
    test_cases = [
        "the",
        "arjuna said",
        "o son of",
        "the great",
        "king yudhishthira",
        "the sons of",
        "having heard",
        "it was",
        "then said",
        "o bharata"
    ]
    
    print("\nTesting predictions with different contexts:")
    print("=" * 50)
    
    for test_case in test_cases:
        predictor.clear_context()
        
        # Add words to context
        words = predictor.preprocessor.preprocess_text(test_case)
        for word in words:
            predictor.update_context(word)
        
        predictions = predictor.predict_next_word(n=5)
        
        print(f"\nContext: \"{test_case}\"")
        print(f"Current context queue: {predictor.get_context()}")
        
        if predictions:
            print("Top 5 predictions:")
            for i, (word, prob) in enumerate(predictions[:5], 1):
                print(f"  {i}. \"{word}\" (probability: {prob:.6f})")
        else:
            print("No predictions available")
        
        # Show bigram and trigram contexts being used
        bigram_context = predictor.context_manager.get_bigram_context()
        trigram_context = predictor.context_manager.get_trigram_context()
        
        if trigram_context:
            print(f"  Trigram context: \"{trigram_context}\"")
            if trigram_context in predictor.ngram_builder.trigrams:
                trigram_options = dict(predictor.ngram_builder.trigrams[trigram_context])
                print(f"  Available trigram completions: {len(trigram_options)}")
        
        if bigram_context:
            print(f"  Bigram context: \"{bigram_context}\"")
            if bigram_context in predictor.ngram_builder.bigrams:
                bigram_options = dict(predictor.ngram_builder.bigrams[bigram_context])
                print(f"  Available bigram completions: {len(bigram_options)}")

def test_interactive_session():
    """Test an interactive session simulation"""
    print("\n\nInteractive Session Simulation")
    print("=" * 50)
    
    predictor = NextWordPredictor()
    training_files = ['dataset/01.txt', 'dataset/02.txt']
    predictor.train_from_files(training_files)
    
    # Simulate typing a sentence word by word
    sentence = ["the", "great", "warrior"]
    
    print("Simulating typing sentence word by word:")
    for i, word in enumerate(sentence):
        predictor.update_context(word)
        print(f"\nAfter typing \"{word}\":")
        print(f"Current context: {' '.join(predictor.get_context())}")
        
        predictions = predictor.predict_next_word(n=3)
        if predictions:
            print("Next word suggestions:")
            for j, (pred_word, prob) in enumerate(predictions[:3], 1):
                print(f"  {j}. \"{pred_word}\" ({prob:.6f})")

if __name__ == "__main__":
    test_predictor()
    test_interactive_session()