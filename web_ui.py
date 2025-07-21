#!/usr/bin/env python3
"""
Simple web UI for the Next Word Predictor using Flask
"""

try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

import json
from next_word_predictor import NextWordPredictor

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize and train the predictor"""
    global predictor
    if predictor is None:
        print("Initializing predictor...")
        predictor = NextWordPredictor()
        training_files = ['dataset/01.txt', 'dataset/02.txt']
        predictor.train_from_files(training_files)
        print("Predictor ready!")

if FLASK_AVAILABLE:
    app = Flask(__name__)

    @app.route('/')
    def home():
        """Serve the main page"""
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        """Handle prediction requests"""
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            # Clear context and add new words
            predictor.clear_context()
            words = predictor.preprocessor.preprocess_text(text)
            for word in words:
                predictor.update_context(word)
            
            # Get predictions
            predictions = predictor.predict_next_word(n=5)
            
            # Format response
            response = {
                'context': ' '.join(predictor.get_context()),
                'predictions': [
                    {'word': word, 'probability': round(prob, 6)}
                    for word, prob in predictions
                ]
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/add_word', methods=['POST'])
    def add_word():
        """Add word to context"""
        try:
            data = request.get_json()
            word = data.get('word', '').strip()
            
            if not word:
                return jsonify({'error': 'No word provided'}), 400
            
            # Add word to context
            clean_word = predictor.preprocessor.preprocess_text(word)[0] if predictor.preprocessor.preprocess_text(word) else word
            predictor.update_context(clean_word)
            
            # Get new predictions
            predictions = predictor.predict_next_word(n=5)
            
            response = {
                'context': ' '.join(predictor.get_context()),
                'predictions': [
                    {'word': word, 'probability': round(prob, 6)}
                    for word, prob in predictions
                ]
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/clear', methods=['POST'])
    def clear_context():
        """Clear the context"""
        try:
            predictor.clear_context()
            predictions = predictor.predict_next_word(n=5)
            
            response = {
                'context': '',
                'predictions': [
                    {'word': word, 'probability': round(prob, 6)}
                    for word, prob in predictions
                ]
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def main():
    """Run the web application"""
    if not FLASK_AVAILABLE:
        print("Please install Flask to use the web UI:")
        print("pip install flask")
        return
    
    print("Starting Next Word Predictor Web UI...")
    initialize_predictor()
    
    print("\nWeb UI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()