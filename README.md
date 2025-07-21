# Next Word Predictor

A probability-based next word predictor built in Python using n-gram language models. This implementation uses bigrams and trigrams to predict the most likely next word based on the current context.

## Features

- **Text Preprocessing**: Cleans and tokenizes input text
- **N-gram Models**: Uses bigrams and trigrams for contextual predictions
- **Probability Calculations**: Implements conditional probability for word prediction
- **Context Management**: Maintains sliding window of recent words
- **Fallback Strategy**: Uses unigram model when context is insufficient
- **Interactive Interface**: Command-line interface for real-time testing

## Architecture

### Core Components

1. **TextPreprocessor**: Handles text cleaning and tokenization
2. **NGramBuilder**: Builds frequency tables for bigrams and trigrams
3. **ProbabilityCalculator**: Computes conditional probabilities
4. **ContextManager**: Manages context queue with sliding window
5. **NextWordPredictor**: Main predictor class orchestrating all components

### Data Structures

```python
# N-gram storage
bigrams = {"hello": {"world": 5, "there": 3}}
trigrams = {"hello world": {"today": 2, "again": 1}}

# Context management
context_queue = deque(["word1", "word2"], maxlen=3)
```

## Training Data

The system is trained on two books from the Mahabharata:
- `dataset/01.txt`: Adi Parva (Book 1)
- `dataset/02.txt`: Sabha Parva (Book 2)

**Training Statistics:**
- Total words processed: 308,851
- Unique words: 12,816
- Bigram contexts: 12,815
- Trigram contexts: 114,174

## Usage

The Next Word Predictor offers multiple interfaces to suit different use cases:

### 1. Command Line Interface (CLI)

**Quick Start:**
```bash
python next_word_predictor.py
```

**Interactive Commands:**
- Type words to build context and get automatic predictions
- `predict` - Get predictions for current context
- `clear` - Clear the context
- `quit` - Exit the program

**Example Session:**
```
> the
Current context: the
Predictions: great (0.0856), king (0.0642), son (0.0428), same (0.0321), first (0.0285)

> great
Current context: the great
Predictions: rishi (0.1000), rishis (0.0706), energy (0.0509), asura (0.0441), strength (0.0271)
```

### 2. Desktop GUI (Tkinter)

**Start the GUI:**
```bash
python simple_ui.py
```

**Features:**
- 🖥️ Native desktop interface (no dependencies required)
- 📝 Real-time text input and prediction
- 🎯 Click predictions to add them to context
- 🧹 Clear context with one button
- 📊 Visual display of current context and predictions

**Requirements:** Python 3.6+ (tkinter is included with Python)

### 3. Web Interface (Flask)

**Install Flask (if not already installed):**
```bash
pip install flask
```

**Start the web server:**
```bash
python web_ui.py
```

**Access:** Open http://localhost:5000 in your browser

**Features:**
- 🌐 Browser-based interface
- 📱 Responsive design
- ⚡ Real-time API predictions
- 🔄 Interactive word selection
- 📊 JSON API endpoints

**API Endpoints:**
- `POST /predict` - Get predictions for text
- `POST /add_word` - Add word to context
- `POST /clear` - Clear context

### 4. Python Library/Module

**Basic Usage:**
```python
from next_word_predictor import NextWordPredictor

# Initialize and train
predictor = NextWordPredictor()
predictor.train_from_files(['dataset/01.txt', 'dataset/02.txt'])

# Add context and predict
predictor.update_context("the")
predictor.update_context("great")
predictions = predictor.predict_next_word(n=5)

for word, prob in predictions:
    print(f"{word}: {prob:.4f}")
```

**Advanced Usage:**
```python
# Clear context
predictor.clear_context()

# Get current context
context = predictor.get_context()
print(f"Current context: {' '.join(context)}")

# Process raw text
raw_text = "The great king ruled wisely"
words = predictor.preprocessor.preprocess_text(raw_text)
for word in words:
    predictor.update_context(word)

# Get multiple predictions
top_predictions = predictor.predict_next_word(n=10)
```

### 5. Graph Visualization

**Analyze the model structure:**
```bash
python graph_visualization.py
```

**Features:**
- 📈 Visualize n-gram connections as a graph
- 🔍 Analyze word relationships and probabilities
- 📤 Export graph data for external tools (Gephi, Cytoscape, D3.js)
- 📊 Statistical analysis of the model

**Example Analysis:**
- View how words connect to each other
- See probability distributions for word transitions
- Export data for advanced visualization tools

### Running Tests

```bash
python test_predictor.py
```

## Example Results

### Context: "the great"
```
1. "rishi" (probability: 0.100000)
2. "rishis" (probability: 0.070588)
3. "energy" (probability: 0.050889)
4. "asura" (probability: 0.044118)
5. "strength" (probability: 0.027060)
```

### Context: "arjuna said"
```
1. "o" (probability: 0.300000)
2. "i" (probability: 0.150000)
3. "this" (probability: 0.050000)
4. "what" (probability: 0.050000)
5. "blockhead" (probability: 0.050000)
```

### Context: "o son of"
```
1. "the" (probability: 0.111957)
2. "pandu" (probability: 0.106481)
3. "kunti" (probability: 0.081481)
4. "pritha" (probability: 0.035185)
5. "suvala" (probability: 0.025926)
```

## Algorithm

### Prediction Strategy

1. **Trigram Priority**: First attempts to use trigram context (last two words)
2. **Bigram Fallback**: Falls back to bigram context (last word) if trigram unavailable
3. **Unigram Baseline**: Uses most frequent words as final fallback
4. **Weighted Combination**: Combines predictions with different weights:
   - Trigrams: 70% weight
   - Bigrams: 50% weight
   - Unigrams: 10% weight

### Probability Calculation

```python
# Bigram probability
P(word2|word1) = count(word1, word2) / count(word1)

# Trigram probability
P(word3|word1, word2) = count(word1, word2, word3) / count(word1, word2)
```

## How It Works

### 1. Training Phase

**Text Preprocessing:**
The system first processes the training data through several steps:
```python
# Text cleaning and normalization
text = text.lower()                    # Convert to lowercase
text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
tokens = text.split()                  # Tokenize into words
```

**N-gram Generation:**
The preprocessed text is used to build frequency tables:
```python
# Unigrams (single words)
unigrams = {"the": 1500, "king": 450, "great": 320}

# Bigrams (word pairs)
bigrams = {
    "the": {"king": 89, "great": 67, "son": 45},
    "king": {"of": 156, "was": 78, "said": 34}
}

# Trigrams (three-word sequences)
trigrams = {
    "the great": {"king": 23, "warrior": 15, "sage": 12},
    "king of": {"the": 67, "dharma": 34, "pandavas": 21}
}
```

### 2. Prediction Process

**Context Management:**
The system maintains a sliding window of recent words:
```python
from collections import deque
context_queue = deque(maxlen=3)  # Stores last 3 words
context_queue.append("the")      # ["the"]
context_queue.append("great")    # ["the", "great"]
context_queue.append("king")     # ["the", "great", "king"]
```

**Multi-level Prediction Strategy:**
1. **Trigram Attempt:** Try using last two words as context
   ```python
   context = "the great"
   if context in trigrams:
       candidates = trigrams[context]  # {"king": 23, "warrior": 15, ...}
   ```

2. **Bigram Fallback:** Use last word if trigram unavailable
   ```python
   context = "great"
   if context in bigrams:
       candidates = bigrams[context]  # {"king": 45, "warrior": 23, ...}
   ```

3. **Unigram Baseline:** Use most frequent words as final fallback
   ```python
   candidates = unigrams  # {"the": 1500, "of": 1200, ...}
   ```

**Probability Weighting:**
The system combines predictions from different n-gram levels:
```python
def combine_predictions():
    final_scores = {}
    
    # Weight trigram predictions (70%)
    for word, count in trigram_predictions.items():
        final_scores[word] = count * 0.7
    
    # Add bigram predictions (50%)
    for word, count in bigram_predictions.items():
        final_scores[word] += count * 0.5
        
    # Add unigram baseline (10%)
    for word, count in unigram_predictions.items():
        final_scores[word] += count * 0.1
    
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
```

### 3. Real-time Operation

**Context Updates:**
As users type, the system continuously updates context:
```python
def update_context(new_word):
    # Add word to context queue (automatic sliding window)
    self.context_queue.append(new_word)
    
    # Generate new predictions with updated context
    predictions = self.predict_next_word(n=5)
    return predictions
```

**Adaptive Prediction:**
The system adapts to different context lengths:
- **No context:** Returns most frequent words (unigrams)
- **One word:** Uses bigram predictions
- **Two+ words:** Prioritizes trigram predictions with bigram fallback

### 4. Mathematical Foundation

**Conditional Probability:**
The core principle is estimating P(word|context) using maximum likelihood estimation:

```
P(w₃|w₁,w₂) = Count(w₁,w₂,w₃) / Count(w₁,w₂)
```

**Smoothing Strategy:**
The system handles unseen n-grams through hierarchical fallback:
- Missing trigram → fall back to bigram
- Missing bigram → fall back to unigram
- Missing unigram → return empty predictions

**Graph Structure:**
The n-gram model creates a probabilistic directed graph:
- **Nodes:** Individual words from the vocabulary
- **Edges:** Transition probabilities between words
- **Weights:** Conditional probabilities P(word₂|word₁)
- **Paths:** Possible word sequences through the graph

### 5. Performance Characteristics

**Time Complexity:**
- Training: O(n) where n = corpus size
- Prediction: O(1) dictionary lookups
- Context Update: O(1) deque operations

**Space Complexity:**
- Unigrams: O(|V|) where |V| = vocabulary size
- Bigrams: O(|V|²) in worst case
- Trigrams: O(|V|³) in worst case
- Actual usage: Much smaller due to Zipf's law distribution

**Memory Optimization:**
The system only stores n-grams that actually appear in the training data, making it much more memory-efficient than theoretical worst-case bounds.

## File Structure

```
Next word Predictor/
├── next_word_predictor.py    # Main implementation and CLI
├── simple_ui.py             # Desktop GUI (Tkinter)
├── web_ui.py                # Web interface (Flask)
├── graph_visualization.py   # Model visualization and analysis
├── test_predictor.py        # Test script
├── README.md                # This documentation
├── index.html               # Standalone web demo
├── dataset/
│   ├── 01.txt              # Training data (Adi Parva)
│   └── 02.txt              # Training data (Sabha Parva)
├── templates/
│   └── index.html          # Flask web UI template
└── js/
    ├── text-processor.js   # Client-side text processing
    └── ngram-builder.js    # JavaScript n-gram implementation
```

### Interface Options Summary

| Interface | File | Dependencies | Best For |
|-----------|------|--------------|----------|
| **Command Line** | [`next_word_predictor.py`](next_word_predictor.py) | None | Quick testing, scripting |
| **Desktop GUI** | [`simple_ui.py`](simple_ui.py) | None (tkinter built-in) | Offline use, native feel |
| **Web Interface** | [`web_ui.py`](web_ui.py) | Flask | Multi-user, remote access |
| **Standalone Demo** | [`index.html`](index.html) | None | Client-side demo |
| **Graph Analysis** | [`graph_visualization.py`](graph_visualization.py) | None | Model understanding |

## Implementation Details

### Text Preprocessing
- Converts to lowercase
- Removes punctuation
- Normalizes whitespace
- Filters empty tokens

### Context Management
- Uses `deque` with maximum length of 3
- Maintains sliding window of recent words
- Supports both bigram and trigram contexts

### Error Handling
- Graceful handling of file not found errors
- Fallback mechanisms for insufficient context
- Input validation and sanitization

## Performance Characteristics

- **Memory Usage**: Efficient dictionary-based storage
- **Training Time**: Linear in corpus size
- **Prediction Time**: Constant time O(1) lookups
- **Accuracy**: Context-dependent, typically 20-70% for common phrases

## Future Enhancements

- **Smoothing**: Add-one or Kneser-Ney smoothing for better handling of unseen n-grams
- **Dynamic Weighting**: Adaptive weighting based on context availability
- **Larger Context**: Support for 4-grams and 5-grams
- **Learning**: Online learning from user selections
- **GUI Interface**: Web-based or desktop GUI
- **Multiple Models**: Support for different domain-specific models

## Dependencies

- Python 3.6+
- Standard library only (no external dependencies)

## License

This project is open source and available under the MIT License.