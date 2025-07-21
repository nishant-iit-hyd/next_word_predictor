#!/usr/bin/env python3
"""
Graph Visualization for Next Word Predictor
Shows how the n-gram model creates a probabilistic word graph
"""

import json
from collections import defaultdict
from next_word_predictor import NextWordPredictor

class WordGraphVisualizer:
    """Visualizes the word prediction model as a graph"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def create_word_graph(self, max_nodes=50):
        """
        Create a graph representation showing words as nodes and probabilities as edges
        
        Returns:
            Dictionary representing the graph structure
        """
        graph = {
            'nodes': [],
            'edges': [],
            'statistics': {}
        }
        
        # Get most common words as nodes
        word_frequencies = dict(self.predictor.ngram_builder.unigrams)
        most_common_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        
        # Create nodes
        for word, freq in most_common_words:
            graph['nodes'].append({
                'id': word,
                'frequency': freq,
                'type': 'word'
            })
        
        node_words = set([word for word, _ in most_common_words])
        
        # Create bigram edges
        for word1 in node_words:
            if word1 in self.predictor.ngram_builder.bigrams:
                total_count = self.predictor.ngram_builder.bigram_counts[word1]
                
                for word2, count in self.predictor.ngram_builder.bigrams[word1].items():
                    if word2 in node_words:  # Only include edges between our selected nodes
                        probability = count / total_count
                        
                        graph['edges'].append({
                            'source': word1,
                            'target': word2,
                            'weight': count,
                            'probability': round(probability, 4),
                            'type': 'bigram'
                        })
        
        # Add statistics
        graph['statistics'] = {
            'total_nodes': len(graph['nodes']),
            'total_edges': len(graph['edges']),
            'graph_type': 'Probabilistic Word Graph (N-gram based)',
            'edge_weights': 'Transition probabilities between words'
        }
        
        return graph
    
    def analyze_word_connections(self, word, top_n=10):
        """
        Analyze connections for a specific word
        
        Args:
            word: Word to analyze
            top_n: Number of top connections to show
            
        Returns:
            Dictionary with incoming and outgoing connections
        """
        analysis = {
            'word': word,
            'outgoing_edges': [],  # Words that can follow this word
            'incoming_edges': [],  # Words that can precede this word
            'trigram_contexts': []  # Trigram contexts involving this word
        }
        
        # Outgoing edges (bigrams: word -> next_word)
        if word in self.predictor.ngram_builder.bigrams:
            total_count = self.predictor.ngram_builder.bigram_counts[word]
            
            for next_word, count in self.predictor.ngram_builder.bigrams[word].items():
                probability = count / total_count
                analysis['outgoing_edges'].append({
                    'target': next_word,
                    'count': count,
                    'probability': round(probability, 4)
                })
            
            # Sort by probability
            analysis['outgoing_edges'].sort(key=lambda x: x['probability'], reverse=True)
            analysis['outgoing_edges'] = analysis['outgoing_edges'][:top_n]
        
        # Incoming edges (find words that can precede this word)
        for source_word, targets in self.predictor.ngram_builder.bigrams.items():
            if word in targets:
                count = targets[word]
                total_count = self.predictor.ngram_builder.bigram_counts[source_word]
                probability = count / total_count
                
                analysis['incoming_edges'].append({
                    'source': source_word,
                    'count': count,
                    'probability': round(probability, 4)
                })
        
        # Sort incoming edges by probability
        analysis['incoming_edges'].sort(key=lambda x: x['probability'], reverse=True)
        analysis['incoming_edges'] = analysis['incoming_edges'][:top_n]
        
        # Trigram contexts
        for context, targets in self.predictor.ngram_builder.trigrams.items():
            if word in targets:
                count = targets[word]
                total_count = self.predictor.ngram_builder.trigram_counts[context]
                probability = count / total_count
                
                analysis['trigram_contexts'].append({
                    'context': context,
                    'count': count,
                    'probability': round(probability, 4)
                })
        
        # Sort trigram contexts by probability
        analysis['trigram_contexts'].sort(key=lambda x: x['probability'], reverse=True)
        analysis['trigram_contexts'] = analysis['trigram_contexts'][:top_n]
        
        return analysis
    
    def print_graph_structure(self):
        """Print a text-based representation of the graph structure"""
        print("🔗 WORD GRAPH STRUCTURE")
        print("=" * 50)
        
        print(f"📊 Graph Statistics:")
        print(f"   • Total unique words (nodes): {len(self.predictor.ngram_builder.unigrams):,}")
        print(f"   • Bigram connections (edges): {sum(len(targets) for targets in self.predictor.ngram_builder.bigrams.values()):,}")
        print(f"   • Trigram contexts: {len(self.predictor.ngram_builder.trigrams):,}")
        
        print(f"\n🎯 Graph Properties:")
        print(f"   • Type: Directed Probabilistic Graph")
        print(f"   • Nodes: Words from the corpus")
        print(f"   • Edges: Transition probabilities between words")
        print(f"   • Weights: P(word₂|word₁) for bigrams, P(word₃|word₁,word₂) for trigrams")
        
        # Show some example paths
        print(f"\n🛤️  Example Graph Paths:")
        examples = ["the", "king", "arjuna", "said"]
        
        for word in examples:
            if word in self.predictor.ngram_builder.bigrams:
                connections = self.predictor.ngram_builder.bigrams[word]
                top_3 = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:3]
                
                print(f"\n   '{word}' → ")
                for next_word, count in top_3:
                    total = self.predictor.ngram_builder.bigram_counts[word]
                    prob = count / total
                    print(f"      → '{next_word}' (p={prob:.3f})")
    
    def export_graph_data(self, filename="word_graph.json"):
        """Export graph data to JSON file for external visualization"""
        graph_data = self.create_word_graph(max_nodes=100)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Graph data exported to {filename}")
        print(f"   Can be imported into graph visualization tools like:")
        print(f"   • Gephi, Cytoscape, D3.js, NetworkX, etc.")


def main():
    """Demonstrate the graph structure of the word predictor"""
    print("🔮 Next Word Predictor - Graph Analysis")
    print("=" * 50)
    
    # Initialize and train predictor
    print("Training model...")
    predictor = NextWordPredictor()
    training_files = ['dataset/01.txt', 'dataset/02.txt']
    predictor.train_from_files(training_files)
    
    # Create visualizer
    visualizer = WordGraphVisualizer(predictor)
    
    # Show graph structure
    visualizer.print_graph_structure()
    
    # Analyze specific words
    print(f"\n🔍 WORD CONNECTION ANALYSIS")
    print("=" * 50)
    
    test_words = ["the", "king", "arjuna", "said", "great"]
    
    for word in test_words:
        print(f"\n📝 Analysis for word: '{word}'")
        print("-" * 30)
        
        analysis = visualizer.analyze_word_connections(word, top_n=5)
        
        print(f"Outgoing edges (words that can follow '{word}'):")
        for edge in analysis['outgoing_edges'][:5]:
            print(f"   '{word}' → '{edge['target']}' (p={edge['probability']})")
        
        print(f"\nIncoming edges (words that can precede '{word}'):")
        for edge in analysis['incoming_edges'][:5]:
            print(f"   '{edge['source']}' → '{word}' (p={edge['probability']})")
        
        if analysis['trigram_contexts']:
            print(f"\nTop trigram contexts:")
            for ctx in analysis['trigram_contexts'][:3]:
                print(f"   '{ctx['context']}' → '{word}' (p={ctx['probability']})")
    
    # Export graph data
    print(f"\n📤 EXPORT GRAPH DATA")
    print("=" * 50)
    visualizer.export_graph_data()
    
    print(f"\n✨ GRAPH INTERPRETATION")
    print("=" * 50)
    print(f"This n-gram model creates a probabilistic graph where:")
    print(f"• Each WORD is a NODE")
    print(f"• Each TRANSITION is an EDGE with PROBABILITY weight")
    print(f"• Prediction = Finding highest probability paths from current context")
    print(f"• Context = Current position in the graph")
    print(f"• Next word = Most likely outgoing edge from current node(s)")


if __name__ == "__main__":
    main()