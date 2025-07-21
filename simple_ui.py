#!/usr/bin/env python3
"""
Simple GUI for the Next Word Predictor using tkinter
No external dependencies required - uses Python's built-in tkinter
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from next_word_predictor import NextWordPredictor

class NextWordPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔮 Next Word Predictor")
        self.root.geometry("700x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize predictor
        self.predictor = None
        self.is_trained = False
        
        self.setup_ui()
        self.train_model()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame, 
            text="🔮 Next Word Predictor",
            font=('Arial', 24, 'bold'),
            fg='#2c3e50',
            bg='#f0f0f0'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="AI-powered word prediction using N-gram models",
            font=('Arial', 12),
            fg='#7f8c8d',
            bg='#f0f0f0'
        )
        subtitle_label.pack()
        
        # Input section
        input_frame = tk.LabelFrame(
            self.root,
            text="Input Text",
            font=('Arial', 12, 'bold'),
            fg='#2c3e50',
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        input_frame.pack(fill='x', padx=20, pady=10)
        
        # Text input
        self.text_input = tk.Entry(
            input_frame,
            font=('Arial', 14),
            relief='solid',
            bd=1
        )
        self.text_input.pack(fill='x', pady=5)
        self.text_input.bind('<Return>', lambda e: self.predict_from_input())
        
        # Buttons
        button_frame = tk.Frame(input_frame, bg='#f0f0f0')
        button_frame.pack(pady=5)
        
        self.predict_btn = tk.Button(
            button_frame,
            text="Predict Next Word",
            command=self.predict_from_input,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=20,
            pady=8
        )
        self.predict_btn.pack(side='left', padx=5)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="Clear Context",
            command=self.clear_context,
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief='flat',
            padx=20,
            pady=8
        )
        self.clear_btn.pack(side='left', padx=5)
        
        # Context display
        context_frame = tk.LabelFrame(
            self.root,
            text="Current Context",
            font=('Arial', 12, 'bold'),
            fg='#2c3e50',
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        context_frame.pack(fill='x', padx=20, pady=10)
        
        self.context_display = tk.Label(
            context_frame,
            text="Start typing to build context...",
            font=('Arial', 12),
            fg='#7f8c8d',
            bg='#ecf0f1',
            relief='sunken',
            padx=10,
            pady=10,
            anchor='w',
            justify='left'
        )
        self.context_display.pack(fill='x', pady=5)
        
        # Predictions section
        predictions_frame = tk.LabelFrame(
            self.root,
            text="Word Predictions",
            font=('Arial', 12, 'bold'),
            fg='#2c3e50',
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        predictions_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Predictions listbox
        listbox_frame = tk.Frame(predictions_frame, bg='#f0f0f0')
        listbox_frame.pack(fill='both', expand=True)
        
        self.predictions_listbox = tk.Listbox(
            listbox_frame,
            font=('Arial', 12),
            relief='solid',
            bd=1,
            height=8
        )
        self.predictions_listbox.pack(side='left', fill='both', expand=True)
        self.predictions_listbox.bind('<Double-Button-1>', self.on_prediction_select)
        
        # Scrollbar for listbox
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side='right', fill='y')
        self.predictions_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.predictions_listbox.yview)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 10),
            fg='#7f8c8d',
            bg='#ecf0f1',
            relief='sunken',
            anchor='w',
            padx=10
        )
        status_bar.pack(fill='x', side='bottom')
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="💡 Tip: Type text and press Enter or click 'Predict'. Double-click predictions to add them to context.",
            font=('Arial', 10),
            fg='#27ae60',
            bg='#f0f0f0'
        )
        instructions.pack(pady=5)
    
    def train_model(self):
        """Train the model in a separate thread"""
        def train():
            try:
                self.status_var.set("Training model... Please wait.")
                self.root.update()
                
                self.predictor = NextWordPredictor()
                training_files = ['dataset/01.txt', 'dataset/02.txt']
                self.predictor.train_from_files(training_files)
                
                self.is_trained = True
                self.status_var.set("Model trained successfully! Ready to predict.")
                self.predict_btn.config(state='normal')
                self.clear_btn.config(state='normal')
                
                # Initial predictions
                self.update_predictions()
                
            except Exception as e:
                self.status_var.set(f"Error training model: {str(e)}")
                messagebox.showerror("Training Error", f"Failed to train model:\n{str(e)}")
        
        # Disable buttons during training
        self.predict_btn.config(state='disabled')
        self.clear_btn.config(state='disabled')
        
        # Start training in background thread
        training_thread = threading.Thread(target=train)
        training_thread.daemon = True
        training_thread.start()
    
    def predict_from_input(self):
        """Get predictions from input text"""
        if not self.is_trained:
            messagebox.showwarning("Not Ready", "Model is still training. Please wait.")
            return
        
        text = self.text_input.get().strip()
        if not text:
            return
        
        try:
            # Clear context and add new words
            self.predictor.clear_context()
            words = self.predictor.preprocessor.preprocess_text(text)
            for word in words:
                self.predictor.update_context(word)
            
            self.text_input.delete(0, tk.END)  # Clear input
            self.update_display()
            self.status_var.set(f"Predicted for: '{text}'")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction:\n{str(e)}")
    
    def clear_context(self):
        """Clear the current context"""
        if not self.is_trained:
            return
        
        try:
            self.predictor.clear_context()
            self.update_display()
            self.status_var.set("Context cleared.")
        except Exception as e:
            messagebox.showerror("Error", f"Error clearing context:\n{str(e)}")
    
    def update_display(self):
        """Update context and predictions display"""
        if not self.is_trained:
            return
        
        # Update context display
        context = ' '.join(self.predictor.get_context())
        if context:
            self.context_display.config(text=context, fg='#2c3e50')
        else:
            self.context_display.config(text="Start typing to build context...", fg='#7f8c8d')
        
        self.update_predictions()
    
    def update_predictions(self):
        """Update predictions listbox"""
        if not self.is_trained:
            return
        
        try:
            predictions = self.predictor.predict_next_word(n=10)
            
            # Clear current predictions
            self.predictions_listbox.delete(0, tk.END)
            
            if predictions:
                for i, (word, prob) in enumerate(predictions, 1):
                    display_text = f"{i:2d}. \"{word}\" ({prob:.4f})"
                    self.predictions_listbox.insert(tk.END, display_text)
            else:
                self.predictions_listbox.insert(tk.END, "No predictions available")
                
        except Exception as e:
            self.predictions_listbox.delete(0, tk.END)
            self.predictions_listbox.insert(tk.END, f"Error: {str(e)}")
    
    def on_prediction_select(self, event):
        """Handle prediction selection"""
        if not self.is_trained:
            return
        
        selection = self.predictions_listbox.curselection()
        if not selection:
            return
        
        try:
            # Extract word from selection
            selected_text = self.predictions_listbox.get(selection[0])
            if '"' in selected_text:
                word = selected_text.split('"')[1]
                
                # Add word to context
                clean_word = self.predictor.preprocessor.preprocess_text(word)
                if clean_word:
                    self.predictor.update_context(clean_word[0])
                    self.update_display()
                    self.status_var.set(f"Added '{word}' to context.")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding word to context:\n{str(e)}")

def main():
    """Run the GUI application"""
    root = tk.Tk()
    app = NextWordPredictorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed.")

if __name__ == "__main__":
    main()