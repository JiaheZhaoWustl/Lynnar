#!/usr/bin/env python
# viz_midjourney_chat.py
# ----------------------
# Visualize Midjourney chat data with multiple visualization options

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime
import re
import os

# Optional wordcloud import
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

def load_jsonl(filepath):
    """Load JSONL file and return list of conversations"""
    conversations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations

def extract_prompts(conversations):
    """Extract all prompts from conversations"""
    initial_prompts = []
    updated_prompts = []
    questions = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                content = msg['content']
                # Skip if it's an answer to a question (short response)
                if len(content.split()) > 10:  # Likely a prompt, not an answer
                    initial_prompts.append(content)
            elif msg['role'] == 'assistant':
                content = msg['content']
                if 'Updated prompt:' in content:
                    updated_prompts.append(content.replace('Updated prompt:', '').strip())
                elif '?' in content and 'Options:' in content:
                    questions.append(content)
    
    return initial_prompts, updated_prompts, questions

def plot_prompt_length_distribution(initial_prompts, updated_prompts, output_dir='outputs'):
    """Visualize distribution of prompt lengths"""
    initial_lengths = [len(p.split()) for p in initial_prompts]
    updated_lengths = [len(p.split()) for p in updated_prompts]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(initial_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Initial Prompt Length Distribution', fontsize=12)
    axes[0].set_xlabel('Number of Words')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(initial_lengths), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(initial_lengths):.1f}')
    axes[0].legend()
    
    axes[1].hist(updated_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Updated Prompt Length Distribution', fontsize=12)
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(updated_lengths), color='red', linestyle='--',
                    label=f'Mean: {np.mean(updated_lengths):.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prompt_length_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_length_comparison(initial_prompts, updated_prompts, output_dir='outputs'):
    """Compare initial vs updated prompt lengths"""
    initial_lengths = [len(p.split()) for p in initial_prompts]
    updated_lengths = [len(p.split()) for p in updated_prompts]
    
    # Match pairs (assuming same order)
    min_len = min(len(initial_lengths), len(updated_lengths))
    initial_lengths = initial_lengths[:min_len]
    updated_lengths = updated_lengths[:min_len]
    
    # Limit to first 50 for readability
    max_show = min(50, len(initial_lengths))
    initial_lengths = initial_lengths[:max_show]
    updated_lengths = updated_lengths[:max_show]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(initial_lengths))
    width = 0.35
    
    ax.bar(x - width/2, initial_lengths, width, label='Initial', alpha=0.7, color='skyblue')
    ax.bar(x + width/2, updated_lengths, width, label='Updated', alpha=0.7, color='lightcoral')
    
    ax.set_xlabel(f'Prompt Index (showing first {max_show})')
    ax.set_ylabel('Number of Words')
    ax.set_title('Initial vs Updated Prompt Length Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prompt_length_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_word_frequency(prompts, title, output_dir='outputs', top_n=30):
    """Plot most frequent words in prompts"""
    # Extract words (excluding Midjourney parameters)
    all_words = []
    param_pattern = re.compile(r'--\w+|--ar\s+\d+:\d+|--v\s+\d+\.?\d*')
    
    for prompt in prompts:
        # Remove Midjourney parameters
        clean_prompt = param_pattern.sub('', prompt)
        words = clean_prompt.lower().split()
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in words if len(w) > 3 and w not in stop_words]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    
    words, counts = zip(*top_words) if top_words else ([], [])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(words)), counts, color='steelblue')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Most Frequent Words - {title}')
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'word_frequency_{title.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_wordcloud(prompts, title, output_dir='outputs'):
    """Create word cloud from prompts"""
    if not HAS_WORDCLOUD:
        print("WordCloud module not available. Install with: pip install wordcloud")
        return
    
    # Combine all prompts
    text = ' '.join(prompts)
    
    # Remove Midjourney parameters
    param_pattern = re.compile(r'--\w+|--ar\s+\d+:\d+|--v\s+\d+\.?\d*')
    text = param_pattern.sub('', text)
    
    try:
        wordcloud = WordCloud(width=1200, height=600, background_color='white',
                            max_words=100, colormap='viridis').generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {title}', fontsize=16, pad=20)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'wordcloud_{title.lower().replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create word cloud: {e}")

def plot_conversation_stats(conversations, output_dir='outputs'):
    """Plot statistics about conversations"""
    conversation_lengths = []
    refinement_counts = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        conversation_lengths.append(len(messages))
        
        # Count refinement steps (Updated prompt messages)
        refinements = sum(1 for msg in messages if 'Updated prompt:' in msg.get('content', ''))
        refinement_counts.append(refinements)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(conversation_lengths, bins=20, alpha=0.7, color='mediumpurple', edgecolor='black')
    axes[0].set_title('Conversation Length Distribution', fontsize=12)
    axes[0].set_xlabel('Number of Messages')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(conversation_lengths), color='red', linestyle='--',
                    label=f'Mean: {np.mean(conversation_lengths):.1f}')
    axes[0].legend()
    
    axes[1].hist(refinement_counts, bins=10, alpha=0.7, color='gold', edgecolor='black')
    axes[1].set_title('Number of Refinements per Conversation', fontsize=12)
    axes[1].set_xlabel('Number of Refinements')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(refinement_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(refinement_counts):.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'conversation_stats.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_question_analysis(questions, output_dir='outputs'):
    """Analyze question patterns"""
    question_starts = []
    option_counts = []
    
    for q in questions:
        # Extract question text (before "Options:")
        if 'Options:' in q:
            question_text = q.split('Options:')[0].strip()
            question_starts.append(question_text.split()[0].lower() if question_text else '')
            
            # Count options
            options_part = q.split('Options:')[1] if 'Options:' in q else ''
            options = [opt.strip() for opt in options_part.split(',') if opt.strip()]
            option_counts.append(len(options))
    
    # Plot question start words
    start_counter = Counter(question_starts)
    top_starts = start_counter.most_common(15)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if top_starts:
        words, counts = zip(*top_starts)
        axes[0].barh(range(len(words)), counts, color='teal')
        axes[0].set_yticks(range(len(words)))
        axes[0].set_yticklabels(words)
        axes[0].set_xlabel('Frequency')
        axes[0].set_title('Most Common Question Start Words')
        axes[0].invert_yaxis()
    
    axes[1].hist(option_counts, bins=range(1, 6), alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_title('Number of Options per Question')
    axes[1].set_xlabel('Number of Options')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xticks(range(1, 6))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'question_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def print_summary_stats(conversations, initial_prompts, updated_prompts, questions):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("MIDJOURNEY CHAT DATA SUMMARY")
    print("="*60)
    print(f"Total Conversations: {len(conversations)}")
    print(f"Initial Prompts: {len(initial_prompts)}")
    print(f"Updated Prompts: {len(updated_prompts)}")
    print(f"Questions Asked: {len(questions)}")
    
    if initial_prompts:
        avg_initial = np.mean([len(p.split()) for p in initial_prompts])
        print(f"\nAverage Initial Prompt Length: {avg_initial:.1f} words")
    
    if updated_prompts:
        avg_updated = np.mean([len(p.split()) for p in updated_prompts])
        print(f"Average Updated Prompt Length: {avg_updated:.1f} words")
        if initial_prompts:
            length_change = avg_updated - avg_initial
            print(f"Average Length Change: {length_change:+.1f} words")
    
    if conversations:
        avg_conv_length = np.mean([len(c.get('messages', [])) for c in conversations])
        print(f"\nAverage Conversation Length: {avg_conv_length:.1f} messages")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Midjourney chat data with multiple visualization options"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file path (e.g., midjourney_chat_full_data.jsonl)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Output directory for visualizations (default: outputs)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all visualizations"
    )
    parser.add_argument(
        "--lengths",
        action="store_true",
        help="Plot prompt length distributions"
    )
    parser.add_argument(
        "--wordfreq",
        action="store_true",
        help="Plot word frequency charts"
    )
    parser.add_argument(
        "--wordcloud",
        action="store_true",
        help="Create word clouds"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Plot conversation statistics"
    )
    parser.add_argument(
        "--questions",
        action="store_true",
        help="Analyze question patterns"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    conversations = load_jsonl(args.input)
    initial_prompts, updated_prompts, questions = extract_prompts(conversations)
    
    # Print summary
    print_summary_stats(conversations, initial_prompts, updated_prompts, questions)
    
    # Generate visualizations
    if args.all or args.lengths:
        print("Generating length distribution plots...")
        plot_prompt_length_distribution(initial_prompts, updated_prompts, args.output_dir)
        plot_length_comparison(initial_prompts, updated_prompts, args.output_dir)
    
    if args.all or args.wordfreq:
        print("Generating word frequency charts...")
        if initial_prompts:
            plot_word_frequency(initial_prompts, "Initial Prompts", args.output_dir)
        if updated_prompts:
            plot_word_frequency(updated_prompts, "Updated Prompts", args.output_dir)
    
    if args.all or args.wordcloud:
        print("Generating word clouds...")
        if initial_prompts:
            create_wordcloud(initial_prompts, "Initial Prompts", args.output_dir)
        if updated_prompts:
            create_wordcloud(updated_prompts, "Updated Prompts", args.output_dir)
    
    if args.all or args.stats:
        print("Generating conversation statistics...")
        plot_conversation_stats(conversations, args.output_dir)
    
    if args.all or args.questions:
        print("Analyzing questions...")
        if questions:
            plot_question_analysis(questions, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()

