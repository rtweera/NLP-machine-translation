import google.generativeai as genai
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv

class GeminiDatasetGenerator:
    """
    Generate spelling error training datasets using Gemini LLM.
    Creates pairs of (misspelled, correct) sentences.
    """
    
    def __init__(self, api_key):
        """
        Initialize Gemini API.
        
        Args:
            api_key: Your Gemini API key from Google AI Studio
        """
        genai.configure(api_key=api_key) # type: ignore
        self.model = genai.GenerativeModel('gemini-2.5-flash') # type: ignore
        
    def generate_batch(self, num_pairs=100, theme=None):
        """
        Generate a batch of sentence pairs using Gemini.
        
        Args:
            num_pairs: Number of sentence pairs to generate
            theme: Optional theme for sentences (e.g., "news", "casual", "academic")
        
        Returns:
            List of (misspelled, correct) tuples
        """
        theme_text = f" with a {theme} theme" if theme else ""
        
        prompt = f"""Generate exactly {num_pairs} pairs of sentences for spelling correction training.
Each pair should have:
1. A sentence with realistic spelling errors (misspelled version)
2. The same sentence with correct spelling (correct version)

IMPORTANT INSTRUCTIONS:
- Include common spelling mistakes like: letter omissions, letter additions, letter transpositions, phonetic errors, double letter errors
- Make errors realistic (the kind humans actually make)
- Vary sentence length (short to medium, 5-15 words)
- Use diverse vocabulary and topics{theme_text}
- Do NOT use the same words repeatedly
- Include different types of errors in each sentence

Format your response EXACTLY as valid JSON:
{{
  "pairs": [
    {{"misspelled": "I hav a dreem today", "correct": "I have a dream today"}},
    {{"misspelled": "The qick brown fox jumps", "correct": "The quick brown fox jumps"}},
    ...
  ]
}}

Generate diverse, realistic examples. Start generating now:"""

        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to find JSON in the response
            if '```json' in response_text:
                # Extract JSON from code block
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                # Extract from generic code block
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                # Assume entire response is JSON
                json_text = response_text
            
            # Parse JSON
            data = json.loads(json_text)
            pairs = data.get('pairs', [])
            
            # Convert to tuple format
            result = [(p['misspelled'], p['correct']) for p in pairs]
            
            print(f"✓ Generated {len(result)} pairs")
            return result
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON parsing error: {e}")
            print(f"Response was: {response_text[:200]}...")    # type: ignore
            return []
        except Exception as e:
            print(f"✗ Error generating batch: {e}")
            return []
    
    def generate_language_model_corpus(self, num_sentences=100, theme=None):
        """
        Generate correct English sentences for language model training.
        
        Args:
            num_sentences: Number of sentences to generate
            theme: Optional theme for sentences
        
        Returns:
            List of correct sentences
        """
        theme_text = f" about {theme}" if theme else ""
        
        prompt = f"""Generate exactly {num_sentences} grammatically correct, well-written English sentences{theme_text}.

REQUIREMENTS:
- All sentences must be grammatically perfect
- Vary sentence length (5-20 words)
- Use diverse vocabulary
- Cover different topics and contexts
- No spelling or grammar errors
- Natural, fluent English

Format as JSON:
{{
  "sentences": [
    "The quick brown fox jumps over the lazy dog.",
    "She enjoys reading books in the library.",
    ...
  ]
}}

Generate now:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            data = json.loads(json_text)
            sentences = data.get('sentences', [])
            
            print(f"✓ Generated {len(sentences)} correct sentences")
            return sentences
            
        except Exception as e:
            print(f"✗ Error generating sentences: {e}")
            return []
    
    def generate_dataset(self, 
                        total_pairs=1000, 
                        batch_size=100, 
                        output_file='parallel_corpus.txt',
                        delay=2):
        """
        Generate full parallel corpus dataset by calling Gemini multiple times.
        
        Args:
            total_pairs: Total number of pairs to generate
            batch_size: Pairs per API call
            output_file: Output file path
            delay: Delay between API calls (seconds)
        """
        print("=" * 60)
        print("GENERATING PARALLEL CORPUS FOR SPELLING CORRECTION")
        print("=" * 60)
        print(f"Target: {total_pairs} sentence pairs")
        print(f"Batch size: {batch_size} pairs per call")
        print(f"Output: {output_file}")
        print("=" * 60)
        print()
        
        # Themes to add variety
        themes = [
            None, "news", "casual conversation", "academic writing",
            "technology", "daily life", "business", "education",
            "travel", "health", "science", "entertainment"
        ]
        
        all_pairs = []
        num_batches = (total_pairs + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            theme = themes[i % len(themes)]
            theme_text = f" ({theme})" if theme else ""
            
            print(f"Batch {i+1}/{num_batches}{theme_text}...", end=" ")
            
            pairs = self.generate_batch(batch_size, theme)
            
            if pairs:
                all_pairs.extend(pairs)
                
                # Append to file immediately (in case of interruption)
                with open(output_file, 'a', encoding='utf-8') as f:
                    for misspelled, correct in pairs:
                        f.write(f"{misspelled}\t{correct}\n")
                
                print(f"Total: {len(all_pairs)} pairs")
            else:
                print("Failed!")
            
            # Delay to avoid rate limiting
            if i < num_batches - 1:
                time.sleep(delay)
        
        print()
        print("=" * 60)
        print(f"✓ Dataset generation complete!")
        print(f"✓ Generated {len(all_pairs)} pairs")
        print(f"✓ Saved to: {output_file}")
        print("=" * 60)
        
        return all_pairs
    
    def generate_lm_corpus(self,
                          total_sentences=1000,
                          batch_size=100,
                          output_file='lm_corpus.txt',
                          delay=2):
        """
        Generate language model corpus by calling Gemini multiple times.
        
        Args:
            total_sentences: Total sentences to generate
            batch_size: Sentences per API call
            output_file: Output file path
            delay: Delay between API calls (seconds)
        """
        print("=" * 60)
        print("GENERATING LANGUAGE MODEL CORPUS")
        print("=" * 60)
        print(f"Target: {total_sentences} sentences")
        print(f"Batch size: {batch_size} sentences per call")
        print(f"Output: {output_file}")
        print("=" * 60)
        print()
        
        themes = [
            None, "news articles", "daily conversations", "formal writing",
            "technology topics", "everyday situations", "professional contexts",
            "educational content", "general knowledge", "various subjects"
        ]
        
        all_sentences = []
        num_batches = (total_sentences + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            theme = themes[i % len(themes)]
            theme_text = f" ({theme})" if theme else ""
            
            print(f"Batch {i+1}/{num_batches}{theme_text}...", end=" ")
            
            sentences = self.generate_language_model_corpus(batch_size, theme)
            
            if sentences:
                all_sentences.extend(sentences)
                
                # Append to file
                with open(output_file, 'a', encoding='utf-8') as f:
                    for sentence in sentences:
                        f.write(f"{sentence}\n")
                
                print(f"Total: {len(all_sentences)} sentences")
            else:
                print("Failed!")
            
            if i < num_batches - 1:
                time.sleep(delay)
        
        print()
        print("=" * 60)
        print(f"✓ Language model corpus complete!")
        print(f"✓ Generated {len(all_sentences)} sentences")
        print(f"✓ Saved to: {output_file}")
        print("=" * 60)
        
        return all_sentences


def load_dataset(parallel_file, lm_file):
    """
    Load generated datasets for use in SMT corrector.
    
    Args:
        parallel_file: Path to parallel corpus file
        lm_file: Path to language model corpus file
    
    Returns:
        parallel_corpus, lm_corpus
    """
    # Load parallel corpus
    parallel_corpus = []
    with open(parallel_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                misspelled, correct = line.strip().split('\t')
                parallel_corpus.append((misspelled, correct))
    
    # Load LM corpus
    lm_corpus = []
    with open(lm_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                lm_corpus.append(sentence)
    
    return parallel_corpus, lm_corpus


def main():
    """
    Main function to generate datasets.
    """
    load_dotenv(override=True, verbose=True)  # Load environment variables from .env file
    api_key = os.getenv("GEMINI_API_KEY")
    # ===== CONFIGURATION =====
    # Get your API key from: https://makersuite.google.com/app/apikey
    API_KEY = api_key  # Replace with your actual API key
    if not API_KEY:
        print("ERROR: Please set your GEMINI_API_KEY environment variable!")
        print("Get it from: https://makersuite.google.com/app/apikey")
    # Dataset sizes
    NUM_PARALLEL_PAIRS = 1000      # Number of (misspelled, correct) pairs
    NUM_LM_SENTENCES = 2000        # Number of correct sentences for language model
    BATCH_SIZE = 100               # Pairs/sentences per API call
    DELAY_SECONDS = 2              # Delay between API calls
    
    # Output files
    PARALLEL_OUTPUT = "parallel_corpus.txt"
    LM_OUTPUT = "lm_corpus.txt"
    # =========================
    
    # Validate API key
    if API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("ERROR: Please set your Gemini API key!")
        print("Get it from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize generator
    generator = GeminiDatasetGenerator(API_KEY)
    
    # Generate parallel corpus (for translation model)
    print("\n🔄 Starting parallel corpus generation...\n")
    generator.generate_dataset(
        total_pairs=NUM_PARALLEL_PAIRS,
        batch_size=BATCH_SIZE,
        output_file=PARALLEL_OUTPUT,
        delay=DELAY_SECONDS
    )
    
    print("\n" + "="*60 + "\n")
    
    # Generate language model corpus
    print("🔄 Starting language model corpus generation...\n")
    generator.generate_lm_corpus(
        total_sentences=NUM_LM_SENTENCES,
        batch_size=BATCH_SIZE,
        output_file=LM_OUTPUT,
        delay=DELAY_SECONDS
    )
    
    print("\n" + "="*60)
    print("🎉 ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  1. {PARALLEL_OUTPUT} - Parallel corpus for translation model")
    print(f"  2. {LM_OUTPUT} - Corpus for language model")
    print("\nYou can now use these with your SMT spelling corrector!")
    print("\nExample usage:")
    print("  parallel_corpus, lm_corpus = load_dataset(")
    print(f"      '{PARALLEL_OUTPUT}', '{LM_OUTPUT}')")


if __name__ == "__main__":
    main()