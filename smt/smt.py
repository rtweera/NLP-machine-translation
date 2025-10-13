import re
import math
from collections import defaultdict, Counter
import numpy as np

class SMTSpellingCorrector:
    """
    Statistical Machine Translation approach to spelling correction.
    Treats spelling correction as translation from misspelled to correct English.
    """
    
    def __init__(self, n=2):
        """
        Initialize the SMT spelling corrector.
        
        Args:
            n: n-gram size for language model (default: 2 for bigrams)
        """
        self.n = n
        # Translation model: P(misspelled|correct)
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        # Language model: P(word|context)
        self.language_model = defaultdict(lambda: defaultdict(float))
        self.unigram_counts = Counter()
        self.vocabulary = set()
        
    def train_translation_model(self, parallel_corpus):
        """
        Train translation model from parallel corpus of (misspelled, correct) pairs.
        
        Args:
            parallel_corpus: List of tuples [(misspelled_sentence, correct_sentence), ...]
        """
        print("Training translation model...")
        
        # Count alignments
        alignment_counts = defaultdict(lambda: defaultdict(int))
        correct_word_counts = defaultdict(int)
        
        for misspelled_sent, correct_sent in parallel_corpus:
            misspelled_words = misspelled_sent.lower().split()
            correct_words = correct_sent.lower().split()
            
            # Simple word alignment (assumes similar word order)
            for m_word in misspelled_words:
                for c_word in correct_words:
                    # Increment alignment count based on edit distance
                    if self._should_align(m_word, c_word):
                        alignment_counts[c_word][m_word] += 1
                        correct_word_counts[c_word] += 1
                        self.vocabulary.add(c_word)
        
        # Calculate translation probabilities
        for c_word, m_words in alignment_counts.items():
            for m_word, count in m_words.items():
                # P(misspelled|correct)
                self.translation_probs[c_word][m_word] = count / correct_word_counts[c_word]
        
        print(f"Translation model trained with {len(self.vocabulary)} vocabulary words")
    
    def _should_align(self, word1, word2, threshold=3):
        """
        Determine if two words should be aligned based on edit distance.
        """
        if word1 == word2:
            return True
        
        edit_dist = self._edit_distance(word1, word2)
        # Align if edit distance is small relative to word length
        return edit_dist <= threshold
    
    def _edit_distance(self, s1, s2):
        """
        Calculate Levenshtein edit distance between two strings.
        """
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def train_language_model(self, corpus):
        """
        Train n-gram language model from correct English corpus.
        
        Args:
            corpus: List of correct English sentences
        """
        print(f"Training {self.n}-gram language model...")
        
        ngram_counts = defaultdict(lambda: defaultdict(int))
        context_counts = defaultdict(int)
        
        for sentence in corpus:
            words = ['<s>'] * (self.n - 1) + sentence.lower().split() + ['</s>']
            self.vocabulary.update(words)
            
            # Count unigrams
            for word in words:
                self.unigram_counts[word] += 1
            
            # Count n-grams
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i + self.n - 1])
                word = words[i + self.n - 1]
                ngram_counts[context][word] += 1
                context_counts[context] += 1
        
        # Calculate probabilities with add-k smoothing
        k = 0.1  # Smoothing parameter
        vocab_size = len(self.vocabulary)
        
        for context, word_counts in ngram_counts.items():
            context_total = context_counts[context]
            for word, count in word_counts.items():
                # Add-k smoothing
                self.language_model[context][word] = \
                    (count + k) / (context_total + k * vocab_size)
        
        print(f"Language model trained on {len(corpus)} sentences")
    
    def get_language_model_prob(self, context, word):
        """
        Get probability of word given context from language model.
        """
        context = tuple(context)
        if context in self.language_model and word in self.language_model[context]:
            return self.language_model[context][word]
        else:
            # Backoff to unigram probability
            total_words = sum(self.unigram_counts.values())
            return (self.unigram_counts[word] + 0.1) / (total_words + 0.1 * len(self.vocabulary))
    
    def get_translation_prob(self, correct_word, misspelled_word):
        """
        Get P(misspelled|correct) from translation model.
        """
        if correct_word in self.translation_probs:
            if misspelled_word in self.translation_probs[correct_word]:
                return self.translation_probs[correct_word][misspelled_word]
        
        # Fallback: use edit distance-based probability
        edit_dist = self._edit_distance(correct_word, misspelled_word)
        if edit_dist == 0:
            return 1.0
        else:
            return 1.0 / (1.0 + edit_dist * 2)
    
    def generate_candidates(self, word, max_edit_distance=2):
        """
        Generate candidate corrections for a word based on edit distance.
        """
        candidates = set()
        
        # If word is already in vocabulary, it's a candidate
        if word.lower() in self.vocabulary:
            candidates.add(word.lower())
        
        # Generate words within edit distance
        for vocab_word in self.vocabulary:
            if vocab_word in ['<s>', '</s>']:
                continue
            if self._edit_distance(word.lower(), vocab_word) <= max_edit_distance:
                candidates.add(vocab_word)
        
        return candidates if candidates else {word.lower()}
    
    def correct_word(self, word, context):
        """
        Correct a single word using SMT approach: P(correct|misspelled) ∝ P(misspelled|correct) * P(correct|context)
        
        Args:
            word: Word to correct
            context: List of previous words (for language model)
        
        Returns:
            Best correction for the word
        """
        candidates = self.generate_candidates(word)
        
        if not candidates:
            return word
        
        best_word = word.lower()
        best_score = float('-inf')
        
        for candidate in candidates:
            # P(misspelled|correct) - translation model
            trans_prob = self.get_translation_prob(candidate, word.lower())
            
            # P(correct|context) - language model
            lm_prob = self.get_language_model_prob(context, candidate)
            
            # Combined score (log probabilities to avoid underflow)
            score = math.log(trans_prob + 1e-10) + math.log(lm_prob + 1e-10)
            
            if score > best_score:
                best_score = score
                best_word = candidate
        
        return best_word
    
    def correct_sentence(self, sentence):
        """
        Correct spelling errors in a sentence.
        
        Args:
            sentence: Input sentence with potential spelling errors
        
        Returns:
            Corrected sentence
        """
        words = sentence.split()
        corrected_words = []
        context = ['<s>'] * (self.n - 1)
        
        for word in words:
            # Preserve punctuation
            match = re.match(r"(\w+)(.*)", word)
            if match:
                word_part = match.group(1)
                punct_part = match.group(2)
                
                corrected_word = self.correct_word(word_part, context[-(self.n-1):])
                corrected_words.append(corrected_word + punct_part)
                
                # Update context
                context.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)


# Example usage and training
def create_sample_data():
    """
    Create sample training data.
    For a real system, you'd use a much larger corpus.
    """
    # Parallel corpus: (misspelled, correct) pairs
    parallel_corpus = [
        ("I hav a dreem", "I have a dream"),
        ("The qick brown fox", "The quick brown fox"),
        ("She is verry happi", "She is very happy"),
        ("He went to the stor", "He went to the store"),
        ("This is realy grate", "This is really great"),
        ("I can spel wel", "I can spell well"),
        ("The weater is nise", "The weather is nice"),
        ("I lov programing", "I love programming"),
        ("Lets meat tomorrow", "Lets meet tomorrow"),
        ("The beutiful scenry", "The beautiful scenery"),
        ("I recieved a mesage", "I received a message"),
        ("She is definately coming", "She is definitely coming"),
        ("The goverment anounced", "The government announced"),
        ("He made a mistke", "He made a mistake"),
        ("This is importent information", "This is important information"),
        ("I beleive in you", "I believe in you"),
        ("We achived our goal", "We achieved our goal"),
        ("The experiance was amasing", "The experience was amazing"),
        ("I apreciate your help", "I appreciate your help"),
        ("This is a beutiful day", "This is a beautiful day"),
    ]
    
    # Language model corpus (correct English)
    lm_corpus = [
        "I have a dream",
        "The quick brown fox jumps over the lazy dog",
        "She is very happy today",
        "He went to the store yesterday",
        "This is really great news",
        "I can spell well",
        "The weather is nice and sunny",
        "I love programming in Python",
        "Let us meet tomorrow at noon",
        "The beautiful scenery took my breath away",
        "I received a message from my friend",
        "She is definitely coming to the party",
        "The government announced new policies",
        "He made a mistake but learned from it",
        "This is important information for everyone",
        "I believe in your abilities",
        "We achieved our goal through hard work",
        "The experience was amazing and unforgettable",
        "I appreciate your help very much",
        "This is a beautiful day for a walk",
        "The quick brown fox is very fast",
        "She is happy and excited",
        "I love to program every day",
        "The weather is beautiful today",
        "This is really important",
    ]
    
    return parallel_corpus, lm_corpus


def main():
    """
    Main function demonstrating the SMT spelling corrector.
    """
    print("=" * 60)
    print("SMT-based Spelling Corrector for English")
    print("=" * 60)
    print()
    
    # Create sample data
    parallel_corpus, lm_corpus = create_sample_data()
    
    # Initialize and train the corrector
    corrector = SMTSpellingCorrector(n=2)
    
    # Train translation model
    corrector.train_translation_model(parallel_corpus)
    print()
    
    # Train language model
    corrector.train_language_model(lm_corpus)
    print()
    
    # Test sentences
    test_sentences = [
        "I hav a dreem today",
        "The qick brown fox jumps",
        "She is verry happi",
        "This is realy importent",
        "I beleive we can achive it",
        "The weater is beutiful",
        "I recieved the mesage yesterday",
        "He definately made a mistke",
        "Lets meat at the stor",
    ]
    
    print("=" * 60)
    print("Spelling Correction Results:")
    print("=" * 60)
    
    for sentence in test_sentences:
        corrected = corrector.correct_sentence(sentence)
        print(f"\nOriginal:  {sentence}")
        print(f"Corrected: {corrected}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()