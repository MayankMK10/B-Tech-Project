"""
language_model.py
=================
Statistical & Neural Language Models for Morse Sentence Reconstruction
Deep Learning–Enhanced Probabilistic Morse Code Decoding

Provides
--------
  1. UnigramLM        — unigram word probabilities (Brown corpus or built-in fallback)
  2. BigramLM         — bigram language model with Laplace smoothing
  3. NeuralCharLM     — character-level neural LM (2-layer LSTM in NumPy)
  4. segment_text     — dynamic-programming word segmenter using any LM
  5. compare_lm       — side-by-side comparison of all LMs
"""

import math
import re
import numpy as np
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# Fallback vocabulary (used when NLTK / Brown
# corpus is not available in the environment)
# ─────────────────────────────────────────────
_FALLBACK_WORDS = (
    "THE OF AND A IN TO IS THAT IT WAS FOR ON ARE AS WITH HIS THEY AT BE FROM "
    "OR THIS HAD BY HOT WORD BUT WHAT SOME WE CAN OUT OTHER WERE ALL YOUR WHEN "
    "UP USE HOW SAID AN EACH SHE WHICH DO THEIR TIME IF WILL WAY ABOUT MANY "
    "THEN THEM WRITE WOULD LIKE SO THESE HER LONG MAKE THING SEE HIM TWO HAS "
    "LOOK MORE DAY COULD GO COME DID NUMBER SOUND NO MOST PEOPLE MY OVER KNOW "
    "WATER THAN CALL FIRST WHO MAY DOWN SIDE BEEN NOW FIND ANY NEW WORK PART "
    "TAKE GET PLACE MADE LIVE WHERE AFTER BACK LITTLE ONLY ROUND MAN YEAR CAME "
    "SHOW EVERY GOOD ME GIVE OUR UNDER NAME VERY THROUGH JUST FORM SENTENCE "
    "GREAT THINK SAY HELP LOW LINE DIFFER TURN CAUSE MUCH MEAN BEFORE MOVE "
    "RIGHT BOY OLD TOO SAME TELL DOES SET THREE WANT AIR WELL ALSO PLAY SMALL "
    "END PUT HOME READ HAND PORT LARGE SPELL ADD EVEN LAND HERE MUST BIG HIGH "
    "SUCH FOLLOW ACT WHY ASK MEN CHANGE WENT LIGHT KIND OFF NEED HOUSE PICTURE "
    "TRY US AGAIN ANIMAL POINT MOTHER WORLD NEAR BUILD SELF EARTH FATHER HEAD "
    "STAND OWN PAGE SHOULD COUNTRY FOUND ANSWER SCHOOL GROW STUDY STILL LEARN "
    "PLANT COVER FOOD SUN FOUR BETWEEN STATE KEEP EYE NEVER LAST LET THOUGHT "
    "CITY TREE CROSS FARM HARD START MIGHT STORY SAW FAR SEA DRAW LEFT LATE "
    "RUN WHILE PRESS CLOSE NIGHT REAL LIFE FEW NORTH OPEN SEEM TOGETHER NEXT "
    "WHITE CHILDREN BEGIN GOT WALK EXAMPLE EASE PAPER OFTEN ALWAYS MUSIC THOSE "
    "BOTH MARK BOOK LETTER UNTIL MILE RIVER CAR FEET CARE SECOND ENOUGH PLAIN "
    "GIRL USUAL YOUNG READY ABOVE EVER RED LIST THOUGH FEEL TALK BIRD SOON "
    "BODY DOG FAMILY DIRECT POSE LEAVE SONG MEASURE DOOR PRODUCT BLACK SHORT "
    "NUMERAL CLASS WIND QUESTION HAPPEN COMPLETE SHIP AREA HALF ROCK ORDER FIRE "
    "SOUTH PROBLEM PIECE TOLD KNEW PASS SINCE TOP WHOLE KING SPACE HEARD BEST "
    "HOUR BETTER TRUE DURING HUNDRED FIVE REMEMBER STEP EARLY HOLD WEST GROUND "
    "INTEREST REACH FAST VERB SING LISTEN SIX TABLE TRAVEL LESS MORNING TEN "
    "SIMPLE SEVERAL VOWEL TOWARD WAR LAY AGAINST PATTERN SLOW CENTER LOVE PERSON "
    "MONEY SERVE APPEAR ROAD MAP RAIN RULE GOVERN PULL COLD NOTICE VOICE FALL "
    "POWER TOWN FINE DRIVE LEAD CRY DARK MACHINE NOTE WAIT PLAN FIGURE STAR "
    "BOX NOUN FIELD REST ABLE POUND DONE BEAUTY DRIVE STOOD CONTAIN FRONT TEACH "
    "WEEK FINAL GAVE GREEN OH QUICK DEVELOP OCEAN WARM FREE MINUTE STRONG SPECIAL "
    "BEHIND CLEAR TAIL PRODUCE FACT STREET INCH MULTIPLY NOTHING COURSE STAY "
    "WHEEL FULL FORCE BLUE OBJECT DECIDE SURFACE DEEP MOON ISLAND FOOT SYSTEM "
    "BUSY TEST RECORD BOAT COMMON GOLD POSSIBLE PLANE STEADY DRY WONDER LAUGH "
    "THOUSAND AGO RAN CHECK GAME SHAPE EQUATE HOT MISS BROUGHT HEAT SNOW TIRE "
    "BRING YES DISTANT FILL EAST PAINT LANGUAGE AMONG GRAND BALL YET WAVE DROP "
    "HEART PRESENT HEAVY DANCE ENGINE POSITION ARM WIDE SAIL MATERIAL SIZE VARY "
    "SETTLE DESCRIBE PEEK ELECTRON BALL BASIC START BROAD JOY CHOOSE BROUGHT "
    "CENTURY OUTSIDE OFFICE PERHAPS SQUARE DONE"
).split()

_FALLBACK_COUNTS = Counter(_FALLBACK_WORDS * 20)   # repeat for realistic freq


# ══════════════════════════════════════════════
# 1. Unigram Language Model
# ══════════════════════════════════════════════

class UnigramLM:
    """Word unigram model. Falls back to built-in vocabulary if NLTK unavailable."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.total:  int     = 0
        self._load()

    def _load(self):
        try:
            from nltk.corpus import brown
            words = [w.upper() for w in brown.words()]
            self.counts = Counter(words)
            self.total  = sum(self.counts.values())
            print("[UnigramLM] Brown corpus loaded.")
        except Exception:
            self.counts = _FALLBACK_COUNTS
            self.total  = sum(self.counts.values())
            print("[UnigramLM] Using built-in fallback vocabulary.")

    def log_prob(self, word: str) -> float:
        word = word.upper()
        if word in self.counts:
            return math.log(self.counts[word] / self.total)
        if re.match(r'^\d+$', word):
            return math.log(1 / self.total)
        return -10.0 * len(word)

    def prob(self, word: str) -> float:
        return math.exp(self.log_prob(word))


# ══════════════════════════════════════════════
# 2. Bigram Language Model
# ══════════════════════════════════════════════

class BigramLM:
    """
    Word-level bigram LM with Laplace (add-1) smoothing.
    Trained on the same word list as UnigramLM.
    """

    def __init__(self, unigram: UnigramLM):
        self.unigram  = unigram
        self.bigrams: defaultdict = defaultdict(Counter)
        self.vocab_size = len(unigram.counts)
        self._train()

    def _train(self):
        try:
            from nltk.corpus import brown
            words   = ['<s>'] + [w.upper() for w in brown.words()] + ['</s>']
            for w1, w2 in zip(words, words[1:]):
                self.bigrams[w1][w2] += 1
            print("[BigramLM] Trained on Brown corpus.")
        except Exception:
            # Build pseudo-bigrams from fallback list
            words = list(_FALLBACK_WORDS)
            for i in range(len(words) - 1):
                self.bigrams[words[i]][words[i+1]] += 1
            print("[BigramLM] Trained on fallback vocabulary.")

    def log_prob(self, w2: str, w1: str) -> float:
        """Log P(w2 | w1) with Laplace smoothing."""
        w1, w2 = w1.upper(), w2.upper()
        count_w1w2 = self.bigrams[w1][w2]
        count_w1   = sum(self.bigrams[w1].values())
        # Laplace smoothing
        prob = (count_w1w2 + 1) / (count_w1 + self.vocab_size + 1)
        return math.log(prob)

    def score_sentence(self, words: list) -> float:
        """Log-prob of full word sequence."""
        if not words:
            return -math.inf
        score = self.unigram.log_prob(words[0])
        for i in range(1, len(words)):
            score += self.log_prob(words[i], words[i-1])
        return score


# ══════════════════════════════════════════════
# 3. Neural Character Language Model (NumPy LSTM)
# ══════════════════════════════════════════════

class NeuralCharLM:
    """
    Lightweight character-level LSTM language model (one layer, 64 hidden units).
    Pre-trained on the built-in word list using truncated backprop.
    Used for re-scoring candidate sentences.
    """

    CHARS   = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
    H_DIM   = 64
    EPOCHS  = 5
    LR      = 0.005

    def __init__(self, seed: int = 7):
        self.vocab   = {c: i for i, c in enumerate(self.CHARS)}
        self.V       = len(self.CHARS)
        self._init_weights(seed)
        self._train()

    def _init_weights(self, seed):
        rng = np.random.default_rng(seed)
        H, V = self.H_DIM, self.V
        s = lambda n: np.sqrt(1.0 / n)
        # Gates: i, f, g, o  (all in one matrix for efficiency)
        self.Wx = rng.standard_normal((4 * H, V)).astype(np.float32) * s(V)
        self.Wh = rng.standard_normal((4 * H, H)).astype(np.float32) * s(H)
        self.b  = np.zeros((4 * H,), dtype=np.float32)
        # Output projection
        self.Wy = rng.standard_normal((V, H)).astype(np.float32) * s(H)
        self.by = np.zeros(V, dtype=np.float32)

    def _lstm_step(self, x_t, h_prev, c_prev):
        H  = self.H_DIM
        z  = self.Wx @ x_t + self.Wh @ h_prev + self.b
        i  = 1 / (1 + np.exp(-z[:H]))
        f  = 1 / (1 + np.exp(-z[H:2*H]))
        g  = np.tanh(z[2*H:3*H])
        o  = 1 / (1 + np.exp(-z[3*H:4*H]))
        c  = f * c_prev + i * g
        h  = o * np.tanh(c)
        return h, c

    def _forward(self, indices):
        """Return log-probs for each position."""
        H = self.H_DIM
        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        log_probs = []
        for idx in indices:
            x = np.zeros(self.V, dtype=np.float32)
            x[idx] = 1.0
            h, c = self._lstm_step(x, h, c)
            logit = self.Wy @ h + self.by
            logit -= logit.max()
            e    = np.exp(logit)
            prob = e / (e.sum() + 1e-12)
            log_probs.append(np.log(prob + 1e-12))
        return log_probs

    def _train(self):
        """Tiny training loop on built-in vocab sentences."""
        corpus = ' '.join(_FALLBACK_WORDS[:200]) + ' '
        corpus = corpus.upper()
        idxs   = [self.vocab.get(c, self.vocab[' ']) for c in corpus]

        print(f"[NeuralCharLM] Training on {len(idxs)}-char corpus …", end='')
        # Single-pass gradient approximation (very lightweight)
        H = self.H_DIM
        for ep in range(self.EPOCHS):
            h = np.zeros(H, dtype=np.float32)
            c = np.zeros(H, dtype=np.float32)
            total_loss = 0
            for t in range(len(idxs) - 1):
                x = np.zeros(self.V, dtype=np.float32)
                x[idxs[t]] = 1.0
                h, c  = self._lstm_step(x, h, c)
                logit = self.Wy @ h + self.by
                logit -= logit.max()
                e    = np.exp(logit)
                prob = e / (e.sum() + 1e-12)

                target = idxs[t + 1]
                total_loss -= math.log(prob[target] + 1e-12)

                # Output layer gradient only (frozen LSTM for speed)
                d_logit    = prob.copy()
                d_logit[target] -= 1.0
                self.Wy   -= self.LR * np.outer(d_logit, h)
                self.by   -= self.LR * d_logit

        print(" done.")

    def score(self, text: str) -> float:
        """
        Return mean log-probability of *text* under the character LM.
        Higher (less negative) = more likely.
        """
        text = text.upper()
        idxs = [self.vocab.get(c, self.vocab[' ']) for c in text]
        if len(idxs) < 2:
            return -math.inf
        log_probs = self._forward(idxs[:-1])
        return float(np.mean([lp[idxs[i+1]] for i, lp in enumerate(log_probs)]))


# ══════════════════════════════════════════════
# 4. Dynamic-programming word segmenter
# ══════════════════════════════════════════════

def segment_text(text: str,
                 lm: UnigramLM,
                 bigram_lm: BigramLM = None,
                 max_word_len: int = 20) -> tuple:
    """
    Viterbi-style DP segmentation of a continuous character string
    using the supplied language model.

    Returns
    -------
    (words_list, confidence_score)
    """
    text = text.upper()
    n    = len(text)
    NEG_INF = -float('inf')

    # dp[i] = (best_score, [word_list])
    dp = [(NEG_INF, []) for _ in range(n + 1)]
    dp[0] = (0.0, [])

    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            if dp[j][0] == NEG_INF:
                continue
            word = text[j:i]
            if bigram_lm and dp[j][1]:
                prev_word = dp[j][1][-1]
                score     = dp[j][0] + bigram_lm.log_prob(word, prev_word)
            else:
                score     = dp[j][0] + lm.log_prob(word)
            if score > dp[i][0]:
                dp[i] = (score, dp[j][1] + [word])

    best_score = dp[n][0]
    sentence   = dp[n][1]

    # Confidence: compare best to worst single-word parse
    worst = lm.log_prob(text)
    if worst == 0 or best_score == NEG_INF:
        confidence = 0.0
    else:
        confidence = max(0.0, min(1.0, (best_score - worst) / abs(worst)))

    return sentence, confidence


# ══════════════════════════════════════════════
# 5. LM Comparison
# ══════════════════════════════════════════════

def compare_lm(raw_text: str,
               unigram: UnigramLM,
               bigram:  BigramLM,
               neural:  NeuralCharLM) -> dict:
    """
    Run all three language models on *raw_text* and return results dict.
    """
    results = {}

    seg_uni, conf_uni = segment_text(raw_text, unigram)
    results['unigram'] = {
        'sentence':   ' '.join(seg_uni),
        'confidence': conf_uni,
        'score':      sum(unigram.log_prob(w) for w in seg_uni),
    }

    seg_bi, conf_bi = segment_text(raw_text, unigram, bigram)
    results['bigram'] = {
        'sentence':   ' '.join(seg_bi),
        'confidence': conf_bi,
        'score':      bigram.score_sentence(seg_bi),
    }

    # Neural re-scoring of the two candidates
    candidates = [' '.join(seg_uni), ' '.join(seg_bi)]
    neural_scores = [neural.score(c) for c in candidates]
    best_cand = candidates[int(np.argmax(neural_scores))]
    results['neural_reranked'] = {
        'sentence':     best_cand,
        'neural_score': max(neural_scores),
    }

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    sample = 'HELLOWORLDTHISISMORSETEST'
    print(f"Input text : {sample}\n")

    uni    = UnigramLM()
    bi     = BigramLM(uni)
    neural = NeuralCharLM()

    results = compare_lm(sample, uni, bi, neural)

    for name, res in results.items():
        print(f"[{name.upper()}]")
        print(f"  Sentence : {res.get('sentence','')}")
        print(f"  Score    : {res.get('score', res.get('neural_score', '—')):.4f}")
        if 'confidence' in res:
            print(f"  Confidence: {res['confidence']*100:.1f}%")
        print()
