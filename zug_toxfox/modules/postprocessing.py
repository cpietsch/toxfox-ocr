# type: ignore
import os
import re
import warnings

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from zug_toxfox import default_config, getLogger, pipeline_config
from zug_toxfox.utils import load_json, process_pollutants, remove_duplicates

log = getLogger(__name__)
warnings.filterwarnings("ignore")

config = pipeline_config.postprocessing

# Marker that introduces an INCI list on a label ("Ingredients:", "Inhaltsstoffe:", and a few
# EU-language variants). The leading-letter class tolerates the common OCR confusion of I/l/1/|.
_INGREDIENT_MARKER = re.compile(r"(?i)(?:[il1|!]ngredient|ingr[ée]dient|inhaltsstoff|ingredien[tz])")
# Strips a line up to and including the marker (+ optional colon), leaving the list remainder.
_MARKER_STRIP = re.compile(
    r"(?i)^.*?(?:[il1|!]ngredients?|ingr[ée]dients?|inhaltsstoffe?|ingredien[tz][a-z]*)\s*[:.\-]?\s*"
)


def _is_subseq(small: list[str], big: list[str]) -> bool:
    """True if `small` appears as a contiguous run of whole words inside `big`."""
    n, m = len(small), len(big)
    if n == 0 or n >= m:
        return False
    return any(big[i:i + n] == small for i in range(m - n + 1))


class FAISSIndexer:
    def __init__(self):
        # Allow swapping the embedding model via the EMBED_MODEL env var without editing
        # config, so a benchmark experiment is a one-line change. Falls back to config.
        self.model_name = os.environ.get("EMBED_MODEL") or config.FAISSIndexer_model_name
        self.model = SentenceTransformer(self.model_name)
        # A cached FAISS index is embedding-specific (different model -> different dim and
        # vectors), so namespace the index files per model. The configured default keeps its
        # legacy filename, so the prebuilt index ships and loads unchanged.
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_name).strip("-").lower()
        self.index_suffix = "" if self.model_name == config.FAISSIndexer_model_name else f"__{slug}"
        self.indices = {}
        self.key_tokens = {}
        self.use_gpu = faiss.get_num_gpus() > 0
        self.res = faiss.StandardGpuResources() if self.use_gpu else None

    def build_index(self, tokens: list[str], index_path: str) -> faiss.Index:
        if os.path.exists(index_path):
            log.info("Loading existing index...")
            index = faiss.read_index(index_path)
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(self.res, 0, index)
            return index

        log.info("Building new index...")
        embeddings = self.model.encode(tokens, convert_to_numpy=True, show_progress_bar=True)

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])

        if self.use_gpu:
            index = faiss.index_cpu_to_gpu(self.res, 0, index)

        index.add(embeddings)

        log.info("Saving index to %s...", index_path)
        faiss.write_index(faiss.index_gpu_to_cpu(index) if self.use_gpu else index, index_path)
        return index

    def add_index(self, name: str, tokens: list[str], index_path: str) -> None:
        self.indices[name] = self.build_index(tokens, index_path)
        self.key_tokens[name] = tokens

    def search(self, name: str, queries: list[str], threshold: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        index = self.indices[name]
        distances, indices = index.search(query_embeddings, 1)
        result_tokens = np.array(self.key_tokens[name])[indices.flatten()]
        mask = distances.flatten() >= threshold
        return result_tokens, mask, distances

    def rapidfuzz_search(
        self, name: str, queries: list[str], threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lexical nearest-neighbour over the same vocab, drop-in for ``search``.

        INCI matching is an orthographic problem (OCR character errors on a fixed Latin
        vocabulary), not a semantic one, so character-aware fuzzy similarity is a better fit
        than dense cosine and directly optimizes the edit-distance metric the eval scores on.
        ``threshold`` is on rapidfuzz's 0-100 scale (not cosine).
        """
        from rapidfuzz import fuzz, process

        choices = self.key_tokens[name]
        # WRatio blends ratio/partial/token-set, robust to length and word-order differences.
        scores_matrix = process.cdist(queries, choices, scorer=fuzz.WRatio, workers=-1)
        best_idx = scores_matrix.argmax(axis=1)
        best_scores = scores_matrix[np.arange(len(queries)), best_idx]
        result_tokens = np.array(choices, dtype=object)[best_idx]
        mask = best_scores >= threshold
        return result_tokens, mask, best_scores


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()
        self.indexer = FAISSIndexer

    def insert(self, word: str) -> None:
        """Inserts a word into the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def non_matching(
        self, results: list[str], mask: list[bool], word: str, is_delimiter: str, i: int
    ) -> tuple[int, bool]:
        """Handles cases where no match in found in the Trie. Non-matching tokens are appended as a single string up to
        the next matching ingredient."""
        if ";" in word:
            cleaned_word = re.sub(r"[;]", "", word)
            if is_delimiter or not results:
                results.append(cleaned_word)
                mask.append(False)
            else:
                results[-1] += cleaned_word
            i += len(word) + 1
            is_delimiter = True
        elif is_delimiter:
            results.append(word)
            mask.append(False)
            i += len(word)
            is_delimiter = False
        elif not results or mask[-1]:
            word_split = word.split()
            if word_split:
                results.append(word_split[0])
                mask.append(False)
                i += len(word_split[0])
            else:
                i += 1
        else:
            results[-1] += word
            i += len(word)
        return i, is_delimiter

    def search(self, text: str) -> tuple[list[str], list[bool]]:
        """Searches for matching ingredients in the Trie."""
        results: list[str] = []
        mask: list[bool] = []
        i: int = 0
        is_delimiter = False
        while i < len(text):
            node: TrieNode = self.root
            word: str = ""
            longest_match: str = ""
            for n, char in enumerate(text[i:]):
                word += char
                if char in node.children:
                    node = node.children[char]
                    if node.is_end_of_word:
                        longest_match = word
                        if n == len(text[i:]) - 1:
                            results.append(longest_match.strip())
                            mask.append(True)
                            i += len(longest_match)
                            break
                else:
                    if longest_match:
                        results.append(longest_match.strip())
                        mask.append(True)
                        i += len(longest_match.strip()) + 1
                        if char == ";":
                            is_delimiter = False
                            i += 1
                    else:
                        i, is_delimiter = self.non_matching(results, mask, word, is_delimiter, i)
                    break

                if n == len(text[i:]) - 1:
                    i += 1

        # dtype=bool matters: an empty match list otherwise yields a float64 array, which
        # blows up when used as a boolean index in get_ingredients (engines other than EasyOCR
        # can emit tokens that wholly miss the Trie). Keep results as object for the same reason.
        return np.array([r.strip() for r in results], dtype=object), np.array(mask, dtype=bool)


class TokenCleaner:
    def __init__(self, indexer):
        self.indexer = indexer
        self.typo_threshold = config.typo_threshold
        self.misspelling_set = config.misspelling_set

        # Word-level typo correction backend: "faiss" (semantic) or "symspell" (Symmetric-Delete
        # edit distance over the fixed INCI word vocab). Edit distance is the textbook fit for OCR
        # character errors on a closed vocabulary. Precedence: TYPO_BACKEND env > config > "faiss".
        _tb = config.typo_backend
        self.typo_backend = (os.environ.get("TYPO_BACKEND") or (_tb if isinstance(_tb, str) else "faiss")).lower()
        self._symspell = self._build_symspell() if self.typo_backend == "symspell" else None

    def _build_symspell(self):
        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for word in self.indexer.key_tokens["words"]:
            sym.create_dictionary_entry(word, 1)  # uniform freq: fixed vocab, no corpus stats
        return sym

    def _symspell_correct(self, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from symspellpy import Verbosity

        corrected: list[str] = []
        mask: list[bool] = []
        for q in queries:
            ql = q.lower()
            # Guard: don't edit-correct very short tokens (over-corrects to wrong INCI words).
            if len(ql) < 5:
                corrected.append(q)
                mask.append(False)
                continue
            sug = self._symspell.lookup(ql, Verbosity.TOP, max_edit_distance=2)
            if sug:
                corrected.append(sug[0].term)
                mask.append(True)
            else:
                corrected.append(q)
                mask.append(False)
        return np.array(corrected, dtype=object), np.array(mask, dtype=bool)

    def clean_token(self, tokens: list[str]) -> list[str]:
        tokens = self.split_colon(tokens)
        cleaned_tokens = self.hyphen_and_parentheses(tokens)
        cleaned_tokens = [self.clean_word(word) for word in cleaned_tokens]
        return " ".join([token for token in cleaned_tokens for token in token.split() if len(token) > 1])

    def split_colon(self, tokens: list[str]) -> list[str]:
        words = []
        for token in tokens:
            parts = token.split(":")
            words.extend([part.strip() + (":" if i < len(parts) - 1 else "") for i, part in enumerate(parts)])
        return words

    def clean_word(self, word: str) -> str:
        cleaned_word = re.sub(r"[^\w\s():\\/]", "", word).strip()
        cleaned_word = re.sub(r"[^\w\s():\\/\-,;.]", "", word).strip()
        return self.correct_typos(cleaned_word.lower()) if cleaned_word else ""

    def hyphen_and_parentheses(self, split_token: list[str]) -> list[str]:
        length = len(split_token)

        cleaned_split_token = []
        skip_next = False

        for n in range(length):
            if skip_next:
                skip_next = False
                continue
            if split_token[n].endswith("-") and n + 1 < length:
                cleaned_split_token.append(split_token[n][:-1] + split_token[n + 1])
                skip_next = True
            elif (
                "(" in split_token[n]
                and ")" not in split_token[n]
                and n + 1 < length
                and ")" in split_token[n + 1]
                and "(" not in split_token[n + 1]
            ):
                cleaned_split_token.append(split_token[n] + " " + split_token[n + 1])
                skip_next = True
            else:
                cleaned_split_token.append(split_token[n])
                skip_next = False
        return cleaned_split_token

    def correct_typos(self, token: str) -> str:
        """Correct typos and misspelling of the OCR output using embedding search."""
        token = re.sub(r"\s+([,;.])", r"\1", token)

        queries = token.split()
        delimiter_mask = []
        delimiter_queries = []

        for q in queries:
            if any(delim in q for delim in [",", ";", "."]):
                delimiter_mask.append(True)
                q = re.sub(r"[,;.]", "", q)
            else:
                delimiter_mask.append(False)
            delimiter_queries.append(q)

        if self.typo_backend == "symspell":
            corrected_tokens, mask = self._symspell_correct(delimiter_queries)
            delimiter_queries = np.array(delimiter_queries, dtype=object)
            delimiter_queries[mask] = corrected_tokens[mask]
        else:
            corrected_tokens, mask, _ = self.indexer.search("words", delimiter_queries, self.typo_threshold)
            delimiter_queries = np.array(delimiter_queries)
            delimiter_queries[mask] = corrected_tokens[mask].tolist()

        corrected_queries = [
            query if query.lower() not in self.misspelling_set else "oil" for query in delimiter_queries
        ]  # Correct typically misspelled word 'Oil'

        corrected_queries_with_delimiters = []
        for query, has_delimiter in zip(corrected_queries, delimiter_mask):  # noqa: B905
            if has_delimiter:
                query += ";"
            corrected_queries_with_delimiters.append(query)

        return " ".join(corrected_queries_with_delimiters)


class PostProcessor:
    def __init__(
        # TODO: Add type hints
        self,
        indexer: FAISSIndexer,
    ):
        self.ingredient_threshold = config.ingredient_threshold
        self.typo_threshold = config.typo_threshold
        self.misspelling_set = set(config.misspelling_set)
        self.detection_type = config.detection_type

        # Ingredient matcher backend: "faiss" (dense cosine) or "rapidfuzz" (lexical).
        # Precedence: MATCH_BACKEND env > config.match_backend > "faiss". Toggled for A/B.
        _mb = config.match_backend
        self.match_backend = (os.environ.get("MATCH_BACKEND") or (_mb if isinstance(_mb, str) else "faiss")).lower()
        self.rapidfuzz_threshold = float(os.environ.get("RAPIDFUZZ_THRESHOLD", "85"))

        # Isolate the ingredient region at the "Ingredients:" marker (drops surrounding
        # marketing/usage text on real product photos). Precedence: ISOLATE_REGION env > config > on.
        _ir = config.isolate_region
        _env_ir = os.environ.get("ISOLATE_REGION")
        self.isolate_region = (
            _env_ir.lower() not in ("0", "false", "no", "off") if _env_ir is not None
            else (_ir if isinstance(_ir, bool) else True)
        )

        # Ingredient extraction strategy: "trie" (default) | "segment". Precedence: env > config.
        _ms = config.match_strategy
        self.match_strategy = (os.environ.get("MATCH_STRATEGY") or (_ms if isinstance(_ms, str) else "trie")).lower()
        _st = config.segment_threshold
        self.segment_threshold = float(os.environ.get("SEGMENT_THRESHOLD", _st if isinstance(_st, (int, float)) else 90))
        _wt = config.window_threshold
        self.window_threshold = float(os.environ.get("WINDOW_THRESHOLD", _wt if isinstance(_wt, (int, float)) else 90))
        # 'auto' routes to the segment matcher only when at least this many lines were dropped
        # before the marker (i.e. a real full-label photo, not a bare cropped panel with a header).
        _smd = config.segment_min_dropped
        self.segment_min_dropped = int(os.environ.get("SEGMENT_MIN_DROPPED", _smd if isinstance(_smd, (int, float)) else 5))
        # Drop predicted ingredients that are whole-word sub-phrases of another prediction (the
        # Trie's greedy short-fragment FPs). Precedence: DROP_FRAGMENTS env > config > on.
        _df = config.drop_fragments
        _env_df = os.environ.get("DROP_FRAGMENTS")
        self.drop_fragments = (
            _env_df.lower() not in ("0", "false", "no", "off") if _env_df is not None
            else (_df if isinstance(_df, bool) else True)
        )

        self.faiss_path = default_config.faiss_path
        self.pollutants_path = default_config.pollutants_path_simple
        self.inci_path = default_config.inci_path_simple
        self.synonym_path = default_config.synonym_path

        process_pollutants()

        with open(self.synonym_path) as file:
            self.synonym_to_ingredient = yaml.safe_load(file)

        self.indexer = indexer
        self.trie = Trie()

        self.known_ingredients = load_json(self.inci_path)

        if self.detection_type in ["pollutants", "both"]:
            self.pollutants = load_json(self.pollutants_path)

        self.combined_tokens = [ing.lower() for ing in self.known_ingredients]
        if self.detection_type in ["pollutants", "both"]:
            self.combined_tokens += remove_duplicates([pol.lower() for pol in self.pollutants])
        self.known_words = remove_duplicates([t.lower() for token in self.combined_tokens for t in token.split()])
        # Space-stripped vocab (index-aligned with combined_tokens) for the segment matcher's
        # despaced fallback, which recovers space-dropping/fragmenting OCR reads ('ALCOHOLDENAT',
        # 'PRUNUS AMYGDALUS DUL CIS OIL').
        self._combined_tokens_ns = [re.sub(r"[^a-z0-9]", "", t) for t in self.combined_tokens]
        # CI colour-index codes present in the vocab, normalised to 'ci NNNNN', for direct matching.
        self._ci_codes = {re.sub(r"\s+", " ", t) for t in self.combined_tokens if re.fullmatch(r"ci\s*\d{4,5}", t)}
        # Despaced -> canonical name map for the window matcher's O(1) exact fast path (clean
        # fragmented reads like 'prunus amygdalus dul cis oil' are exact once spaces are removed).
        self._ns_to_name = {}
        for t, tns in zip(self.combined_tokens, self._combined_tokens_ns):
            if len(tns) >= 6:
                self._ns_to_name.setdefault(tns, t)

        os.makedirs(self.faiss_path, exist_ok=True)
        suffix = self.indexer.index_suffix
        self.indexer.add_index("tokens", self.combined_tokens, os.path.join(self.faiss_path, f"faiss_tokens{suffix}"))
        self.indexer.add_index("words", self.known_words, os.path.join(self.faiss_path, f"faiss_words{suffix}"))

        for ingredient in self.combined_tokens:
            self.trie.insert(ingredient)

        self.token_cleaner = TokenCleaner(indexer=self.indexer)

    def remove_redundancy(self, ingredients: list[str]) -> list[str]:
        corrected_ingredients = [ingredients[0]]
        for i in range(1, len(ingredients)):
            if not (
                i + 1 < len(ingredients)
                and ingredients[i] in ingredients[i - 1].split()
                and len(ingredients[i - 1].split()) > 1
            ):
                corrected_ingredients.append(ingredients[i])
        return corrected_ingredients

    def find_pollutants(self, synonym):
        return self.synonym_to_ingredient.get(synonym.strip(), None)

    def _isolate_ingredient_region(self, tokens: list[str]) -> list[str]:
        """Keep only OCR lines from the 'Ingredients:'/'Inhaltsstoffe:' marker onward.

        Real product photos surround the ingredient panel with marketing/usage text (e.g. a
        'free of ... Octocrylene' claim, a 'Pro-Melanin extract' slogan) whose words get matched
        to INCI names -> dangerous false positives. The INCI list is conventionally introduced by
        an 'Ingredients:' marker, so drop everything before it. If no marker is found (the curated
        eval crops are bare panels), return the tokens unchanged so marker-less inputs are unaffected.
        """
        for i, line in enumerate(tokens):
            if _INGREDIENT_MARKER.search(line):
                remainder = _MARKER_STRIP.sub("", line, count=1).strip()
                rest = list(tokens[i + 1:])
                return (([remainder] + rest) if remainder else rest), i  # i = lines dropped before marker
        return list(tokens), -1

    def _compute_pollutants(self, ingredients: list[str]) -> list[str]:
        if self.detection_type in ["pollutants", "both"] and len(ingredients) > 0:
            return remove_duplicates(
                [self.find_pollutants(ing) for ing in ingredients if self.find_pollutants(ing) is not None]
            )
        return []

    def _segment_get_ingredients(self, tokens: list[str]) -> dict:
        """Comma-split the (isolated) region and fuzzy-match each whole segment to an INCI name.

        OCR garbles multi-word names on curved labels ('Tcopheryl Acetate', 'Capernicia Cerilera
        Cero'), which the Trie's longest-prefix match drops or reduces to a wrong fragment. Matching
        the whole comma-delimited segment with rapidfuzz (char-aware, length-robust) recovers them.
        Lines are joined first because the OCR wraps the list mid-name across lines.
        """
        from rapidfuzz import fuzz, process

        blob = " ".join(tokens)
        matched: list[str] = []
        for seg in re.split(r"[,;•·]", blob):
            seg_l = seg.lower()
            # CI colour-index codes (ci 77491, ci 19140 ...) ARE valid INCI ingredients and appear in
            # ground truth, so match them rather than discarding them. Tolerate the OCR c/l confusion.
            for code in re.findall(r"\b[cс][il1|]\s*(\d{4,5})\b", seg_l):
                name = f"ci {code}"
                if name in self._ci_codes:
                    matched.append(name)
            s = re.sub(r"\b[cс][il1|]\s*\d{4,5}\b", " ", seg_l)  # remove the code, match the rest
            s = re.sub(r"[^a-z0-9/+\- ]", " ", s)  # keep INCI-ish characters only
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) < 3:
                continue
            # token_sort_ratio compares whole strings (no substring shortcut), so a long garbled
            # segment can't spuriously match a tiny INCI name (imidazole/phenol). The length guard
            # rejects matches far shorter than the segment (another fragment-FP source).
            m = process.extractOne(s, self.combined_tokens, scorer=fuzz.token_sort_ratio)
            if m and m[1] >= self.segment_threshold and len(m[0]) >= 0.6 * len(s):
                matched.append(m[0])
                continue
            # Despaced fallback: OCR fragments multi-word names ('PRUNUS AMYGDALUS DUL CIS OIL') or
            # drops the spaces entirely ('ALCOHOLDENAT'); token_sort can't align either. Comparing the
            # space-stripped segment to the space-stripped vocab handles BOTH (it is near-exact once
            # spaces are removed). High ratio + tight length ratio keep precision without the
            # word-order slack, so the space-count gate is unnecessary.
            s_ns = s.replace(" ", "")
            if len(s_ns) >= 7:
                m2 = process.extractOne(s_ns, self._combined_tokens_ns, scorer=fuzz.ratio)
                if m2 and m2[1] >= max(self.segment_threshold, 88):
                    cand_ns = self._combined_tokens_ns[m2[2]]
                    if 0.82 <= len(cand_ns) / len(s_ns) <= 1.22:
                        matched.append(self.combined_tokens[m2[2]])
        ingredients = remove_duplicates(matched)
        return {"ingredients": ingredients, "pollutants": self._compute_pollutants(ingredients)}

    def _window_get_ingredients(self, tokens: list[str]) -> dict:
        """Greedy longest-match fuzzy window scan over the OCR word stream (delimiter-agnostic).

        The segment matcher relies on the OCR preserving commas/bullets between names. When it
        doesn't -- a name is split across an inserted delimiter, or RapidOCR fragments/reorders words
        ('PRUNUS AMYGDALUS DUL CIS OIL', words scattered on a curved/multi-column panel) -- the
        segment match fails even though the characters are all present. This scans the raw word
        sequence: at each position try the longest window (up to 6 words) whose space-stripped form
        near-exactly matches an INCI name, accept it, and advance past it. Space-stripping makes it
        robust to the exact word boundaries the OCR chose; the high ratio + tight length ratio keep
        precision. CI codes are matched directly.
        """
        from rapidfuzz import fuzz, process

        blob = " ".join(tokens).lower()
        matched: list[str] = []
        for code in re.findall(r"\b[cс][il1|]\s*(\d{4,5})\b", blob):
            if f"ci {code}" in self._ci_codes:
                matched.append(f"ci {code}")
        words = re.sub(r"[^a-z0-9 ]", " ", blob).split()
        i, n = 0, len(words)
        while i < n:
            hit = None
            for L in range(min(6, n - i), 0, -1):  # longest window first (greedy longest match)
                win = "".join(words[i:i + L])
                if len(win) < 6:
                    continue
                exact = self._ns_to_name.get(win)  # O(1) fast path: clean fragmented read
                if exact is not None:
                    hit = (L, exact)
                    break
                m = process.extractOne(win, self._combined_tokens_ns, scorer=fuzz.ratio,
                                       score_cutoff=self.window_threshold)
                if m and 0.85 <= len(self._combined_tokens_ns[m[2]]) / len(win) <= 1.18:
                    hit = (L, self.combined_tokens[m[2]])
                    break
            if hit:
                matched.append(hit[1])
                i += hit[0]
            else:
                i += 1
        ings = remove_duplicates(matched)
        return {"ingredients": ings, "pollutants": self._compute_pollutants(ings)}

    @staticmethod
    def _drop_prefix_false_friends(ings: list[str], tokens: list[str]) -> list[str]:
        """Drop a single short dictionary word that appears in the OCR ONLY as the prefix of a
        longer alphabetic run, never as a standalone token.

        The Trie's longest-prefix match emits 'hydrogen' off 'HYDROGENATED CASTOR OIL' and 'phenol'
        off 'PHENOLSULFONATE' etc. -- coincidental dictionary prefixes of a longer word the Trie
        failed to assemble. The tell is purely textual: the matched name never occurs as a complete
        OCR word, only embedded at the start of a longer one. A genuinely-listed short ingredient
        ('silica', 'talc', 'alcohol', 'glycerin') appears as its own delimited token, so it is a
        complete word and is kept. Multi-word names are never touched (they are assembled, not
        prefix-coincidences). Conservative cap keeps long names out of scope.
        """
        maxlen = int(os.environ.get("PREFIX_FF_MAXLEN", "9"))
        words: set[str] = set()
        longer: set[str] = set()
        for line in tokens:
            for w in re.findall(r"[a-z0-9]+", str(line).lower()):
                words.add(w)
                if len(w) > 1:
                    longer.add(w)
        keep = []
        for nm in ings:
            parts = nm.split()
            if len(parts) == 1 and 3 <= len(nm) <= maxlen and nm not in words:
                if any(w != nm and w.startswith(nm) for w in longer):
                    continue  # only ever a prefix of a longer OCR word -> false friend
            keep.append(nm)
        return keep

    @staticmethod
    def _drop_fragments(ings: list[str], protected: set | frozenset = frozenset()) -> list[str]:
        """Drop a predicted ingredient that is a whole-word sub-phrase of another prediction.

        The Trie's longest-prefix match greedily emits short dictionary words that are really
        fragments of a longer name on the same panel ('hydrogen' from 'hydrogenated castor oil',
        'alcohol' from 'cetearyl alcohol', 'betaine' from 'cocamidopropyl betaine', 'butter' from
        '... parkii butter'). When the longer name is also predicted, the short one is a spurious
        FP -> drop it. Word-boundary containment only, so 'aqua' is never dropped by 'aqua-something'
        unless that token is actually present.

        BUT real INCI lists routinely contain BOTH a base ingredient and a derivative of it as
        SEPARATE list items ('dimethicone' + 'hydrogen dimethicone' + 'cetyl peg/ppg-10/1
        dimethicone'; 'silica' + 'hydrated silica'). Blindly dropping the base then costs a real TP.
        The window matcher is span-consuming (greedy longest match that advances past what it
        accepts), so when it emits the base name from its OWN delimited span -- distinct from the
        span of the longer name -- that base is a genuine separate listing, not a fragment. Such
        ``protected`` names are kept. A pure Trie false-friend like 'hydrogen' (only ever produced
        inside 'hydrogenated...') is never window-matched on its own, so it is still dropped.
        """
        wordsets = {i: i.split() for i in ings}
        keep = []
        for x in ings:
            if x in protected:
                keep.append(x)
                continue
            xs = wordsets[x]
            redundant = any(
                y != x and len(wordsets[y]) > len(xs) and _is_subseq(xs, wordsets[y])
                for y in ings
            )
            if not redundant:
                keep.append(x)
        return keep

    def get_ingredients_ensemble(self, primary_tokens, secondary_tokens, secondary_seg: float = 90.0) -> dict:
        """Match two OCR engines' tokens SEPARATELY and union the resulting INCI names.

        Concatenating raw tokens lets the noisier engine's garble spawn false positives on panels
        the primary engine already read cleanly. Matching each engine in its own context and unioning
        the *results* keeps the primary's precision while adding the secondary's extra recall on
        panels the primary garbled. The secondary runs at a stricter segment cutoff (its output is
        noisier) so it contributes high-confidence names only. Measured best on both eval sets:
        docTR(primary)@80 ∪ RapidOCR(secondary)@90 -> scraped 0.782 / curated 0.827 exact-F1.
        """
        return self.get_ingredients_multi([(primary_tokens, self.segment_threshold),
                                           (secondary_tokens, secondary_seg)])

    def get_ingredients_multi(self, sources) -> dict:
        """Generalised separate-match union over N OCR engines: sources = [(tokens, seg_thr), ...].
        Each engine is matched in its own context at its own segment threshold, then the resulting
        INCI names are unioned and fragment-cleaned. Adding more engines (docTR + RapidOCR + EasyOCR)
        raises recall on panels each engine alone garbles."""
        saved = self.segment_threshold
        union: list[str] = []
        protected: set = set()
        try:
            for tokens, seg in sources:
                self.segment_threshold = seg
                r = self.get_ingredients(tokens)
                union += list(r.get("ingredients", []))
                protected |= r.get("_win", set())  # span-authoritative window matches (see _drop_fragments)
        finally:
            self.segment_threshold = saved
        ings = remove_duplicates(union)
        if self.drop_fragments and ings:
            ings = self._drop_fragments(ings, protected)
        return {"ingredients": ings, "pollutants": self._compute_pollutants(ings)}

    def get_ingredients(self, tokens: list[str]) -> list[str]:
        marker_idx = -1
        if self.isolate_region:
            tokens, marker_idx = self._isolate_ingredient_region(tokens)
        strategy = self.match_strategy
        if strategy == "auto":
            # Route by how much pre-marker text was discarded. A real full-label photo buries the
            # list under many marketing/usage lines (NIVEA drops ~19) -> the segment matcher, robust
            # to garbled multi-word names, wins. A bare cropped panel often still has an
            # "Ingredients:" header but ~no text before it -> the Trie matcher is stronger and the
            # segment matcher's comma-splitting is fragile to panel delimiter styles. The amount of
            # dropped prose (not mere marker presence) is what distinguishes the two.
            strategy = "segment" if marker_idx >= self.segment_min_dropped else "trie"
        if strategy == "segment":
            res = self._segment_get_ingredients(tokens)
        elif strategy == "window":
            res = self._window_get_ingredients(tokens)
        elif strategy in ("union", "union3"):
            # Trie precision + segment recall on garbled visible names the Trie's exact-prefix match
            # drops + (union3) the delimiter-agnostic window scan that assembles names the OCR split
            # across commas or fragmented/reordered. Each matcher runs at its own precision-tuned bar.
            trie = list(self._trie_get_ingredients(tokens)["ingredients"])
            seg = list(self._segment_get_ingredients(tokens)["ingredients"])
            win = list(self._window_get_ingredients(tokens)["ingredients"]) if strategy == "union3" else []
            # Short Trie matches are the Trie's biggest false-friend source -- a coincidental dict
            # word read out of OCR garble, OR a dict prefix of a longer name the Trie failed to
            # assemble ('hydrogen' from 'hydrogenated...', 'betaine' from 'cocamidopropyl betaine',
            # 'phenol'/'butter'/'carbon'). Require a short Trie-only match to be corroborated by the
            # segment OR window matcher (which align WHOLE delimited names), so genuine short
            # ingredients ('aqua','talc','mica') -- read as clean delimited tokens, hence also found
            # by segment/window -- survive, while garble fragments drop. CORROB_MAXLEN sets the
            # length at/under which corroboration is required (default 4; raising it trims the
            # 6-8 char false-friends at some recall risk -- swept).
            corroborated = set(seg) | set(win)
            _cml = int(os.environ.get("CORROB_MAXLEN", "4"))
            trie = [t for t in trie if len(t) > _cml or t in corroborated]
            ings = remove_duplicates(trie + seg + win)
            if os.environ.get("PREFIX_FF", "1").lower() not in ("0", "false", "no", "off"):
                ings = self._drop_prefix_false_friends(ings, tokens)
            # Span-authoritative matches: a base name a delimiter-respecting matcher emits from its
            # OWN span (not inside the longer name's span) is a real separate listing, so shield it
            # from _drop_fragments. PROTECT_SRC picks the shield: 'seg' (comma/bullet segments, high
            # precision) | 'win' (window scan) | 'both' | 'none' (legacy). Carried to
            # get_ingredients_multi via the internal "_win" key.
            _psrc = os.environ.get("PROTECT_SRC", "seg").lower()
            protect = (set(seg) if _psrc in ("seg", "both") else set()) | (set(win) if _psrc in ("win", "both") else set())
            res = {"ingredients": ings, "pollutants": self._compute_pollutants(ings), "_win": protect}
        else:
            res = self._trie_get_ingredients(tokens)
        if self.drop_fragments and res.get("ingredients"):
            res["ingredients"] = self._drop_fragments(list(res["ingredients"]), res.get("_win", frozenset()))
            res["pollutants"] = self._compute_pollutants(res["ingredients"])
        return res

    def _trie_get_ingredients(self, tokens: list[str]) -> dict:
        cleaned_tokens = self.token_cleaner.clean_token(tokens)

        results = {"ingredients": [], "pollutants": []}
        trie_ingredients, mask_trie = self.trie.search(cleaned_tokens)
        remaining_tokens = trie_ingredients[mask_trie]

        if len(remaining_tokens) != 0:
            if self.match_backend == "rapidfuzz":
                matches, mask, _ = self.indexer.rapidfuzz_search(
                    "tokens", list(remaining_tokens), self.rapidfuzz_threshold
                )
            else:
                matches, mask, _ = self.indexer.search("tokens", remaining_tokens, self.ingredient_threshold)
            mask_corrected = np.bitwise_and(mask_trie[mask_trie], mask)

            idx_mask_trie = np.where(mask_trie)[0]
            trie_ingredients[idx_mask_trie[mask]] = matches[mask]

            mask_corrected_full = np.zeros_like(mask_trie, dtype=bool)
            mask_corrected_full[idx_mask_trie] = mask_corrected
            mask_trie[:] = mask_corrected_full

        ingredients = trie_ingredients[mask_trie]

        if len(ingredients) != 0:
            ingredients = self.remove_redundancy(remove_duplicates(ingredients))
            results["ingredients"] = ingredients

        if self.detection_type in ["pollutants", "both"] and len(ingredients) > 0:
            results["pollutants"] = remove_duplicates(
                [self.find_pollutants(ing) for ing in ingredients if self.find_pollutants(ing) is not None]
            )
        return results
