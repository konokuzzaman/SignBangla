"""Bangla speech-to-text with any Whisper model from Hugging Face (model name in configs/config.yaml).

Whisper's tokens are pieces of UTF-8 bytes, so a Bangla word costs about 11 tokens. The 60 words do not fit in one initial
prompt (the limit is 448 tokens including the answer), so prompt_batches splits them up, transcribe tries each batch, and
the text matcher picks the best result (text_processing.best_candidate).
"""
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

from ..labels import LABELS
from .dictionary import split_variants
from .text_processing import best_candidate

N_START_TOKENS = 4  # <|startoftranscript|> <|bn|> <|transcribe|> <|notimestamps|>
SAFETY_TOKENS = 2


def vocabulary_words(labels=LABELS):
    """Every Bangla spelling in the sign dictionary."""
    return [name for _, bangla, *_ in labels for name in split_variants(bangla)]


def prompt_batches(words, count_tokens, limit):
    """Split words into prompts of at most `limit` tokens, in order, each word in exactly one. count_tokens(text) -> int."""
    batches, current = [], []
    for word in words:
        if current and count_tokens(" ".join(current + [word])) > limit:
            batches.append(current)
            current = []
        current.append(word)
    if current:
        batches.append(current)
    return [" ".join(b) for b in batches]


class WhisperASR:
    """Load once, then call transcribe(audio) for each recording. `audio` is float32 mono at 16 kHz.

    generation_config_from: some fine-tuned models ship a config too old for the `language` argument. Their language
    tokens are the standard ones, so the config of this stock Whisper model is used for them instead.
    """

    def __init__(self, model, language="bn", device="auto", max_new_tokens=64, words=None,
                 generation_config_from="openai/whisper-small"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.language, self.max_new_tokens = language, max_new_tokens
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        if not hasattr(self.model.generation_config, "lang_to_id"):
            self.model.generation_config = GenerationConfig.from_pretrained(generation_config_from)
        limit = self.model.config.max_target_positions - N_START_TOKENS - max_new_tokens - SAFETY_TOKENS
        count = lambda text: len(self.processor.get_prompt_ids(text))
        self.prompts = [self.processor.get_prompt_ids(p, return_tensors="pt").to(self.device)
                        for p in prompt_batches(words or [], count, limit)]

    def _generate(self, features, prompt_ids=None):
        kw = dict(language=self.language, task="transcribe", max_new_tokens=self.max_new_tokens)
        if prompt_ids is not None:
            kw["prompt_ids"] = prompt_ids
        with torch.inference_mode():
            out = self.model.generate(features, **kw)
        # max_new_tokens can stop in the middle of a Bangla letter, which decodes to the replacement character
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].replace("�", "").strip()

    def candidates(self, audio, sample_rate=16000):
        """One transcription per prompt batch, or a single one when there are no prompts."""
        features = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
        features = features.to(self.device, self.dtype)
        return [self._generate(features, p) for p in self.prompts] or [self._generate(features)]

    def transcribe(self, audio, matcher, sample_rate=16000):
        """The transcription that matches the sign dictionary best."""
        return best_candidate(self.candidates(audio, sample_rate), matcher)
