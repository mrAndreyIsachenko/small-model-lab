from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS: list[str] = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def train_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int = 32768,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens or SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train_from_iterator(iterator, trainer=trainer)
        return cls(tokenizer)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(p))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Token '{token}' not found in vocabulary")
        return token_id

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id("<|bos|>")

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id("<|eos|>")

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id("<|pad|>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def compression_ratio_bytes_per_token(self, texts: Iterable[str]) -> float:
        total_bytes = 0
        total_tokens = 0

        for text in texts:
            # Encode without BOS/EOS to get pure token count
            encoded = self.encode(text)
            total_tokens += len(encoded)
            total_bytes += len(text.encode("utf-8"))

        if total_tokens == 0:
            return 0.0

        return total_bytes / total_tokens
