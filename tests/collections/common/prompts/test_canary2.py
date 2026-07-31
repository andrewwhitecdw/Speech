import unittest
from contextlib import ExitStack
from unittest import mock

import torch

from nemo.collections.common.prompts.canary2 import CANARY_EOS, canary2


class FakeTokenizer:
    def __init__(self, eos_id):
        self.eos_id = eos_id

    def token_to_id(self, token):
        return self.eos_id


class FakePrompt:
    OUTPUT_ROLE = "assistant"
    PROMPT_LANGUAGE_SLOT = "prompt_language"

    def __init__(self, eos_id=3):
        self.tokenizer = FakeTokenizer(eos_id)

    def encode_dialog(self, turns):
        eos = self.tokenizer.eos_id if self.tokenizer.eos_id is not None else 99
        return {
            "answer_ids": torch.tensor([1, 2, eos]),
            "turns": turns,
        }


class FakeSupervision:
    def __init__(self, text=None, language=None):
        self.text = text
        self.language = language


class FakeMonoCut:
    def __init__(self, id, custom, supervisions):
        self.id = id
        self.custom = custom
        self.supervisions = supervisions


class FakeMixedCut:
    pass


class TestCanary2PromptFn(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        target = "nemo.collections.common.prompts.canary2"
        self.stack.enter_context(mock.patch(f"{target}.MonoCut", FakeMonoCut))
        self.stack.enter_context(mock.patch(f"{target}.MixedCut", FakeMixedCut))

    def tearDown(self):
        self.stack.close()

    def test_canary2_handles_empty_supervisions(self):
        cut = FakeMonoCut(
            id="cut-1",
            custom={"source_lang": "en", "target_lang": "en"},
            supervisions=[],
        )
        prompt = FakePrompt(eos_id=3)
        result = canary2(cut, prompt)
        self.assertEqual(len(result["answer_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
