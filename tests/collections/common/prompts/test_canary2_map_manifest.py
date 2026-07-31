from nemo.collections.common.prompts.canary2 import (
    CANARY_SPECIAL_TOKENIZER,
    map_manifest_values_to_special_tokens,
)


def test_map_manifest_values_wraps_plain_tokens_and_injects_language():
    result = map_manifest_values_to_special_tokens({
        "source_lang": "en",
        "target_lang": "pl",
        "pnc": "true",
        "itn": "no",
    })
    assert result == {
        "source_lang": "<|en|>",
        "target_lang": "<|pl|>",
        "pnc": "<|pnc|>",
        "itn": "<|noitn|>",
        "prompt_language": CANARY_SPECIAL_TOKENIZER,
    }
