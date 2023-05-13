def tokenize_splits_preview_iter(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    processed_tokens = 0
    i = 1
    tokens_to_form_full_word = []
    full_word = ""
    while i <= len(tokens):
        test_tokens_to_form_full_word = tokens[processed_tokens:i]
        test_full_word = tokenizer.convert_tokens_to_string(
            test_tokens_to_form_full_word)
        if len(test_full_word) > 1:
            if full_word:
                # We got tokens that should belong to the next word.
                # Yield the previous full word and reset the list.
                yield (full_word, len(tokens_to_form_full_word))
            else:
                # We do not have a previous word, so this might be an English word. Yield it.
                yield (test_full_word, len(test_tokens_to_form_full_word))
                i += 1
            # Reset the list of tokens to form a full word.
            tokens_to_form_full_word = []
            full_word = ""
            # Set processed_tokens to the first token of the next word.
            processed_tokens = i - 1
        else:
            tokens_to_form_full_word = test_tokens_to_form_full_word
            full_word = test_full_word
            # Try to add another token to the word on the next iteration.
            i += 1
    # If we have anything left, yield it.
    if full_word:
        yield (full_word, len(tokens_to_form_full_word))


def tokenize_splits_preview(tokenizer, text):
    return list(tokenize_splits_preview_iter(tokenizer, text))
