from typing import Union

import re
import fire
import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer as TokenizerFast
from datasets import Dataset, load_dataset
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from termcolor import colored
from textwrap import indent, dedent

from utils.config import Config
from utils.paths import Paths
from utils.formatting import (
    better_format_pairs_in_json_text,
    get_human_timestamp,
    human_short_number as hs_number,
    truncate_string_by_lines,
)
from utils.data_processing import (
    shallow_diff_dict,
    unique_list,
)
from utils.type_checking import (
    assert_list_of_strings,
)
from utils.tokenize_splits_preview import tokenize_splits_preview
from utils.update_hf_readme import update_hf_readme


def build_tokenizer(
    cfg: Union[str, None] = None,
    config_file_path: Union[str, None] = None,
    data_dir_path: Union[str, None] = None
):
    paths = Paths(data_dir_path)
    if cfg and not config_file_path:
        config_file_path = paths.get_config_path(cfg)
    config = Config(config_file_path)

    tokenizer_config = config.tokenizer_config

    message = ''
    message += f"Building new tokenizer '{config.tokenizer_name}' by adding "
    message += f"{hs_number(tokenizer_config.tokens_to_add)} new tokens "
    message += f"to '{config.base_tokenizer_name}'..."
    print(message)
    print()

    base_tokenizer = AutoTokenizer.from_pretrained(config.base_tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_tokenizer_name)
    setup_tokenizer(tokenizer)

    if tokenizer_config.build_with == 'word_frequency_list':
        wf_list_name = \
            tokenizer_config.settings['word_frequency_list_name']
        print(
            f"New tokens will be added based on word frequency list: '{wf_list_name}'.")

        tokenizer_json = json.loads(base_tokenizer._tokenizer.to_str())

        vocab = tokenizer_json['model']['vocab']
        merges = tokenizer_json['model']['merges']

        original_last_vocab_id = max(vocab.values())

        # Remove added tokens which are not listed in the vocab, to avoid
        # id conflicts with the new tokens we are going to add.
        tokenizer_json['added_tokens'] = \
            [t for t in tokenizer_json['added_tokens']
             if t['id'] < original_last_vocab_id]
        tokenizer._tokenizer = \
            TokenizerFast.from_str(json.dumps(tokenizer_json))

        specificity_included_words = unique_list(
            assert_list_of_strings(
                tokenizer_config.settings.get('include_words', [])
            )
        )
        if specificity_included_words:
            print(
                f"Also, {len(specificity_included_words)} specificity included words listed in config will be added:",
                specificity_included_words)
            print()

        print(f"Loading word frequency list '{wf_list_name}'...")
        ds: Dataset = load_dataset(wf_list_name)['train']  # type: ignore
        print(f"Words in frequency list: {len(ds)}")

        words_list = []
        added_words_set = set(vocab.keys())

        def add_word_to_list_if_not_added(word):
            if word not in added_words_set:
                added_words_set.add(word)
                words_list.append(word)
                return True
            return False

        print()
        print('Processing characters and specificity included words...')

        progress_bar = tqdm(
            total=(
                len(specificity_included_words) +
                len(ds) +
                len(specificity_included_words)
            )
        )
        # Add all chars into word list
        for word in specificity_included_words:
            chars = list(word)
            for c in chars:
                add_word_to_list_if_not_added(c)
            progress_bar.update(1)
        # Add all chars from word frequency list into word list
        for w in ds:
            word = w['word']  # type: ignore
            chars = list(word)
            for c in chars:
                add_word_to_list_if_not_added(c)
            progress_bar.update(1)
        # Add all specificity_included_words into word list
        for word in specificity_included_words:
            add_word_to_list_if_not_added(word)
            progress_bar.update(1)
        progress_bar.close()

        print(
            f"{len(words_list)} characters and specificity included words will be added as new tokens.")

        remaining_tokens_to_add = \
            tokenizer_config.tokens_to_add - len(words_list)

        print()
        if remaining_tokens_to_add < 0:
            print(colored(
                f"Warning: the number of unique characters and specificity included words ({len(words_list)}) already exceeds the limit of tokens to add ({tokenizer_config.tokens_to_add}). Some will be ignored.",
                'yellow',
                attrs=['bold']
            ))
            words_list = words_list[:tokenizer_config.tokens_to_add]
        if remaining_tokens_to_add <= 0:
            print("No space for other words to be added.")
        else:
            print(
                f"Collecting {remaining_tokens_to_add} more words from the word frequency list...")

            modified_words = []
            process_word = get_process_word_fn(
                tokenizer_config=tokenizer_config,
                modified_words_list=modified_words,
            )

            progress_bar = tqdm(total=remaining_tokens_to_add)
            for w in ds:
                if len(words_list) > tokenizer_config.tokens_to_add:
                    break

                word = w['word']  # type: ignore
                pos = w['pos']  # type: ignore
                words = process_word(word, pos)
                if not isinstance(words, list):
                    words = [words]
                for word in words:
                    if word:
                        word = word.strip()
                    if not word:
                        continue
                    added = add_word_to_list_if_not_added(word)
                    if added:
                        progress_bar.update(1)

            progress_bar.close()

            if len(words_list) < tokenizer_config.tokens_to_add:
                print(colored(
                    f"Warning: the number of unique words in the word frequency list isn't enough for adding {tokenizer_config.tokens_to_add} new tokens.",
                    'yellow',
                    attrs=['bold']
                ))

            if modified_words:
                modified_words_log_path = paths.get_log_path(
                    f"{config.tokenizer_name}-modified-words.jsonl")
                with open(modified_words_log_path, 'w') as f:
                    f.write(
                        '\n'.join([
                            json.dumps(item, ensure_ascii=False)
                            for item in modified_words]))
                print(colored(
                    f"{len(modified_words)} words has been modified base on rules, see {modified_words_log_path} for details.",
                    attrs=['bold']
                ))

            words_list = sorted(words_list, key=len)

        words_to_add_log_path = paths.get_log_path(
            f"{config.tokenizer_name}-words-to-add.txt")
        with open(words_to_add_log_path, 'w') as f:
            f.write('\n'.join(words_list))
        print()
        print(colored(
            f"{len(words_list)} new words will be added to the tokenizer, see {words_to_add_log_path} for details.",
            attrs=['bold']
        ))

        print()
        print("Building new tokenizer...")

        vocab_set = set(vocab.keys())
        next_new_token_id = max(vocab.values()) + 1

        def add_token_to_vocab(new_token):
            nonlocal next_new_token_id, vocab_set
            vocab[new_token] = next_new_token_id
            next_new_token_id += 1
            vocab_set.add(new_token)

        build_tokenizer_log_path = paths.get_log_path(
            f"{config.tokenizer_name}-build.txt")
        build_tokenizer_log_file = open(build_tokenizer_log_path, 'w')

        def write_log(text, print=True):
            build_tokenizer_log_file.write(text + '\n')
            if print:
                tqdm.write(text)

        progress = tqdm(enumerate(words_list), total=len(words_list))
        for w_i, new_word in progress:
            if w_i % 10 == 0:
                progress.set_description(new_word)
            tokens = tokenizer.tokenize(new_word)
            new_token = ''.join(tokens)
            if new_token in vocab_set:
                continue

            if len(tokens) > 2:
                write_log(
                    f"New word '{new_word}' consists of more than 2 old tokens: {tokens} ({[tokenizer.convert_tokens_to_string([t]) for t in tokens]}).",
                    # do not be too verbose since we can't do anything about a
                    # single character word to consists of more than 2 old
                    # tokens
                    print=len(new_word) > 1
                )
                while len(tokens) > 2:
                    merges.append(f"{tokens[0]} {tokens[1]}")
                    # merge the first two tokens
                    tokens[0:2] = [''.join(tokens[0:2])]
                    # Add the new merged token to vocab if it's not already in vocab
                    n_t = tokens[0]
                    if n_t not in vocab_set:
                        add_token_to_vocab(n_t)
            if len(tokens) > 1:
                merges.append(f"{tokens[0]} {tokens[1]}")

            add_token_to_vocab(new_token)
            tokenizer._tokenizer = TokenizerFast.from_str(
                json.dumps(tokenizer_json))

        summary = ''
        summary += f"Original vocab size: {base_tokenizer.vocab_size}"
        summary += '\n'
        summary += f"New vocab size: {tokenizer.vocab_size}"
        summary += '\n'
        summary += f"{tokenizer.vocab_size - base_tokenizer.vocab_size} new tokens are added."
        write_log(summary, print=False)
        print()
        print(colored(
            summary,
            attrs=['bold']
        ))
    else:
        raise ValueError(
            f"Unknown build_with value: {tokenizer_config.build_with}")

    new_tokenizer_save_path = paths.get_tokenizer_path(config.tokenizer_name)
    tokenizer.save_pretrained(new_tokenizer_save_path)
    print("New tokenizer Saved to:", new_tokenizer_save_path)

    print()
    print()
    print(colored(
        '---- Samples ----',
        attrs=['bold']
    ))
    print()
    samples = [
        '網際網路（英語：Internet）是指 20 世紀末期興起電腦網路與電腦網路之間所串連成的龐大網路系統。',
        '人工智慧（英語：artificial intelligence，縮寫為 AI），是指由人製造出來的機器所表現出來的智慧。',
        '程式設計師們越來越依賴 Git 進行版本控制、使用 Python、Ruby 或 JavaScript 等程式語言開發 Web 應用程式。',
    ]
    for sample in samples:
        print('Sample:', sample)
        print('Original:', tokenize_splits_preview(base_tokenizer, sample))
        print('     New:', tokenize_splits_preview(tokenizer, sample))
        print()
    print()

    if config.push_outputs_to_hf:
        hf_model_name = f"{config.hf_user_or_org_name}/{config.tokenizer_name}"
        try:
            old_tokenizer_on_hf = AutoTokenizer.from_pretrained(hf_model_name)
            old_vocab = old_tokenizer_on_hf.vocab
            new_vocab = tokenizer.vocab

            diff_results = shallow_diff_dict(old_vocab, new_vocab)
            if diff_results['added'] or diff_results['updated'] or diff_results['removed']:
                vocab_diff_log_path = paths.get_log_path(
                    f"{config.tokenizer_name}-vocab-diff-{get_human_timestamp()}.json")
                print(colored(
                    f"The vocab has been modified.",
                    color='yellow',
                    attrs=['bold']
                ))
                for t in ['added', 'updated', 'removed']:
                    if not diff_results[t]:
                        continue
                    ter = tokenizer
                    if t == 'removed':
                        ter = old_tokenizer_on_hf
                    diff_results[t] = {
                        ter.convert_tokens_to_string([k]): v
                        for k, v in diff_results[t].items()}
                    details = json.dumps(
                        diff_results[t], indent=2, ensure_ascii=False)
                    details = better_format_pairs_in_json_text(details)
                    details = truncate_string_by_lines(details, max_lines=12)
                    details = t.title() + ': ' + details
                    details = indent(details, '  ')
                    print(colored(
                        details,
                        color='yellow',
                        attrs=['bold']
                    ))
                with open(vocab_diff_log_path, 'w') as f:
                    json_text = json.dumps(
                        diff_results, indent=2, ensure_ascii=False)
                    json_text = better_format_pairs_in_json_text(json_text)
                    f.write(json_text)
                print(colored(
                    f"Any already trained model will not be compatible with this updated tokenizer. See {vocab_diff_log_path} for details.",
                    color='yellow',
                    attrs=['bold']
                ))

        except Exception as e:
            if not isinstance(e, OSError):
                print(colored(
                    f"Warning: Error on loading existing tokenizer '{hf_model_name}' and compare the vocab. Error: {e}",
                    color='red',
                    attrs=['bold']
                ))
        # api = HfApi()
        # user_info = api.whoami()
        print("Pushing to HF Hub...")
        results = tokenizer.push_to_hub(
            hf_model_name,
            private=True
        )
        print(results)
        print("Updating model card...")
        model_card_frontmatter = {
            'language': ['zh', 'en']
        }
        model_card_content = dedent(f"""
        # {config.tokenizer_name}

        This tokenizer is a part of the `{config.project_name}` project.

        * Base tokenizer: `{config.base_tokenizer_name}`
        * Built with: `{tokenizer_config.build_with}`
        * Vocab size: `{tokenizer.vocab_size}`
        * Tokens added (planned/actual): `{tokenizer_config.tokens_to_add}` / `{tokenizer.vocab_size - base_tokenizer.vocab_size}`
        * Full config:
          ```json
          {tokenizer_config.to_json()}
          ```
        """).strip()
        update_hf_readme(hf_model_name, model_card_content,
                         model_card_frontmatter)
        print(colored(
            f"Model uploaded to https://huggingface.co/{hf_model_name}.",
            attrs=['bold']
        ))
        fs = HfFileSystem()
        with fs.open(f"{hf_model_name}/human_tokens_map.json", 'w') as f:
            f.write(json.dumps({
                tokenizer.convert_tokens_to_string([k]): v
                for k, v in sorted(tokenizer.vocab.items(), key=lambda x: x[1])
            }, indent=2, ensure_ascii=False))


def setup_tokenizer(tokenizer):
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    added_tokens = tokenizer_json.get('added_tokens', [])
    eos_token_id = None
    pad_token_id = None

    # Find the special tokens
    for t in added_tokens:
        if t.get('content') == '<|endoftext|>':
            eos_token_id = t.get('id')
        elif t.get('content') == '<|padding|>':
            pad_token_id = t.get('id')

    # Let the unk and bos to be different from the eos token.
    if eos_token_id is not None and pad_token_id is not None:
        if tokenizer.unk_token_id == eos_token_id:
            tokenizer.unk_token_id = pad_token_id
        if tokenizer.bos_token_id == eos_token_id:
            tokenizer.bos_token_id = pad_token_id


def get_process_word_fn(tokenizer_config, modified_words_list):
    word_replace_rules = \
        tokenizer_config.settings.get('replace_rules', [])

    def raise_invalid_replace_rule_error(message):
        raise ValueError(
            f"Invalid config: {'.'.join(tokenizer_config.config_level)}.word_frequency_list_settings.replace_rules: {message}. Please check {tokenizer_config.config_file_path}."
        )

    if not isinstance(word_replace_rules, list):
        raise_invalid_replace_rule_error("should b a list")

    def process_word(word, pos):
        has_matched = False
        for rule in word_replace_rules:
            # Break if the word has been matched by another rule.
            if has_matched:
                break

            # Don't proceed if the word in the "except" list.
            if word in rule.get('except', []):
                continue

            rule_match = rule.get('match')
            if not rule_match:
                raise_invalid_replace_rule_error(
                    'A rule must have a "match" field.')

            # Simple matching: rule_match is a string, matching the word directly.
            if isinstance(rule_match, str):
                if rule_match == word:
                    replace_with = rule.get('replace')
                    modified_words_list.append({
                        'word': word,
                        'replaced_by': replace_with,
                        'rule': rule
                    })
                    word = replace_with
                    has_matched = True

            # Complex matching: rule_match is a dict.
            elif isinstance(rule_match, dict):
                # Keep track of processed keys so we can raise an error if
                # there're any unexpected ones.
                processed_rule_match_keys = []

                # Pos should be in the list if provided.
                if 'pos' in rule_match:
                    processed_rule_match_keys.append('pos')
                    expect_pos = rule_match['pos']
                    if isinstance(expect_pos, str):
                        expect_pos = [expect_pos]

                    # Pos is not listed, try next rule
                    if pos not in expect_pos:
                        continue

                # Should match the regex if provided.
                matching_regex = None
                if 'regex' in rule_match:
                    processed_rule_match_keys.append('regex')
                    regex = rule_match['regex']
                    if isinstance(regex, str):
                        regex = [regex]

                    for r in regex:
                        # Break if already has a match
                        if matching_regex:
                            break
                        if not isinstance(r, str):
                            raise_invalid_replace_rule_error(
                                f"regex must be a string, got: {r}")
                        compiled_r = re.compile(r)
                        if re.search(compiled_r, word):
                            matching_regex = compiled_r

                    # No matching regex, try next rule
                    if not matching_regex:
                        continue

                # Check if there're any unexpected keys in the 'match' dict
                rule_match_keys_set = set(rule_match.keys())
                processed_rule_match_keys_set = set(
                    processed_rule_match_keys)
                if rule_match_keys_set > processed_rule_match_keys_set:
                    raise_invalid_replace_rule_error(
                        f"there are unprocessed keys in the 'match' field: {rule_match_keys_set - processed_rule_match_keys_set}"
                    )

                # When we reach this point, it means that we already had passed
                # all match rules. Proceed with replacing.
                has_matched = True

                # Replace with regex sub
                sub_with = rule.get('sub')
                if sub_with:
                    if not matching_regex:
                        raise_invalid_replace_rule_error(
                            f"sub_with ('{sub_with}') can only be used with regex match")
                    try:
                        new_word = re.sub(
                            matching_regex, sub_with, word)  # type: ignore
                    except Exception as e:
                        error_message = '\n'.join([
                            str(e),
                            f"regex: {matching_regex}",
                            f"sub: {sub_with}",
                        ])
                        raise Exception(error_message) from e
                    modified_words_list.append({
                        'word': word,
                        'replaced_by': new_word,
                        'rule': rule
                    })
                    word = new_word

                # Replace with word or null (remove)
                else:
                    replace_with = rule.get('replace')
                    modified_words_list.append({
                        'word': word,
                        'replaced_by': replace_with,
                        'rule': rule
                    })
                    word = replace_with
            else:
                raise_invalid_replace_rule_error(
                    'A the "match" field of a rule must be a string or dict.')

        return word

    return process_word


if __name__ == "__main__":
    fire.Fire(build_tokenizer)
