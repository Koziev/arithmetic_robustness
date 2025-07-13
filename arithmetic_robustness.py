import re
import random

import terminaltables
from datasets import load_dataset
import torch
import transformers
import numpy as np
import tqdm
from statsmodels.stats.contingency_tables import mcnemar


def execute_query(query: str) -> str:
    messages = [{"role": "user",
                 "content": query + ' Compute and print only the result of the computation, surrounded by the tokens <start> and </start>.'}, ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    outputs = model.generate(input_ids, **generation_args)
    output_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    m = re.search(r'<start>(.+?)</start>', output_text)
    if m:
        answer = m.group(1).strip()
        return answer

    return None


def report(hits_original, hits_distorted):
    """
    Here we process the inference results. The input vectors hits_original and hits_distorted contain 0 and 1,
    which are obtained by comparing the model generation and the reference response for the original query
    and the query with added/removed spaces.
    """

    hits_original = np.array(hits_original)
    hits_distorted = np.array(hits_distorted)

    print('Population: {}'.format(len(hits_original)))
    print('Original query performance: {}'.format(round(np.mean(hits_original), 3)))
    print('Distorted query performance: {}'.format(round(np.mean(hits_distorted), 3)))

    # the contingency table
    table = np.array([
        [sum((hits_original == 1) & (hits_distorted == 1)), sum((hits_original == 1) & (hits_distorted == 0))],  # a, b
        [sum((hits_original == 0) & (hits_distorted == 1)), sum((hits_original == 0) & (hits_distorted == 0))]  # c, d
    ])

    # McNemar's test
    result = mcnemar(table, exact=False, correction=True)  # exact=False for chi-square, True for exact binomial
    print(f"McNemar's chi-square statistic: {result.statistic}")
    print(f"p-value: {result.pvalue}")

    # Interpretation
    alpha = 0.05
    if result.pvalue < alpha:
        print("Reject the null hypothesis (significant difference)")
    else:
        print("Fail to reject the null hypothesis (no significant difference)")

    return np.mean(hits_original), np.mean(hits_distorted), result.pvalue < alpha


if __name__ == '__main__':
    #model_name = "Qwen/Qwen3-0.6B"
    #model_name = "Qwen/Qwen3-8B"
    #model_name = "Qwen/Qwen3-14B"
    #model_name = "microsoft/Phi-3.5-mini-instruct"
    #model_name = "t-tech/T-lite-it-1.0"
    #model_name = "yandex/YandexGPT-5-Lite-8B-instruct"
    #model_name = "ai-sage/GigaChat-20B-A3B-instruct"
    #model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    #model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    #model_name = "GSAI-ML/LLaDA-8B-Instruct"
    #model_name = "tiiuae/falcon-7b-instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()

    generation_args = {'num_return_sequences': 1,
                       'do_sample': False,
                       'max_new_tokens': 50,
                       'eos_token_id': tokenizer.eos_token_id,
                       'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                       #'no_repeat_ngram_size': 10,
                       #'renormalize_logits': args.renormalize_logits,
                       'temperature': None,
                       'top_p': None,
                       'top_k': None,
                       #'typical_p': args.typical_p,
                       }

    # ----------------------------------------------------------------------------------

    results = [['Model', 'Dataset', 'Task', 'Instruction-following acc.', 'Population', 'Acc. of original', 'Acc. of distorted', 'Diff. significant']]

    ds_name = "EleutherAI/arithmetic"
    for task in ["arithmetic_1dc"]:
        ds = load_dataset(ds_name, task, streaming=False)["validation"]

        hits_original = []
        hits_distorted = []
        num_instruct_compliant_responses = 0
        total_num_responses = 0
        for sample in tqdm.tqdm(ds):
            m = re.search('^Question:(.+)\n', sample['context'], flags=re.MULTILINE)
            query = m.group(1).strip()

            answer_original = execute_query(query)

            query = re.sub(' ', lambda m: ' ' * random.randint(1, 4), query)
            query = re.sub('[ ]*([()+\-*/])[ ]*', r'\1', query)
            query = re.sub(r'\?', ' ' * random.randint(0, 4)+'?'+' ' * random.randint(0, 4), query)
            query = re.sub(r'(\w)\(', r'\1  (', query)  # restore whitespace in substring "is("
            query = ' ' * random.randint(0, 4) + query
            #query = query + ' ' * random.randint(0, 4)

            answer_distorted = execute_query(query)

            # LLM runs 2 times per sample. So the responses for original and distorted query are taken into account independently.
            total_num_responses += 1
            if answer_original:
                num_instruct_compliant_responses += 1

            total_num_responses += 1
            if answer_distorted:
                num_instruct_compliant_responses += 1

            # If both original and distorted query result to compliant responses
            if answer_original and answer_distorted:
                hits_original.append(int(answer_original == sample['completion'].strip()))
                hits_distorted.append(int(answer_distorted == sample['completion'].strip()))

                if 0 == len(hits_original)%50:
                    print('\n\n')
                    report(hits_original, hits_distorted)
                    print('\n\n')

        print('-'*80)
        acc_original, acc_distorted, diff_significant = report(hits_original, hits_distorted)
        ifa = num_instruct_compliant_responses / total_num_responses  # instruction-following accuracy
        results.append((model_name, ds_name, task, ifa, len(hits_original), round(acc_original, 3), round(acc_distorted, 3), diff_significant))

    print('Final results:')
    print(terminaltables.AsciiTable(results).table)

    with open('airthmetic_robustness.{}.md'.format(model_name.split('/')[-1]), 'w') as f:
        f.write('\n\n' + terminaltables.GithubFlavoredMarkdownTable(results).table+'\n\n')
