# How Whitespaces Affect LLM Performance in Arithmetic Calculations?

We conducted an experiment to determine whether whitespace modifications in arithmetic expressions influence how LLMs generate answers.

## Dataset

We used the `arithmetic_1dc` subset from [EleutherAI/arithmetic](https://huggingface.co/datasets/EleutherAI/arithmetic), which contains pairs like:

```
context: Question: What is (9 + 8) * 2? Answer:
completion: 34
```

The complexity level of these problems is appropriate for modern ~8B parameter LLMs. Short prompts also avoid context window limitations, allowing us to isolate the effect of minor prompt modifications. This dataset is also included in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/arithmetic) benchmark.

The dataset is loaded via `datasets` library from huggingface hub.

## Code

To run the analysis:

1) Open [arithmetic_robustness.py](arithmetic_robustness.py)

2) Set the `model_name` variable to your target model

3) Execute the script


## Methodology

### Whitespace Modifications

We added or removed spaces (ASCII 0x20) in positions that don't alter the mathematical meaning:
- Never inside words or numbers
- Up to 3 consecutive spaces could be added
- Spaces before brackets and operators could be removed

Example modification:

Original: `What is (9 + 8) * 2?`  
Modified: `What is ( 9+8)   * 2  ?`

### Experimental Setup

- **Prompt Format**: Fixed across all tested models (no model-specific optimization)
- **Generation Paradigm**: Greedy decoding with answer extraction via `<start>number</start>` pattern
- **Chat Template**: Standard template applied via tokenizer's `apply_chat_template` with `enable_thinking=False`
- **Statistical Test**: McNemar's test (p<0.05 threshold) for significance testing

*Note*: McNemar's test becomes unreliable when failure rates are very low (e.g., for Qwen3-32B's near-perfect accuracy).

## Results

| Model                                  | Instruction-following Acc. | Population | Original Acc. | Distorted Acc. | Significant Difference |
|----------------------------------------|----------------------------|------------|---------------|----------------|------------------------|
| Qwen/Qwen3-14B                         | 1.0                        | 2000       | 0.946         | 0.964          | True                   |
| Qwen/Qwen3-8B                          | 1.0                        | 2000       | 0.897         | 0.939          | True                   |
| t-tech/T-lite-it-1.0                   | 1.0                        | 2000       | 0.762         | 0.676          | True                   |
| ai-sage/GigaChat-20B-A3B-instruct      | 0.99625                    | 1985       | 0.498         | 0.507          | False                  |
| mistralai/Mistral-7B-Instruct-v0.3     | 0.93275                    | 1749       | 0.207         | 0.189          | True                   |
| yandex/YandexGPT-5-Lite-8B-instruct    | 0.981                      | 1926       | 0.201         | 0.088          | True                   |
| HuggingFaceTB/SmolLM2-1.7B-Instruct    | 0.972                      | 1890       | 0.0           | 0.0            | True                   |

**Column Descriptions:**
- **Instruction-following accuracy**: Ratio of responses containing the `<start>number</start>` pattern
- **Population**: Valid response pairs (both original/modified prompts contained extractable answers)
- **Original Acc.**: Accuracy with unmodified prompts
- **Distorted Acc.**: Accuracy with whitespace-modified prompts
- **Significant Difference**: McNemar's test result (p<0.05)

## Key Findings

1. **Whitespaces significantly affect LLM performance** in arithmetic calculations
2. **Qwen3 models show improved accuracy** with added whitespaces, while other models degrade
3. **GigaChat-20B-A3B-instruct performance is not affected** by whitespace noise (difference of mean accuracies is not significant)
4. **Instruction-following capability**:
   - Qwen3 achieves perfect compliance
   - Other models occasionally deviate from response formatting requirements
5. **Russian-language LLMs** underperform likely due to English prompt mismatch (we maintained consistency by not using Russian prompts)

   
