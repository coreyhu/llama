import argparse
import gc
import json
import os
import shutil
import warnings

import torch

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.convert_llama_weights_to_hf import write_model, compute_intermediate_size, write_json, read_json, NUM_SHARDS

try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

def write_model(model_path, input_base_path, model_size, tokenizer_path=None, safe_serialization=True):
    # for backward compatibility, before you needed the repo to be called `my_repo/model_size`
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        max_position_embeddings = 2048

    tokenizer_class = LlamaTokenizerFast or LlamaTokenizer
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if num_shards == 1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if num_shards == 1:
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            # Sharded
            # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
            # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
            # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim),
                num_key_value_heads,
                key_value_dim,
                dim,
            )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards == 1:
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    print("Saved pretrained models")
    shutil.rmtree(tmp_model_path)


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)

if __name__ == "__main__":
    write_model(
        model_path="/home/corey/menlo/llama/llama-2-13b-huggingface",
        input_base_path="/home/corey/menlo/llama/llama-2-13b",
        model_size="13B",
        safe_serialization=False,
        tokenizer_path="/home/corey/menlo/llama/tokenizer.model"
    )
    
