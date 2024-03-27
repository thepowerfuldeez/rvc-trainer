import torch
import argparse


def transform_mha_checkpoint(old_checkpoint_path, new_checkpoint_path):
    # Load the old checkpoint
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')

    new_state_dict = {"iteration": checkpoint['iteration'],
                      'optimizer': checkpoint['optimizer'],
                      'learning_rate': checkpoint['learning_rate'],
                      'model': {}
                      }
    for key, param in checkpoint['model'].items():
        # Handle MultiHeadAttention weights transformation
        if "attn_layers" in key and "conv_" in key:
            print(key)
            key_normalized = key[len("enc_p.encoder."):]  # keep only attn_layers ... part
            layer_index = key_normalized.split(".")[1]  # Extract layer index from key
            param_type = key_normalized.split(".")[-1]  # Extract parameter type (weight or bias)

            # Query, Key, Value weights and biases concatenation for in_proj
            # q, k, v keys go in correct order
            if "conv_q" in key or "conv_k" in key or "conv_v" in key:
                base_key = f"enc_p.encoder.attn_layers.{layer_index}.in_proj_{param_type}"
                if param_type == 'bias':
                    param_to_paste = param.view(-1)
                else:
                    param_to_paste = param  # .view(1, -1)

                if param_to_paste.size(-1) == 1:
                    param_to_paste = param_to_paste.squeeze(-1)
                if base_key not in new_state_dict['model']:
                    new_state_dict['model'][base_key] = param_to_paste  # Reshape and init
                else:
                    new_state_dict['model'][base_key] = torch.cat((new_state_dict['model'][base_key], param_to_paste),
                                                                  dim=0)  # Concatenate
                print(base_key, new_state_dict['model'][base_key].shape)
            # Output projection weights and biases remain the same, just adjust the key
            elif "conv_o" in key:
                new_key = key.replace("conv_o", "out_proj")
                if param.size(-1) == 1:
                    param_to_paste = param.squeeze(-1)
                else:
                    param_to_paste = param
                new_state_dict['model'][new_key] = param_to_paste
                print(new_key)

        # Handle LayerNorm parameter name change (gamma, beta to weight, bias)
        elif "encoder.norm_layers" in key:
            new_key = key.replace("gamma", "weight").replace("beta", "bias")
            new_state_dict['model'][new_key] = param
        else:
            # Directly transfer parameters that don't need transformation
            new_state_dict['model'][key] = param

    # Save the modified state dict as the new checkpoint
    torch.save(new_state_dict, new_checkpoint_path)
    print(f"Transformed checkpoint has been saved to {new_checkpoint_path}")


if __name__ == "__main__":
    # Example usage
    transform_mha_checkpoint("logs/G_696069.pth", "logs/pretrain_0327_with_yt/G_696069.pth")
