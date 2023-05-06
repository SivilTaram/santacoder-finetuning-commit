import copy
import torch


def transfer_avg_weights_from_store(avg_model_paths, final_save_model) -> None:
    # add pytorch_model.bin to the path
    avg_model_paths = [path + "/pytorch_model.bin" for path in avg_model_paths]
    # load all models
    param_store = []
    for path in avg_model_paths:
        parameters = torch.load(path, map_location=torch.device('cpu'))
        param_store.append(parameters)
    # average all parameters
    avg_params = copy.deepcopy(param_store[0])
    for params in param_store[1:]:
        for key in avg_params.keys():
            # if key does not end with attn.bias
            if not key.endswith("attn.bias"):
                avg_params[key].add_(params[key])
    total_len = len(param_store)
    for key in avg_params.keys():
        if not key.endswith("attn.bias"):
            avg_params[key].div_(total_len)

    # dump the averaged parameters
    torch.save(avg_params, final_save_model + "/pytorch_model.bin")


if __name__ == '__main__':
    transfer_avg_weights_from_store(["/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing/checkpoint-189000",
                                     "/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing/checkpoint-192000",
                                    "/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing/checkpoint-195000",
                                     "/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing/checkpoint-198000"],
                                    "/dev/cache/qian/checkpoints/santacoder-pjj-2048-unpacking-lawa/checkpoint-3000")

