# import json
# import shutil
# import sys

# from allennlp.commands import main
# from pathlib import Path

# config_file = '/home/zijiao/research/Thesis/image_srl/bound_image_srl.jsonnet'

# # overrides = json.dumps({"trainer" : {"cuda_device" : -1}})

# serialization_dir = "/tmp/debugger_train"

# shutil.rmtree(serialization_dir, ignore_errors=True)



# sys.argv = [
#     "allennlp",
#     "train",
#     config_file,
#     "-s", serialization_dir,
#     "--include-package", "utils_srl",
# ]

# main()

