const char* mlperf_conf =
"# The format of this config file is 'key = value'.\n"
"# The key has the format 'model.scenario.key'. Value is mostly int64_t.\n"
"# Model maybe '*' as wildcard. In that case the value applies to all models.\n"
"# All times are in milli seconds\n"
"\n"
"# Set performance_sample_count for each model.\n"
"# User can optionally set this to higher values in user.conf.\n"
"resnet50.*.performance_sample_count_override = 1024\n"
"ssd-mobilenet.*.performance_sample_count_override = 256\n"
"retinanet.*.performance_sample_count_override = 64\n"
"bert.*.performance_sample_count_override = 10833\n"
"dlrm.*.performance_sample_count_override = 204800\n"
"dlrm-v2.*.performance_sample_count_override = 204800\n"
"rnnt.*.performance_sample_count_override = 2513\n"
"gptj.*.performance_sample_count_override = 13368\n"
"llama2-70b.*.performance_sample_count_override = 24576\n"
"stable-diffusion-xl.*.performance_sample_count_override = 5000\n"
"# set to 0 to let entire sample set to be performance sample\n"
"3d-unet.*.performance_sample_count_override = 0\n"
"\n"
"# Set seeds. The seeds will be distributed two weeks before the submission.\n"
"*.*.qsl_rng_seed = 3066443479025735752\n"
"*.*.sample_index_rng_seed = 10688027786191513374\n"
"*.*.schedule_rng_seed = 14962580496156340209\n"
"# Set seeds for TEST_05. The seeds will be distributed two weeks before the submission.\n"
"*.*.test05_qsl_rng_seed = 16799458546791641818\n"
"*.*.test05_sample_index_rng_seed = 5453809927556429288\n"
"*.*.test05_schedule_rng_seed = 5435552105434836064\n"
"\n"
"\n"
"*.SingleStream.target_latency_percentile = 90\n"
"*.SingleStream.min_duration = 600000\n"
"\n"
"*.MultiStream.target_latency_percentile = 99\n"
"*.MultiStream.samples_per_query = 8\n"
"*.MultiStream.min_duration = 600000\n"
"*.MultiStream.min_query_count = 662\n"
"retinanet.MultiStream.target_latency = 528\n"
"\n"
"# 3D-UNet uses equal issue mode because it has non-uniform inputs\n"
"3d-unet.*.sample_concatenate_permutation = 1\n"
"\n"
"# LLM benchmarks have non-uniform inputs and outputs, and use equal issue mode for all latency scenario\n"
"gptj.*.sample_concatenate_permutation = 1\n"
"llama2-70b.*.sample_concatenate_permutation = 1\n"
"mixtral-8x7b.*.sample_concatenate_permutation = 1\n"
"\n"
"*.Server.target_latency = 10\n"
"*.Server.target_latency_percentile = 99\n"
"*.Server.target_duration = 0\n"
"*.Server.min_duration = 600000\n"
"resnet50.Server.target_latency = 15\n"
"retinanet.Server.target_latency = 100\n"
"bert.Server.target_latency = 130\n"
"dlrm.Server.target_latency = 60\n"
"dlrm-v2.Server.target_latency = 60\n"
"rnnt.Server.target_latency = 1000\n"
"gptj.Server.target_latency = 20000\n"
"stable-diffusion-xl.Server.target_latency = 20000\n"
"# Llama2-70b benchmarks measures token latencies\n"
"llama2-70b.*.use_token_latencies = 1\n"
"mixtral-8x7b.*.use_token_latencies = 1\n"
"# gptj benchmark infers token latencies\n"
"gptj.*.infer_token_latencies = 1\n"
"gptj.*.token_latency_scaling_factor = 69\n"
"# Only ttft and tpot are tracked for the llama2-70b & mixtral-8x7B benchmark therefore target_latency = 0\n"
"llama2-70b.Server.target_latency = 0\n"
"llama2-70b.Server.ttft_latency = 2000\n"
"llama2-70b.Server.tpot_latency = 200\n"
"\n"
"mixtral-8x7b.Server.target_latency = 0\n"
"mixtral-8x7b.Server.ttft_latency = 2000\n"
"mixtral-8x7b.Server.tpot_latency = 200\n"
"\n"
"*.Offline.target_latency_percentile = 90\n"
"*.Offline.min_duration = 600000\n"
"\n"
"# In Offline scenario, we always have one query. But LoadGen maps this to\n"
"# min_sample_count internally in Offline scenario. If the dataset size is larger \n"
"# than 24576 we limit the min_query_count to 24576 and otherwise we use \n"
"# the dataset size as the limit\n"
"\n"
"resnet50.Offline.min_query_count = 24576\n"
"retinanet.Offline.min_query_count = 24576\n"
"dlrm-v2.Offline.min_query_count = 24576\n"
"bert.Offline.min_query_count = 10833\n"
"gptj.Offline.min_query_count = 13368\n"
"rnnt.Offline.min_query_count = 2513\n"
"3d-unet.Offline.min_query_count = 43\n"
"stable-diffusion-xl.Offline.min_query_count = 5000\n"
"llama2-70b.Offline.min_query_count = 24576\n"
"mixtral-8x7b.Offline.min_query_count = 15000\n"
"\n"
"# These fields should be defined and overridden by user.conf.\n"
"*.SingleStream.target_latency = 10\n"
"*.MultiStream.target_latency = 80\n"
"*.Server.target_qps = 1.0\n"
"*.Offline.target_qps = 1.0\n"
"";
