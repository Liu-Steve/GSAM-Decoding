import json

classification = {
        "coding": "Multi-turn Conversation",
        "extraction": "Multi-turn Conversation",
        "humanities": "Multi-turn Conversation",
        "math": "Multi-turn Conversation",
        "reasoning": "Multi-turn Conversation",
        "roleplay": "Multi-turn Conversation",
        "stem": "Multi-turn Conversation",
        "writing": "Multi-turn Conversation",
        "math_reasoning": "Mathematical Reasoning",
        "rag": "Retrieval-augmented Generation",
        "qa": "Question Answering",
        "translation": "Translation",
        "summarization": "Summarization",
    }

def print_accepted_lengths(path):
    model_answers = []
    with open(path, "r") as f:
        for line in f:
            model_answers.append(json.loads(line))

    accepted_lengths = {
        "Multi-turn Conversation": [],
        "Retrieval-augmented Generation": [],
        "Mathematical Reasoning": [],
        "Question Answering": [],
        "Summarization": [],
        "Translation": [],
    }

    for answer in model_answers:
        category = classification[answer["category"]]
        accepted_lengths[category].extend(answer["choices"][0]["accept_lengths"])

    for category, al in accepted_lengths.items():
        if len(al) > 0:
            print(category, sum(al) / len(al))
        else:
            print(category, 0)
    
    mean_accepted_len = sum((sum(al) for al in accepted_lengths.values())) / sum((len(al) for al in accepted_lengths.values()))
    print(f"Mean accepted length {mean_accepted_len}")

def print_token_generation_speed(path):
    model_answers = []
    with open(path, "r") as f:
        for line in f:
            model_answers.append(json.loads(line))

    new_tokens = {
        "Multi-turn Conversation": [],
        "Retrieval-augmented Generation": [],
        "Mathematical Reasoning": [],
        "Question Answering": [],
        "Summarization": [],
        "Translation": [],
    }

    wall_clock_times = {
        "Multi-turn Conversation": [],
        "Retrieval-augmented Generation": [],
        "Mathematical Reasoning": [],
        "Question Answering": [],
        "Summarization": [],
        "Translation": [],
    }

    for answer in model_answers:
        category = classification[answer["category"]]
        new_tokens[category].extend(answer["choices"][0]["new_tokens"])
        wall_clock_times[category].extend(answer["choices"][0]["wall_time"])

    category_sumed_tokens = []
    category_sumed_times = []

    for category in new_tokens.keys():
        category_new_tokens = sum(new_tokens[category])
        category_wall_time = sum(wall_clock_times[category])
        category_sumed_tokens.append(category_new_tokens)
        category_sumed_times.append(category_wall_time)
        print(category, category_new_tokens / category_wall_time, "tok/s")

    print("Average", sum(category_sumed_tokens) / sum(category_sumed_times), "tok/s")

def get_wall_clock_time(path):
    model_answers = []
    with open(path, "r") as f:
        for line in f:
            model_answers.append(json.loads(line))

    classification = {
        "coding": "Multi-turn Conversation",
        "extraction": "Multi-turn Conversation",
        "humanities": "Multi-turn Conversation",
        "math": "Multi-turn Conversation",
        "reasoning": "Multi-turn Conversation",
        "roleplay": "Multi-turn Conversation",
        "stem": "Multi-turn Conversation",
        "writing": "Multi-turn Conversation",
        "math_reasoning": "Mathematical Reasoning",
        "rag": "Retrieval-augmented Generation",
        "qa": "Question Answering",
        "translation": "Translation",
        "summarization": "Summarization",
    }

    wall_clock_time = {
        "Multi-turn Conversation": {},
        "Retrieval-augmented Generation": {},
        "Mathematical Reasoning": {},
        "Question Answering": {},
        "Summarization": {},
        "Translation": {},
    }

    for answer in model_answers:
        category = classification[answer["category"]]
        ques_id = answer["question_id"]
        wall_clock_time[category][ques_id] = sum(
            answer["choices"][0]["wall_time"])

    return wall_clock_time


def get_wall_clock_speedup_from_file(path, baseline_path, exclude_ques_id=None):
    wall_clock_time = get_wall_clock_time(path)
    baseline_wall_clock_time = get_wall_clock_time(baseline_path)

    speedups = {}
    for category in wall_clock_time.keys():
        total_time = 0
        base_total_time = 0

        for ques_id in wall_clock_time[category].keys():
            if exclude_ques_id is not None and ques_id in exclude_ques_id:
                continue

            total_time += wall_clock_time[category][ques_id]
            base_total_time += baseline_wall_clock_time[category][ques_id]

        speedup = base_total_time / total_time
        speedups[category] = speedup

    speedups["Average"] = sum(speedups.values()) / len(speedups)
    return speedups


def print_wall_clock_speedup(path, baseline_path):
    dic = get_wall_clock_speedup_from_file(path, baseline_path)
    for cat, speedup in dic.items():
        print(cat, speedup)


print("### Dual Table Setup ###")
print("# Mean Accepted Length #")
print_accepted_lengths("plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-float16.jsonl")
print("# Token Generation Speed #")
print_token_generation_speed("plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-float16.jsonl")
print("# Speedup #")
print_wall_clock_speedup(
    path="plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-float16.jsonl",
    baseline_path="plot_data/shotgun_partial_answer/vicuna-7b-v1.3-baseline-float16.jsonl"
)
print()

print("### No Frozen ###")
print("# Mean Accepted Length #")
print_accepted_lengths("plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-no-frozen-float16.jsonl")
print("# Token Generation Speed #")
print_token_generation_speed("plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-no-frozen-float16.jsonl")
print("# Speedup #")
print_wall_clock_speedup(
    path="plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-no-frozen-float16.jsonl",
    baseline_path="plot_data/shotgun_partial_answer/vicuna-7b-v1.3-baseline-float16.jsonl"
)
print()

print("### Only Frozen ###")
print("# Mean Accepted Length #")
print_accepted_lengths("plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-only-frozen-float16.jsonl")
print("# Token Generation Speed #")
print_token_generation_speed("plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-only-frozen-float16.jsonl")
print("# Speedup #")
print_wall_clock_speedup(
    path="plot_data/shotgun_partial_answer/vicuna-7b-v1.3-shotgun-only-frozen-float16.jsonl",
    baseline_path="plot_data/shotgun_partial_answer/vicuna-7b-v1.3-baseline-float16.jsonl"
)
print()