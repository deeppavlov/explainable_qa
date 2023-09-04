import time
import logging
from typing import List, Tuple, Optional, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    )

logger = logging.getLogger(__name__)


instruction_with_q = """
A chat between a curious human and an artificial intelligence assistant.
The assistant's job is to answer the given question using only the information provided in the RDF triplet format. The assistant's answer should be in a human-readable format, with proper sentences and grammar and should be concise and short.
The RDF triplets will be provided in triplets, where triplets are always in the (subject, relation, object) format and are separated by a semicolon. The assistant should understand that if multiple triplets are provided, the answer to the question should use all of the information from triplets and make aggregation. The assistant MUST NOT add any additional information, beside form the one proveded in the triplets.
The assistant should try to reply as short as possible, and perform counting or aggregation operations over triplets by himself when necessary.
"""

instruction_wo_q = """
A chat between a curious human and an artificial intelligence assistant.
The assistant's job is convert the provided input in RDF triplet format into human-readable text format, with proper sentences and grammar. The triplets are always in the (subject, relation, object) format, where each triplet is separated by a semicolon. The assistant should understand that if multiple triplets are provided, the generated human-readable text should use all of the information from input. The assistant MUST NOT add any additional information, beside form the one proveded in the input.
"""

instruction_zero_shot_wo_q = """
Rewrite the following triplets to human-readable full sentence in natural language.
Triplets: """

instruction_zero_shot_with_q = """
Generate long sentence answer to the following question using the provided RDF triplets.
"""


history_with_q = [
        ("Human", "Question: Is Essex the Ceremonial County of West Tilbury? Triplets: ('West Tilbury', 'Ceremonial County', 'Essex');"),
        ("Assistant", "Essex is the Ceremonial County of West Tributary"),
        ("Human", "Question: What nation is Hornito located in, where Jamie Bateman Cayn died too? Triplets: ('Jaime Bateman Cayón', 'death place', 'Panama'); ('Hornito, Chiriquí', 'country', 'Panama');"),
        ("Assistant", "Hornito, Chiriquí is located in Panama, where Jaime Bateman Cayón died."),
        ("Human", "Question: Who are the shareholder of the soccer club for whom Steve Holland plays? Triplets: ('Steve Holland', 'current club', 'Chelsea F.C.'); ('Chelsea F.C.', 'owner', 'Roman Abramovich');"),
        ("Assistant", "Roman Abramovich owns Chelsea F.C., where Steve Holland plays."),
        ("Human", "Question: Who is the chancellor of Falmouth University? Triplets: ('Falmouth University', 'chancellor', 'Dawn French');"),
        ("Assistant", "The chancellor of the Falmouth University is Dawn French.")
    ]


history_wo_q = [
        ("Human", "('West Tilbury', 'Ceremonial County', 'Essex');"),
        ("Assistant", "Essex is the Ceremonial County of West Tributary"),
        ("Human", "('Jaime Bateman Cayón', 'death place', 'Panama'); ('Hornito, Chiriquí', 'country', 'Panama');"),
        ("Assistant", "Hornito, Chiriquí is located in Panama, where Jaime Bateman Cayón died."),
        ("Human", "('Steve Holland', 'current club', 'Chelsea F.C.'); ('Chelsea F.C.', 'owner', 'Roman Abramovich');"),
        ("Assistant", "Roman Abramovich owns Chelsea F.C., where Steve Holland plays."),
        ("Human", "('Falmouth University', 'chancellor', 'Dawn French');"),
        ("Assistant", "The chancellor of the Falmouth University is Dawn French.")

    ]


mapping = {
    "fewshot_question": (instruction_with_q, history_with_q),
    "fewshot_triplets": (instruction_wo_q, history_wo_q),
    "zeroshot_question": (instruction_zero_shot_with_q, None),
    "zeroshot_triplets": (instruction_zero_shot_wo_q, None),
}


def prepare_input(style: str, triplets: List[List[str]], question: str=None) ->  str:
    linearized = ""
    for t in triplets:
        linearized += f"{str(tuple(t))};"
    if question and "question" in style:
        input_text = f"Question: {question.strip()} Triplets: {linearized}"
    else:
        input_text = linearized
    return input_text


def make_prompt(
    instruction: str,
    roles: List[str],
    curr_input: str,
    history: List[Tuple[str, str]]=None,
    sep_toks: List[str]=None,
    model_type: str="gptj"
) -> str:
    if not history:
        ret = f"{instruction}{curr_input}\nResponse: "
    elif "vicuna" in model_type:
        ret = instruction
        for i, (role, message) in enumerate(history):
            ret += roles[i % 2] + ": " + message + sep_toks[i % 2]
        ret += roles[0] + ": " + curr_input + sep_toks[0] + roles[1] + ": \n"
    elif "pythia" in model_type:
        sep_tok = sep_toks[0]
        ret = instruction
        for i, (role, message) in enumerate(history):
            ret += roles[i % 2] + ": " + message + sep_tok
        ret += roles[0] + ": " + curr_input + sep_tok + roles[1] + ": "
    else:
        ret = instruction
        for i, (role, message) in enumerate(history):
            ret += roles[i % 2] + ": " + message + "\n"
        ret += roles[0] + ": " + curr_input + "\n" + roles[1] + ":"
    return ret


def get_prompt(style, params, model_type, triplets, question=None):
    if question and "question" not in style:
        style = style.replace("questions", "triplets")
    instruction, history = mapping[style]
    curr_input = prepare_input(style, triplets, question)
    prompt = make_prompt(instruction=instruction, roles=params['roles'], curr_input=curr_input, history=history, sep_toks=params['sep_toks'], model_type=model_type)
    return prompt


def generate_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    params: Dict[str, str],
    triplets: List[List[str]], 
    question: str=None,
    style: str="fewshot_triplets",
    generation_config: GenerationConfig=None,
):

    if params["stop_str"]:
        logger.info(f"stopping criteria ON: [{params['stop_str']}]")
        stop_token_ids = tokenizer.convert_tokens_to_ids([params['stop_str']])
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        stop = StopOnTokens()

    instruction = get_prompt(style, params, params["type"], triplets, question)
    input_ids = tokenizer(instruction, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    
    generate_kwargs = dict(
        input_ids=input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    with torch.no_grad():
        outputs = model.generate(
            **generate_kwargs,
            generation_config=generation_config,
        )

    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    for tok in tokenizer.additional_special_tokens+[tokenizer.eos_token]:
        instruction = instruction.replace(tok, '')
    response = response[len(instruction): ] .strip("\n;:# ")
    return response
