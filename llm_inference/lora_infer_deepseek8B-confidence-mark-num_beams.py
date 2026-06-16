import os
GPUS = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import PeftModel
import time
import math
import threading
import json

torch.cuda.empty_cache()


class NerEtl:
    def __init__(self, use_lora=True):
        self.base_model_path = '/data/yh/HF-LLM-models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1'
        # Only required when loading the LoRA adapter
        self.lora_model_path = '/data/yh/LLM-train-saves/DeepSeek-R1-Distill-Llama-8B-mark/lora/train_lora_5epochs_indexU' if use_lora else None
        self.tokenizer_path = '/data/yh/HF-LLM-models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1'
        self.load_in_4bit = False
        self.load_in_8bit = False
        self.load_type = torch.float16
        self.use_lora = use_lora

        # Number of candidate answers
        self.answer_num = 5
        self.num_beams = 5            # Must be >= answer_num

        # Mark-style system prompt (Chinese by design — used as the LLM input)
        self.system_prompt = (
            "您是医学实体识别机器人。您的任务是将输入文本中的实体类别按要求输出。"
            "您将仅使用预定义的实体类别进行响应。 请勿提供额外的解释或注释。\n"
            "需提取的实体分类分为以下几类：\n"
            " dru : 药物 \n bod :身体部位 \n sym :临床表现、症状、体征 \n"
            " mic :微生物类 \n equ :医疗设备、检查设备、治疗设备 \n"
            " ite : 医学检验项目 \n pro :医疗程序、检查程序、治疗或预防程序 \n"
            " dep :科室 \n dis :疾病、疾病名或综合征名、受伤或中毒、器官或细胞损伤 \n"
            " 在输出内容中，首先明确指出实体的类型，随后输出原始文本序列。"
            "每行应该按实体类型分组，每行只标记当前类型的实体。在输出原始文本序列时，"
            "通过使用特殊符号\"[\"和\"]\"对其中的实体进行标注，其中\"[\"表示实体的起始位置，"
            "\"]\"表示实体的终止位置。\n 例如：\n"
            " bod :对儿童[SARST细胞亚群]的研究表明，与成人SARS相比，儿童[细胞]下降不明显，证明上述推测成立。\n"
            " dis :对儿童SARST细胞亚群的研究表明，与[成人SARS]相比，儿童[细胞]下降不明显，证明上述推测成立。\n"
            " sym :对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童[细胞下降]不明显，证明上述推测成立。"
        )

        self.template = "<｜begin▁of▁sentence｜><｜User｜>{system_prompt}\n{instruction}<｜Assistant｜>"

        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=self.num_beams,
            max_new_tokens=1024
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, legacy=True)

        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_compute_dtype=self.load_type,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            quantization_config=quantization_config
        )

        # Align the vocabulary sizes between the base model and the tokenizer
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(self.tokenizer)
        if model_vocab_size != tokenizer_vocab_size:
            base_model.resize_token_embeddings(tokenizer_vocab_size)

        # Load the LoRA adapter when explicitly requested
        if self.use_lora and self.lora_model_path is not None:
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_model_path,
                torch_dtype=self.load_type,
                device_map='auto',
            ).half()
            print("LoRA adapter loaded, using the fine-tuned model.")
        else:
            self.model = base_model.half()
            print("No LoRA loaded, using the pre-trained base model.")

        self.model.eval()

    def generate_prompt(self, instruction):
        return self.template.format(
            system_prompt=self.system_prompt,
            instruction=instruction
        )

    def inference(self, input_data, label):
        process_id = os.getpid()
        thread_id = threading.get_ident()
        print(f"===== Start processing data ===== process_id:{process_id}, thread_id:{thread_id}")

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model has not been loaded. Call load_model() first.")

        try:
            start_time = time.time()
            input_text = self.generate_prompt(input_data)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            generation_output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs['attention_mask'],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                generation_config=self.generation_config,
                num_return_sequences=self.answer_num,
                return_dict_in_generate=True,
                output_scores=True,
            )

            res_original = []
            for s, score in zip(generation_output.sequences, generation_output.sequences_scores):
                pv = round(math.exp(score), 4)
                output = self.tokenizer.decode(s, skip_special_tokens=True)
                result = output.split('<｜Assistant｜>', 1)[-1].strip()
                # Strip out any <think>...</think> reasoning blocks
                if '<think>' in result:
                    result = result.split('<think>', 1)[-1].strip()
                if '</think>' in result:
                    result = result.split('</think>', 1)[-1].strip()
                res_original.append({"predict": result, "confidence": pv})

            # Sort by confidence (descending) and keep the top `answer_num`
            sorted_res = sorted(res_original, key=lambda x: x["confidence"], reverse=True)
            ner_models = [{"predict": res["predict"], "confidence": res["confidence"]} for res in sorted_res[:self.answer_num]]

            end_time = time.time()
            print(f"Inference time: {end_time-start_time:.2f}s")

            return {
                "text": input_data,
                "ner_models": ner_models,
                "msg": "ok",
                "label": label
            }

        except Exception as e:
            return {
                "text": input_data,
                "ner_models": [],
                "msg": f"error:{e}",
                "label": label
            }
        finally:
            print(f"===== Inference finished =====")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # ========== Global switch: True → load LoRA (fine-tuned), False → base model only ==========
    USE_LORA = False   # Flip here to switch between fine-tuned and base modes

    # Instantiate and load the selected model (only one is loaded at a time)
    ner_etl = NerEtl(use_lora=USE_LORA)
    ner_etl.load_model()

    # Load the evaluation data
    file_path_data_evl = "data/CMeEE-V2/llm/CMeEE-V2_llm_mark_test.json"
    with open(file_path_data_evl, 'r', encoding='utf-8') as f:
        samplejsonArry = json.load(f)

    lines = []
    index = 1
    for samplejson in samplejsonArry:
        input_data_str = samplejson["input"]
        label_str = samplejson["output"]
        output = ner_etl.inference(input_data=input_data_str, label=label_str)
        print(output)
        print(index)
        index += 1
        if index % 100 == 0:
            print(f"Processed {index} records")
        lines.append(output)

    # Name the output file according to the selected mode
    mode = "lora" if USE_LORA else "base"
    out_file = f"deepseek8B_ner_confidence_beams_5_{mode}.jsonl"
    with open(out_file, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(f"{json.dumps(line, ensure_ascii=False)}\n")
    print(f"Processing complete, results saved to {out_file}.")