import os
GPUS = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import PeftModel
import time
import math
import threading
import json

torch.cuda.empty_cache()


class NerEtl:
    def __init__(self):
        self.base_model_path = '/data/yh/HF-LLM-models/hub/models--shenzhi-wang--Llama3-8B-Chinese-Chat/snapshots/f25f13cb2571e70e285121faceac92926b51e6f5'
        self.lora_model_path = '/data/yh/LLM-train-saves/Llama3-8B-Chinese-Chat-mark/lora/train_lora_5epochs_indexU'
        self.tokenizer_path = '/data/yh/HF-LLM-models/hub/models--shenzhi-wang--Llama3-8B-Chinese-Chat/snapshots/f25f13cb2571e70e285121faceac92926b51e6f5'
        self.load_in_4bit = False
        self.load_in_8bit = False
        self.with_prompt = True

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

        # 多答案设置
        self.answer_num = 5          # 期望返回的答案数量
        self.num_beams = 5          # 必须 >= answer_num
        self.load_type = torch.float16

        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_k=40,
            top_p=1,
            do_sample=False,
            num_beams=self.num_beams,   # 同步为5
            repetition_penalty=1.1,
            max_new_tokens=2048
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

        base_model = LlamaForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            quantization_config=quantization_config
        )

        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(self.tokenizer)
        if model_vocab_size != tokenizer_vocab_size:
            base_model.resize_token_embeddings(tokenizer_vocab_size)

        self.model = PeftModel.from_pretrained(
            base_model,
            self.lora_model_path,
            torch_dtype=self.load_type,
            device_map='auto',
        ).half()
        self.model.eval()

    def generate_prompt(self, instruction):
        user_content = f"{self.system_prompt}\n{instruction}"
        prompt = (
            "<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    def _clean_response(self, text: str) -> str:
        """提取 assistant 后的内容，并移除末尾的 <|eot_id|> 等标记"""
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_marker in text:
            result = text.split(assistant_marker, 1)[-1]
        else:
            # 回退处理
            result = text.split('assistant', 1)[-1] if 'assistant' in text else text

        # 循环去除末尾空白和结束符
        while True:
            result = result.rstrip()
            if result.endswith("<|eot_id|>"):
                result = result[:-len("<|eot_id|>")]
            elif result.endswith("<|end_of_text|>"):
                result = result[:-len("<|end_of_text|>")]
            else:
                break
        return result.strip()

    def _compute_confidence_per_sequence(self, generation_output, seq_idx):
        """基于生成步熵的置信度（针对单个序列）"""
        scores = generation_output.scores
        generated_len = len(scores)
        if generated_len == 0:
            return 1.0

        total_entropy = 0.0
        for logits in scores:
            probs = F.softmax(logits[seq_idx], dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -torch.sum(probs * log_probs).item()
            total_entropy += entropy

        avg_entropy = total_entropy / generated_len
        confidence = math.exp(-avg_entropy)
        return round(confidence, 4)

    def inference(self, input_data, label):
        process_id = os.getpid()
        thread_id = threading.get_ident()
        print(f"===== 开始处理数据 ===== process_id:{process_id}, thread_id:{thread_id}")

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
                num_return_sequences=self.answer_num,    # 返回多条候选序列
                return_dict_in_generate=True,
                output_scores=True,
            )

            ner_models = []
            use_sequence_scores = (
                hasattr(generation_output, "sequences_scores") and
                generation_output.sequences_scores is not None
            )

            num_seq = generation_output.sequences.shape[0]   # 通常等于 answer_num
            for i in range(num_seq):
                raw_output = self.tokenizer.decode(
                    generation_output.sequences[i],
                    skip_special_tokens=False
                )
                result = self._clean_response(raw_output)

                if use_sequence_scores:
                    confidence = round(math.exp(generation_output.sequences_scores[i].item()), 4)
                else:
                    confidence = self._compute_confidence_per_sequence(generation_output, i)

                ner_models.append({
                    "predict": result,
                    "confidence": confidence
                })

            # 按置信度降序排列
            ner_models.sort(key=lambda x: x["confidence"], reverse=True)

            end_time = time.time()
            print(f"推理时间: {end_time-start_time:.2f}s")

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
            print(f"===== 结束推理 =====")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ner_etl = NerEtl()
    ner_etl.load_model()
    file_path_data_evl = "data/CMeEE-V2/llm/CMeEE-V2_llm_mark_test.json"
    samplejsonArry = []
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
        index = index + 1
        if index % 100 == 0:
            print(f"已处理 {index} 条数据")
        lines.append(output)

    with open("evl_f1/llm/lla_ner_Llama3-8B-confidence_beams5.jsonl", "w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    print("处理完成，结果已保存。")