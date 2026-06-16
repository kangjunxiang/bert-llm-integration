import json

INSTRUCTION = """您是医学实体识别机器人。您的任务是将输入文本中的实体类别按要求输出。您将仅使用预定义的实体类别进行响应。 请勿提供额外的解释或注释。
需提取的实体分类分为以下几类：
 dru : 药物 
 bod :身体部位 
 sym :临床表现、症状、体征 
 mic :微生物类 
 equ :医疗设备、检查设备、治疗设备 
 ite : 医学检验项目 
 pro :医疗程序、检查程序、治疗或预防程序 
 dep :科室 
 dis :疾病、疾病名或综合征名、受伤或中毒、器官或细胞损伤 
 在输出内容中，首先明确指出实体的类型，随后输出原始文本序列。每行应该按实体类型分组，每行只标记当前类型的实体。在输出原始文本序列时，通过使用特殊符号"["和"]"对其中的实体进行标注，其中"["表示实体的起始位置，"]"表示实体的终止位置。
 例如：
 bod :对儿童[SARST细胞亚群]的研究表明，与成人SARS相比，儿童[细胞]下降不明显，证明上述推测成立。
 dis :对儿童SARST细胞亚群的研究表明，与[成人SARS]相比，儿童[细胞]下降不明显，证明上述推测成立。
 sym :对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童[细胞下降]不明显，证明上述推测成立。"""

def mark_entities_by_type(text, entities, target_type):
    """Mark only entities of the given type; other entities are left as plain text."""
    marked_text = text
    type_entities = [e for e in entities if e['type'] == target_type]
    
    offset = 0
    for ent in type_entities:
        start = ent['start_idx'] + offset
        end = ent['end_idx'] + offset
        marked_text = marked_text[:start] + '[' + marked_text[start:end] + ']' + marked_text[end:]
        offset += 2
    
    return marked_text

def convert_to_llm_format(data):
    """Convert CMeEE data into the LLM training format."""
    results = []
    for item in data:
        text = item['text']
        entities = item['entities']
        
        entity_types = sorted(set(e['type'] for e in entities))
        
        outputs = []
        for etype in entity_types:
            marked_text = mark_entities_by_type(text, entities, etype)
            outputs.append(f"{etype} :{marked_text}")
        
        result_item = {
            "instruction": INSTRUCTION,
            "input": text,
            "output": "\n".join(outputs)
        }
        results.append(result_item)
    
    return results

def process_cmee_file(input_path, output_path):
    """Process a CMeEE data file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    llm_data = convert_to_llm_format(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete: {len(llm_data)} records in total")
    print(f"Output file: {output_path}")

if __name__ == '__main__':
    input_file = r'data\CMeEE-V2\CMeEE-V2_train.json'
    output_file = r'data\CMeEE-V2\CMeEE-V2_llm_mark_train.json'
    #input_file = r'data\CMeEE-V2\CMeEE-V2_dev.json'
    #output_file = r'data\CMeEE-V2\CMeEE-V2_llm_mark_dev.json'
    process_cmee_file(input_file, output_file)