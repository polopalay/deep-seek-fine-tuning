from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "deepseek_finetuned_model"
qa_pipeline = pipeline("text-generation", model=model_path, tokenizer=model_path)

prompt = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"

result = qa_pipeline(prompt, return_full_text=False)
print(result)
