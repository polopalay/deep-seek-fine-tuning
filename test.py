from transformers import pipeline

qa_pipeline = pipeline("text-generation", model="deepseek_finetuned_model")

question = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"
context = "Lâm Bá Kiệt"

result = print(result[0]["generated_text"])
print(result)
