from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def test_merged_model(
    model_path="./colora_output/merged_model_final",
    questions=None,
    device="mps",
    max_new_tokens=64,
):
    if questions is None:
        questions = [
            "Thay thế hóa đơn có cần báo cáo lại không?",
            "Hóa đơn thay thế có cần thời gian cụ thể không?",
            "Tôi muốn kiểm tra lại trước khi gửi",
            "Ký hiệu tôi ghi nhầm sang mẫu cũ",
            "Tên khách hàng cần sửa",
            "Ghi sai tên người giao hàng?",
            "Lỗi ERR:1 là gì?",
            "Lỗi ERR:2 là gì?",
            "Lỗi ERR:3 là gì?",
            "Lỗi ERR:4 là gì?",
            "Lỗi ERR:5 là gì?",
            "Sai hoá đơn cần báo cho người mua không?",
            "Hàm ImportAndPublishInv dùng để làm gì trong hệ thống?",
            "Khi nào phải điều chỉnh chứ không thay thế?",
        ]

    # Load mô hình đã merge và tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for q in questions:
        chat = [{"role": "user", "content": q}]
        input_ids = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = full_output.replace(prompt_text, "").strip()
        answer = answer.split(".")[0].strip()

        print(f"Q: {q}\nA: {answer}\n{'-'*60}")


test_merged_model()
