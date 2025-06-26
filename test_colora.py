from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def test_merged_model(
    model_path="./colora_output/colora_final",
    questions=None,
    device="mps",
    max_new_tokens=64,
):
    if questions is None:
        questions = [
            "Thuế trả ra lỗi: -1:Bộ MST, ký hiệu mẫu số, ký hiệu và số hóa đơn không duy nhất_Kiểm tra lại định dạng dữ liệu.",
            "Bạn kiểm tra giúp bảng kê hoá đơn hàng tháng của đơn vị, những hoá đơn điều chỉnh giảm ko trừ đi số tiền khi báo cáo.",
            "Hoá đơn sai số lượng sản phẩm có phải làm hoá đơn thay thế không nhỉ?",
            "Ký hiệu tôi ghi nhầm sang mẫu cũ",
            "Tên khách hàng cần sửa",
            "Ghi sai tên người giao hàng?",
            "ERR:1 trong ImportAndPublishAssignedNo là gì?",
            "Tôi bị lỗi ERR:1 là do đâu?",
            "API trả về ERR:10?",
            "Controller Department cần viết những phương thức nào?",
            "Tôi muốn tạo API mới thì cần làm gì?",
            "Khi import mình chứng từ hệ thống báo lỗi ngày tạo không hợp lệ là sao nhỉ?",
            "View mặc định của hoá đơn máy tính tiền là gì vậy?",
            "Hoá đơn máy tính tiền là gì?",
            "Làm sao biết công ty chưa đăng ký chứng thư số khi gọi GetCertInfo?",
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
