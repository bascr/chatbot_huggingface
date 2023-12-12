from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

model_name = "./my_autotrain_llm"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, TOKENIZERS_PARALLELISM=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

message_list = []
response_list = []

def chat(message, history):
    message_list.append(message)
    input_ids = tokenizer.encode(message, return_tensors="pt")
    output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens = 200)
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response_list.append(predicted_text)
    return response_list[-1]

demo_chatbot = gr.ChatInterface(chat, title="Instruction Chatbot", description="Enter an instruction to start chatting. Because of the limited computing power resources it could take a significant amount of time to get the response.")

demo_chatbot.queue().launch(share=True)