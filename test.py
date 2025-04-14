from vllm import LLM, SamplingParams

llm = LLM(model="nvidia/Llama-3.3-70B-Instruct-FP4",
          tensor_parallel_size=2,
          trust_remote_code=True,
          enforce_eager=True,
          quantization="nvfp4")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Prompt: 'Hello, my name is', Generated text: ' Diane.\nWelcome to our 2023-24 School Year!\nI am thrilled'
# Prompt: 'The president of the United States is', Generated text: ' at the top of the pyramid. For the protection of the president, the Secret'
# Prompt: 'The capital of France is', Generated text: ' a must-visit destination for any traveler. The city is steeped in history'
# Prompt: 'The future of AI is', Generated text: ' here, and it’s not just for techies\nArtificial intelligence (AI'