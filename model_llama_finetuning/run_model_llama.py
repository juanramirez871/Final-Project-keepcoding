import json
from pathlib import Path
from mlx_lm import load, generate


BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_llama_finetuning" / "llama_ventas_colombiano_mlx_q8"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "Eres un vendedor colombiano amigable y cercano. "
    "Hablas con palabras típicas como 'parce', 'mano', 'bacano', 'chévere', 'qué más', "
    "'ahorita', 'qué nota', 'de una', 'tranquilo', 'todo bien'. "
    "Tu tono debe ser 100% colombiano, natural y cercano.\n\n"

    "--- FLUJO DE HERRAMIENTAS ---\n"

    "1. Para buscar productos usa la herramienta 'get_products'. "
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"get_products\", \"query\": \"término de búsqueda\"}\n\n"

    "2. Para crear una orden usa 'create_order'. "
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"create_order\", \"product_name\": \"nombre\", "
    "\"quantity\": cantidad, \"price\": precio, \"customer_phone\": telefono}\n\n"

    "3. NO escribas nada más que el JSON cuando uses herramientas.\n"
)


model, tokenizer = load(MODEL_PATH)

def get_internal_knowledge(query, top_k=3):
    return f"Resultados de productos para: {query}"

def create_order(product_name, quantity, price, customer_phone):
    return f"Orden creada: {quantity} x {product_name} por {price} para {customer_phone}"

def generate_response_from_model(
    user_message,
    model,
    tokenizer,
    max_tokens=DEFAULT_MAX_TOKENS,
    top_k=3,
):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )

    text = response.strip()
    print(f"Model output: {text}")

    try:
        tool_call = json.loads(text)

        if isinstance(tool_call, dict) and "tool" in tool_call:

            tool_name = tool_call["tool"]
            if tool_name == "get_products":

                query_tool = tool_call.get("query", "")
                print(f"Tool call detectado: get_products('{query_tool}')")

                knowledge = get_internal_knowledge(query_tool, top_k=top_k)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "system",
                        "content": f"RESULTADO DE LA HERRAMIENTA:\n{knowledge}",
                    }
                )

            elif tool_name == "create_order":

                print("Tool call detectado: create_order")
                order_status = create_order(
                    product_name=tool_call.get("product_name"),
                    quantity=tool_call.get("quantity"),
                    price=tool_call.get("price"),
                    customer_phone=tool_call.get("customer_phone"),
                )

                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "system",
                        "content": f"RESULTADO DE LA HERRAMIENTA:\n{order_status}",
                    }
                )

            chat_prompt_final = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            final_response = generate(
                model,
                tokenizer,
                prompt=chat_prompt_final,
                max_tokens=max_tokens,
                verbose=False,
            )

            text = final_response.strip()
            print(f"Respuesta final: {text}")

            return {"response": text, "status": "success"}

    except json.JSONDecodeError:
        print("El modelo respondió texto normal.")

    return {"response": text, "status": "success"}


if __name__ == "__main__":
    result = generate_response_from_model(
        "Hola, ¿cómo estás?",
        model,
        tokenizer
    )